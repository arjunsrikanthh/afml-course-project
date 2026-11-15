"""Independently audit the frozen AFML study, without importing its engine.

Only pandas, NumPy and the standard library are used. No fitting, model loading,
prediction generation, tuning or bootstrap rerun occurs here. Run after the
complete Kaggle output has been downloaded::

    python scripts/verify_final_study.py --root /path/to/afml_final --output /path/to/audit

The original extracted release is found in this checkout's delivery directory,
or can be supplied with --source. Both verification.json and README.md are
written even when a critical check fails; verify_study then raises AuditFailure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd


METHODS = (
    "persistence", "seasonal7", "mean7", "ridge_local", "histgb_local", "randomforest_local",
    *(f"{a}_{m}" for a in ("gru", "lstm", "tcn", "transformer", "mlp")
      for m in ("local", "pooled_zero", "finetune")), "reptile_gru", "fomaml_gru",
)
TARGETS = ("Amaravati", "Guwahati", "Ahmedabad", "Pithampur", "Ludhiana", "Udaipur")
DEVELOPMENT = ("Chandigarh", "Dewas", "Jalandhar", "Ajmer")
SEEDS, BUDGETS, HORIZONS = (17, 43, 71), (30, 90, 180), (1, 2, 3)
PERIODS = {"secondary_2024": 2024, "test_2025": 2025}
CUTOFF = pd.Timestamp("2023-12-31")
REFERENCE_KEY = ["period", "city", "station_id", "date", "horizon"]
CASE_KEY = ["period", "city", "station_id", "budget", "horizon", "seed", "method"]
CITY_KEY = [c for c in CASE_KEY if c != "seed"]
MACRO_KEY = ["period", "budget", "horizon", "method"]
ATOL = 1e-7


class AuditFailure(RuntimeError):
    """The completed audit contains at least one critical failure."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def columns(frame: pd.DataFrame, needed: list[str] | tuple[str, ...], label: str) -> None:
    require(set(needed).issubset(frame), f"{label}: missing columns {sorted(set(needed) - set(frame))}")


def dates(frame: pd.DataFrame, names: list[str], label: str) -> None:
    for name in names:
        parsed = pd.to_datetime(frame[name], errors="raise")
        require(parsed.notna().all(), f"{label}.{name}: missing dates")
        require(parsed.dt.tz is None, f"{label}.{name}: timezone-bearing calendar dates")
        require(parsed.eq(parsed.dt.normalize()).all(), f"{label}.{name}: non-midnight times")
        frame[name] = parsed


def numbers(frame: pd.DataFrame, names: list[str], label: str, integer: bool = False) -> None:
    for name in names:
        value = pd.to_numeric(frame[name], errors="raise")
        require(np.isfinite(value.to_numpy(dtype=float)).all(), f"{label}.{name}: missing/nonfinite values")
        if integer:
            require(value.eq(np.floor(value)).all(), f"{label}.{name}: noninteger values")
            value = value.astype("int64")
        frame[name] = value


def exact_keys(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], label: str) -> None:
    require(not actual.duplicated(keys).any(), f"{label}: duplicate keys")
    require(not expected.duplicated(keys).any(), f"{label}: duplicate reference keys")
    left = pd.MultiIndex.from_frame(actual[keys])
    right = pd.MultiIndex.from_frame(expected[keys])
    difference = left.symmetric_difference(right)
    require(len(difference) == 0, f"{label}: missing/extra keys; examples={difference[:3].tolist()}")


class StudyAudit:
    def __init__(self, root: Path, output: Path, source: Path | None = None):
        self.root, self.output = root.resolve(), output.resolve()
        require(self.root != self.output, "Audit output must differ from the input root")
        self.source = source.resolve() if source else self.find_source()
        self.output.mkdir(parents=True, exist_ok=True)
        self.checks: list[dict] = []
        self.artifacts: dict[str, dict] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.documents: dict[str, dict] = {}
        self.reference: pd.DataFrame | None = None
        self.cohort: pd.DataFrame | None = None
        self.expected_methods = METHODS
        self.entrypoint = Path(__file__).resolve()
        self.profile_label = "final-study"

    def find_source(self) -> Path | None:
        candidates = [self.root / "source", self.root / "source_release",
                      self.root / "CPCB_AQI_Source_Release_v1"]
        for parent in (self.root, *self.root.parents, Path(__file__).resolve().parents[1]):
            candidates.append(parent / "delivery/dataset_release_20261114/CPCB_AQI_Source_Release_v1")
        return next((p.resolve() for p in candidates
                     if (p / "file_register.csv").is_file() and (p / "station_metadata.csv").is_file()), None)

    def artifact(self, path: Path, kind: str = "input") -> Path:
        path = path.resolve()
        require(path.is_file(), f"Missing required artifact: {path}")
        self.artifacts[str(path)] = {"path": str(path), "kind": kind,
                                     "bytes": path.stat().st_size, "sha256": sha256(path)}
        return path

    def csv(self, relative: str, empty: bool = False) -> pd.DataFrame:
        path = self.artifact(self.root / relative)
        try:
            return pd.read_csv(path, dtype={"station_id": str, "site_id": str})
        except pd.errors.EmptyDataError:
            if empty:
                return pd.DataFrame()
            raise

    def write_csv(self, name: str, frame: pd.DataFrame) -> str:
        path = self.output / name
        frame.to_csv(path, index=False)
        self.artifact(path, "independent audit output")
        return str(path)

    def run_check(self, name: str, operation: Callable[[], dict]) -> None:
        try:
            details = operation()
            self.checks.append({"name": name, "status": "PASS", "critical": True, "details": safe(details)})
        except Exception as error:
            self.checks.append({"name": name, "status": "FAIL", "critical": True,
                                "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc(limit=3)})

    def load(self) -> dict:
        for name in ("data/data_summary.json", "data/protocol.json", "frozen_protocol.json",
                     "models/engine_metadata.json"):
            self.documents[name] = json.loads(self.artifact(self.root / name).read_text())
        for name in ("data/observations.csv.gz", "data/forecast_examples.csv.gz", "data/cohort.csv",
                     "data/raw_integrity.csv", "data/eligible_stations.csv", "models/predictions.csv.gz",
                     "models/fit_log.csv", "evaluation/case_metrics.csv", "evaluation/city_metrics.csv",
                     "evaluation/macro_metrics.csv"):
            self.tables[name] = self.csv(name)
        return {"root": str(self.root), "original_release": str(self.source) if self.source else None,
                "required_artifacts": len(self.artifacts)}

    def source_integrity(self) -> dict:
        require(self.source is not None, "Original release not found; supply --source for support hashes and all-station coordinates")
        summary = self.documents["data/data_summary.json"]
        register_path = self.artifact(self.source / "file_register.csv")
        metadata_path = self.artifact(self.source / "station_metadata.csv")
        require(sha256(register_path) == summary["register_sha256"], "Original file_register.csv hash differs from data summary")
        require(sha256(metadata_path) == summary["metadata_sha256"], "Original station_metadata.csv hash differs from data summary")
        register = pd.read_csv(register_path, dtype={"station_id": str})
        self.metadata = pd.read_csv(metadata_path, dtype={"site_id": str})
        columns(register, ["filename", "station_id", "year", "sha256", "bytes"], "source register")
        columns(self.metadata, ["site_id", "city", "city_code", "latitude", "longitude"], "source metadata")
        require(not self.metadata.site_id.duplicated().any(), "Duplicate original metadata station IDs")
        integrity = self.tables["data/raw_integrity.csv"]
        columns(integrity, ["filename", "sha256", "observations", "above_500", "below_zero"], "raw integrity")
        exact_keys(integrity, register, ["filename"], "Workbook inventory")
        joined = register.merge(integrity, on="filename", validate="one_to_one", suffixes=("_register", "_data"))
        require(joined.sha256_register.eq(joined.sha256_data).all(), "Workbook hashes disagree between raw integrity and source register")
        for row in register.itertuples(index=False):
            filename = Path(row.filename)
            require(filename.name == str(filename), "Source register contains a non-basename workbook path")
            path = self.source / "aqi_files" / filename
            require(path.is_file() and path.stat().st_size == int(row.bytes), f"Missing/size-mismatched source workbook {path}")
            require(sha256(path) == row.sha256, f"Source workbook checksum mismatch: {path}")
        obs = self.tables["data/observations.csv.gz"]
        columns(obs, ["station_id", "date", "aqi", "city"], "observations")
        require(obs.station_id.notna().all(), "Observations have missing station IDs")
        dates(obs, ["date"], "observations")
        numbers(obs, ["aqi"], "observations")
        require(not obs.duplicated(["station_id", "date"]).any(), "Duplicate original observed station/date")
        source_city = self.metadata.set_index("site_id").city.reindex(obs.station_id).reset_index(drop=True)
        require(obs.city.fillna("").reset_index(drop=True).eq(source_city.fillna("")).all(), "Observed city identity differs from original metadata")
        actual_counts = {"workbooks": len(register), "observations": len(obs),
                         "matched_observations": int(obs.city.notna().sum()),
                         "unmatched_observations": int(obs.city.isna().sum()),
                         "empty_workbooks": int(integrity.observations.eq(0).sum()),
                         "above500": int(obs.aqi.gt(500).sum()), "duplicates": 0}
        for name, value in actual_counts.items():
            require(int(summary[name]) == value, f"Data summary {name}: reported {summary[name]}, independently counted {value}")
        require(int(summary["parse_errors"]) == 0, "Data summary reports workbook parse errors")
        require(int(integrity.observations.sum()) == len(obs), "Raw-integrity observation total differs from observation panel")
        require(int(integrity.above_500.sum()) == obs.aqi.gt(500).sum(), "Above-range workbook count disagrees with observations")
        require(int(integrity.below_zero.sum()) == obs.aqi.lt(0).sum(), "Below-range workbook count disagrees with observations")
        archive = self.source.with_suffix(".zip")
        archive_check = "not present; extracted support and every workbook verified"
        if archive.is_file():
            self.artifact(archive)
            require(sha256(archive) == self.documents["frozen_protocol.json"]["source_release_sha256"], "Source-release ZIP hash differs from frozen protocol")
            archive_check = str(archive)
        return {**actual_counts, "workbook_byte_checks": len(register), "source_archive": archive_check,
                "observation_provenance_limit": "Workbooks are hash-verified; this audit does not independently reparse XLSX cells."}

    def cohort_integrity(self) -> dict:
        cohort = self.tables["data/cohort.csv"].copy()
        columns(cohort, ["site_id", "city", "city_code", "latitude", "longitude", "role",
                         "training_observed_days", "nearest_target_station_km"], "cohort")
        require(len(cohort) == 30 and not cohort.site_id.duplicated().any()
                and not cohort.city_code.duplicated().any() and not cohort.city.duplicated().any(), "Cohort must have 30 unique representative cities/stations")
        require(set(cohort.loc[cohort.role.eq("target"), "city"]) == set(TARGETS), "Target roles differ from the six frozen target cities")
        numbers(cohort, ["latitude", "longitude", "nearest_target_station_km"], "cohort")
        source = self.metadata.set_index("site_id").reindex(cohort.site_id)
        for name in ("city", "city_code"):
            require(np.array_equal(cohort[name].to_numpy(), source[name].to_numpy()), f"Cohort {name} differs from original source metadata")
        for name in ("latitude", "longitude"):
            require(np.array_equal(cohort[name].to_numpy(float), pd.to_numeric(source[name]).to_numpy(float)), f"Cohort {name} differs from original source metadata")
        targets = self.metadata[self.metadata.city.isin(TARGETS)].copy()
        numbers(targets, ["latitude", "longitude"], "all known target stations")
        require(set(targets.city) == set(TARGETS), "Original metadata lacks at least one target city")
        target_coordinates = np.radians(targets[["latitude", "longitude"]].to_numpy(float))
        nearest = []
        for latitude, longitude in np.radians(cohort[["latitude", "longitude"]].to_numpy(float)):
            a = (np.sin((target_coordinates[:, 0] - latitude) / 2) ** 2
                 + np.cos(latitude) * np.cos(target_coordinates[:, 0])
                 * np.sin((target_coordinates[:, 1] - longitude) / 2) ** 2)
            nearest.append(float((12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).min()))
        require(np.allclose(cohort.nearest_target_station_km, nearest, rtol=0, atol=1e-8), "Saved distances do not use all known target-city stations")
        expected_roles = ["target" if city in TARGETS else "excluded_source_buffer" if km < 25
                          else "donor_development" if city in DEVELOPMENT else "donor"
                          for city, km in zip(cohort.city, nearest)]
        require(cohort.role.tolist() == expected_roles, "Whole-city holdout/development/25-km buffer roles differ from independent geography")
        donors = cohort[cohort.role.isin(["donor", "donor_development"])]
        require(len(donors) == 23 and set(DEVELOPMENT).issubset(set(donors.city)), "Expected 23 donors including four development cities")
        require(not set(donors.city_code).intersection(targets.city_code), "A target-city code remains in donors")
        # Independently reconstruct availability eligibility and representative ranking.
        obs = self.tables["data/observations.csv.gz"]
        counts = obs[obs.date.dt.year.between(2020, 2025)].groupby(["station_id", obs.date.dt.year]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=range(2020, 2026), fill_value=0)
        eligible_ids = counts.index[counts.div([366, 365, 365, 365, 366, 365], axis=1).ge(.8).all(axis=1)]
        eligible = self.metadata[self.metadata.site_id.isin(eligible_ids)].copy()
        eligible["training_observed_days"] = eligible.site_id.map(counts.loc[:, 2020:2023].sum(axis=1))
        eligible["numeric_id"] = pd.to_numeric(eligible.site_id, errors="raise")
        chosen = eligible.sort_values(["training_observed_days", "numeric_id"], ascending=[False, True]).drop_duplicates("city_code")
        exact_keys(cohort, chosen, ["site_id", "city_code"], "Independent availability/ranking cohort")
        stored = self.tables["data/eligible_stations.csv"]
        exact_keys(stored, eligible, ["site_id"], "Eligible station inventory")
        summary = self.documents["data/data_summary.json"]
        for key, value in (("eligible_stations_at_least_80pct", len(eligible)),
                           ("representative_cities", len(cohort)), ("donor_cities", len(donors))):
            require(int(summary[key]) == value, f"Data summary {key} differs from source reconstruction")
        self.cohort, self.donors = cohort, sorted(donors.city)
        return {"targets": list(TARGETS), "donors": self.donors, "known_target_stations_checked": len(targets),
                "buffer_km": 25, "minimum_donor_distance_km": min(np.asarray(nearest)[cohort.role.isin(["donor", "donor_development"])]),
                "excluded": cohort.loc[cohort.role.eq("excluded_source_buffer"), ["city", "site_id", "nearest_target_station_km"]].to_dict("records")}

    def protocol_integrity(self) -> dict:
        frozen = self.documents["frozen_protocol.json"]
        data_protocol = self.documents["data/protocol.json"]
        engine = self.documents["models/engine_metadata.json"]
        for name, expected in (("budgets", BUDGETS), ("seeds", SEEDS), ("horizons", HORIZONS)):
            require(set(frozen[name]) == set(expected), f"Frozen {name} differs from independent required grid")
            require(data_protocol[name] == frozen[name], f"Data and frozen protocol disagree on {name}")
        require(set(frozen.get("target_cities", frozen.get("targets", []))) == set(TARGETS), "Frozen target-city set differs")
        require(set(frozen["expected_methods"]) == set(self.expected_methods), "Frozen method name set differs")
        require(set(frozen["expected_periods"]) == set(PERIODS), "Frozen evaluation period set differs")
        require(frozen["latest_fit_cutoff"] == "2023-12-31"
                and frozen["target_fitting_cutoff"] == "2023-12-31", "Frozen fitting cutoff differs")
        require(frozen["source_release_sha256"] == data_protocol["source_release_sha256"], "Data/frozen source-release hashes differ")
        require(pd.notna(pd.to_datetime(frozen["frozen_utc"], utc=True)), "Invalid frozen_utc")
        config = engine["config"]
        for name, expected in (("budgets", BUDGETS), ("seeds", SEEDS),
                               ("architectures", ("gru", "lstm", "tcn", "transformer", "mlp"))):
            require(set(config[name]) == set(expected), f"Engine config {name} differs from the required grid")
        for name, value in frozen.get("model_config", {}).items():
            require(config.get(name) == value, f"Engine/frozen model config differs on {name}")
        require(not config.get("smoke", False), "Engine reports a smoke run")
        require(int(config["purge_days"]) >= 9, "Engine reports less than nine full purge days")
        require(set(engine["methods"]) == set(self.expected_methods) and set(engine["target_cities"]) == set(TARGETS), "Engine method/target inventory differs")
        require(set(engine["donor_cities"]) == set(self.donors), "Engine donor list differs from independently checked cohort")
        require(set(engine["dev_cities"]) == set(DEVELOPMENT), "Engine development-city list differs")
        require(engine["scaling"]["target_fitted_scalers"] is False
                and engine["scaling"]["output_clipping"] is False, "Engine reports target-fitted scaling or clipped forecasts")
        implementation = frozen.get("implementation_sha256", {})
        expected_names = {"final_data_pipeline.py", "final_model_engine.py", "final_evaluation.py"}
        require(set(implementation) == expected_names, "Frozen implementation hashes must name all three study modules")
        code_checks = []
        for name, expected_digest in sorted(implementation.items()):
            require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_digest)), f"Invalid frozen code digest for {name}")
            candidates = [self.root / "scripts" / name, self.root / "source" / name, self.root / "code" / name, self.root / name,
                          Path(__file__).resolve().parent / name]
            path = next((p for p in candidates if p.is_file()), None)
            if path:
                self.artifact(path, "implementation source")
                require(sha256(path) == expected_digest, f"Available implementation differs from frozen digest: {path}")
            code_checks.append({"module": name, "expected_sha256": expected_digest,
                                "path": str(path.resolve()) if path else None,
                                "status": "hash verified" if path else "source unavailable; declared hash recorded"})
        return {"engine_version": engine.get("engine_version"), "reported_device": engine.get("device"),
                "reported_gpu": engine.get("gpu"), "fingerprint": engine.get("fingerprint"),
                "frozen_utc": frozen["frozen_utc"], "implementation": code_checks}

    def forecast_integrity(self) -> dict:
        frame = self.tables["data/forecast_examples.csv.gz"]
        required = ["station_id", "city", "date", "issue_date", "horizon", "feature_date_min", "feature_date_max",
                    "y", "last", *(f"x{i}" for i in range(7)), "issue_doy", "issue_dow"]
        columns(frame, required, "forecast examples")
        require(frame[required].notna().all().all(), "Forecast examples contain missing required values")
        dates(frame, ["date", "issue_date", "feature_date_min", "feature_date_max"], "forecast examples")
        numbers(frame, ["horizon", "issue_doy", "issue_dow"], "forecast examples", True)
        numbers(frame, ["y", "last", *(f"x{i}" for i in range(7))], "forecast examples")
        require(frame.horizon.isin(HORIZONS).all(), "Unexpected forecast horizon")
        require((frame.date - frame.issue_date).dt.days.eq(frame.horizon).all(), "Label date is not issue_date + horizon")
        require(frame.feature_date_min.eq(frame.issue_date - pd.Timedelta(days=6)).all()
                and frame.feature_date_max.eq(frame.issue_date).all(), "Input feature date interval is not the seven observed issue-day lags")
        require(frame.issue_doy.eq(frame.issue_date.dt.day_of_year).all()
                and frame.issue_dow.eq(frame.issue_date.dt.day_of_week).all(), "Calendar features differ from issue calendar")
        source = self.tables["data/observations.csv.gz"].set_index(["station_id", "date"]).aqi
        for name, lookup_date in [("y", frame.date), ("last", frame.issue_date),
                                  *[(f"x{i}", frame.issue_date - pd.Timedelta(days=6-i)) for i in range(7)]]:
            keys = pd.MultiIndex.from_arrays([frame.station_id, lookup_date])
            values = source.reindex(keys).to_numpy(float)
            require(np.isfinite(values).all(), f"{name}: at least one required date lacks an original observation")
            require(np.array_equal(frame[name].to_numpy(float), values), f"{name}: features/truth differ from exact original observed AQI")
        require(frame["last"].eq(frame.x6).all(), "Last AQI does not equal x6")
        expected = []
        obs = self.tables["data/observations.csv.gz"]
        for row in self.cohort.itertuples(index=False):
            available = pd.DatetimeIndex(obs.loc[obs.station_id.eq(row.site_id), "date"])
            eligible = available[(available >= pd.Timestamp("2020-01-04")) & (available <= pd.Timestamp("2025-12-31"))]
            for delta in range(1, 10):
                eligible = eligible[(eligible - pd.Timedelta(days=delta)).isin(available)]
            for horizon in HORIZONS:
                expected.append(pd.DataFrame({"station_id": row.site_id, "city": row.city,
                                              "date": eligible, "horizon": horizon}))
        exact_keys(frame, pd.concat(expected, ignore_index=True), ["station_id", "city", "date", "horizon"], "All source-eligible common-horizon examples")
        require(not frame.duplicated(["city", "date", "horizon"]).any(), "Duplicate city/date/horizon example")
        reference = []
        target = frame[frame.city.isin(TARGETS)]
        for period, year in PERIODS.items():
            selected = target[target.date.dt.year.eq(year) & target.issue_date.gt(CUTOFF)]
            if year == 2024:
                selected = selected[selected.date.ge(pd.Timestamp("2024-01-04"))]
            reference.append(selected.assign(period=period))
        self.reference = pd.concat(reference, ignore_index=True)
        for (_, _), part in self.reference.groupby(["period", "city"]):
            require(part.groupby("date").horizon.nunique().eq(3).all(), "Reference target dates are not common across all horizons")
        return {"source_checked_examples": len(frame), "exact_original_columns": ["y", "last", *(f"x{i}" for i in range(7))],
                "eligible_evaluation_contexts": len(self.reference),
                "context_inventory": self.write_csv("independent_evaluation_dates.csv", self.reference[REFERENCE_KEY + ["issue_date"]])}

    def prediction_integrity(self) -> dict:
        frame = self.tables["models/predictions.csv.gz"]
        needed = [*REFERENCE_KEY, "issue_date", "budget", "seed", "method", "y_true", "y_pred", "last"]
        columns(frame, needed, "predictions")
        require(not frame.empty and frame[needed].notna().all().all(), "Empty predictions or missing required prediction values")
        dates(frame, ["date", "issue_date"], "predictions")
        numbers(frame, ["budget", "seed", "horizon"], "predictions", True)
        numbers(frame, ["y_true", "y_pred", "last"], "predictions")
        for name, expected in (("method", self.expected_methods), ("city", TARGETS), ("budget", BUDGETS), ("seed", SEEDS),
                               ("horizon", HORIZONS), ("period", PERIODS)):
            require(set(frame[name].unique()) == set(expected), f"Prediction {name} set differs from canonical grid")
        require(not frame.duplicated([*REFERENCE_KEY, "budget", "seed", "method"]).any(), "Duplicate prediction identity")
        require(frame.issue_date.gt(CUTOFF).all(), "An evaluation issue date is on/before 2023-12-31")
        require((frame.date - frame.issue_date).dt.days.eq(frame.horizon).all(), "Prediction chronology differs from horizon")
        for period, year in PERIODS.items():
            require(frame.loc[frame.period.eq(period), "date"].dt.year.eq(year).all(), f"{period} contains labels outside {year}")
        contexts = frame.groupby(REFERENCE_KEY, sort=False, observed=True).size().rename("rows").reset_index()
        exact_keys(contexts, self.reference, REFERENCE_KEY, "Prediction dates versus every source-eligible evaluation context")
        multiplicity = len(BUDGETS) * len(SEEDS) * len(self.expected_methods)
        require(contexts.rows.eq(multiplicity).all(), "Incomplete method/seed/budget grid at one or more source-eligible dates")
        # Uniqueness, canonical axes and the full product size per context imply the
        # complete Cartesian product; no intersection-only scoring is possible.
        context_values = frame[REFERENCE_KEY + ["issue_date", "y_true", "last"]].drop_duplicates()
        exact_keys(context_values, self.reference, REFERENCE_KEY, "Constant prediction truth/context")
        compared = context_values.merge(self.reference[REFERENCE_KEY + ["issue_date", "y", "last"]],
                                        on=REFERENCE_KEY, validate="one_to_one", suffixes=("", "_original"))
        require(compared.issue_date.eq(compared.issue_date_original).all()
                and compared.y_true.eq(compared.y).all() and compared["last"].eq(compared.last_original).all(),
                "Prediction truth/last/issue context differs from source-verified examples")
        for method in ("persistence", "seasonal7"):
            baseline = frame[frame.method.eq(method)].merge(
                self.reference[REFERENCE_KEY + ["x0", "x1", "x2", "last"]],
                on=REFERENCE_KEY, validate="many_to_one", suffixes=("", "_original"))
            expected = baseline.last_original.to_numpy(float) if method == "persistence" else baseline[["x0", "x1", "x2"]].to_numpy(float)[np.arange(len(baseline)), baseline.horizon.to_numpy(int)-1]
            require(np.array_equal(baseline.y_pred.to_numpy(float), expected), f"{method} predictions are not exact original observed references")
        return {"prediction_rows": len(frame), "methods": list(self.expected_methods), "seeds": list(SEEDS),
                "budgets": list(BUDGETS), "horizons": list(HORIZONS), "periods": list(PERIODS),
                "target_cities": list(TARGETS), "complete_cartesian_rows_per_context": multiplicity,
                "same_dates_across_methods_seeds_budgets_horizons": True,
                "persistence_exact": True, "seasonal7_exact_x_horizon_minus_1": True,
                "earliest_issue_date": frame.issue_date.min(), "latest_issue_date": frame.issue_date.max()}

    def budget_and_fitting_integrity(self) -> dict:
        log = self.tables["models/fit_log.csv"]
        columns(log, ["stage", "method", "seed", "city", "budget", "train_examples", "train_label_dates",
                      "train_first_label", "train_last_label", "target_validation_used"], "fit log")
        require(not log.empty and log.target_validation_used.notna().all(), "Fit records lack explicit target-validation declarations")
        require(log.target_validation_used.astype(str).str.lower().isin(["false", "0"]).all(), "A fit record reports target validation or an unknown validation flag")
        allowed_stages = self.allowed_fit_stages()
        require(set(log.stage).issubset(allowed_stages), "Fit log contains unknown or missing fitting stages")
        for name in ("train_last_label", "support_last", "query_last"):
            if name in log:
                reported = pd.to_datetime(log[name].dropna(), errors="raise")
                require(reported.le(CUTOFF).all(), f"Fit-log {name} contains a post-2023 label")
        target = log[log.stage.eq("target_fit")].copy()
        dates(target, ["train_first_label", "train_last_label"], "target fit")
        numbers(target, ["budget", "seed", "train_examples", "train_label_dates", "available_budget_examples"], "target fit", True)
        required = pd.DataFrame([(c, b, s, m) for c in TARGETS for b in BUDGETS for s in SEEDS for m in self.expected_methods],
                                columns=["city", "budget", "seed", "method"])
        exact_keys(target, required, ["city", "budget", "seed", "method"], "Target fit grid")
        examples = self.tables["data/forecast_examples.csv.gz"]
        labels = []
        for city in TARGETS:
            part = examples[examples.city.eq(city) & examples.date.le(CUTOFF)]
            eligible = pd.DatetimeIndex(part.date.unique()).sort_values()
            previous: set[pd.Timestamp] = set()
            for budget in BUDGETS:
                require(len(eligible) >= budget, f"{city}: insufficient eligible fitting labels for B={budget}")
                selected = eligible[-budget:]
                support = part[part.date.isin(selected)]
                require(len(support) == 3 * budget and support.groupby("date").horizon.nunique().eq(3).all(), f"{city} B={budget}: not exactly B three-horizon labels")
                require(previous.issubset(set(selected)), f"{city}: reconstructed label budgets are not nested")
                previous = set(selected)
                rows = target[target.city.eq(city) & target.budget.eq(budget)]
                require(rows.train_label_dates.eq(budget).all() and rows.available_budget_examples.eq(3 * budget).all(), f"{city} B={budget}: fit-log budget counts differ")
                require(rows.train_first_label.eq(selected.min()).all() and rows.train_last_label.eq(selected.max()).all(), f"{city} B={budget}: fit-log first/last labels differ from latest eligible B")
                untrained = rows.method.isin(["persistence", "seasonal7", "mean7"]) | rows.method.str.endswith("pooled_zero")
                require(rows.train_examples.eq(np.where(untrained, 0, 3 * budget)).all(), f"{city} B={budget}: fitted supervised example counts differ")
                labels.append(pd.DataFrame({"city": city, "station_id": part.station_id.iloc[0], "budget": budget, "date": selected}))
        source_details = self.source_fitting_integrity(log, examples)
        return {"target_fit_records": len(target), "fit_records": len(log), "target_validation_used": False,
                "all_recorded_fit_and_selection_labels_through_2023": True,
                "budget_labels": self.write_csv("independent_budget_labels.csv", pd.concat(labels, ignore_index=True)),
                "budget_evidence_limit": "Exact nested last-B labels are independently reconstructed; fit records corroborate counts and endpoints but do not enumerate the labels actually consumed by each optimizer.",
                "source_fitting": source_details}

    def allowed_fit_stages(self) -> set[str]:
        return {"target_fit", "donor_checkpoint_selection", "donor_fixed_refit",
                "donor_meta_checkpoint_selection", "donor_meta_fixed_refit", "donor_pseudo_candidate"}

    def source_fitting_integrity(self, log: pd.DataFrame, examples: pd.DataFrame) -> dict:
        donor = examples[examples.city.isin(self.donors) & examples.issue_date.ge(pd.Timestamp("2020-01-01"))]
        source_methods = [f"{a}_pooled" for a in ("gru", "lstm", "tcn", "transformer", "mlp")]
        for stage, methods in (("donor_checkpoint_selection", source_methods), ("donor_fixed_refit", source_methods),
                               ("donor_meta_checkpoint_selection", ["reptile_gru", "fomaml_gru"]),
                               ("donor_meta_fixed_refit", ["reptile_gru", "fomaml_gru"])):
            rows = log[log.stage.eq(stage)].copy()
            numbers(rows, ["seed"], stage, True)
            exact_keys(rows, pd.DataFrame([(m, s) for m in methods for s in SEEDS], columns=["method", "seed"]), ["method", "seed"], stage)
            is_selection = "selection" in stage
            training = donor[donor.date.le(pd.Timestamp("2022-12-31") if is_selection else CUTOFF)]
            if is_selection:
                training = training[~training.city.isin(DEVELOPMENT)]
            for name, expected in (("train_examples", len(training)), ("train_label_dates", training.date.nunique()), ("train_cities", training.city.nunique())):
                require(pd.to_numeric(rows[name]).eq(expected).all(), f"{stage}: recorded {name} differs from donor-only source reconstruction")
            for name, expected in (("train_first_label", training.date.min()), ("train_last_label", training.date.max())):
                require(pd.to_datetime(rows[name]).eq(expected).all(), f"{stage}: {name} differs from donor-only source reconstruction")
        candidates = log[log.stage.eq("donor_pseudo_candidate")].copy()
        require(not candidates.empty and set(candidates.city).issubset(set(self.donors))
                and not set(candidates.city).intersection(TARGETS), "Candidate selection includes target/unknown cities or has no evidence")
        dates(candidates, ["support_first", "support_last", "query_first", "query_last"], "donor pseudo candidates")
        require(candidates.support_last.le(pd.Timestamp("2022-12-31")).all()
                and candidates.query_first.dt.year.eq(2023).all() and candidates.query_last.dt.year.eq(2023).all(), "Donor candidate support/query violates 2020-22/2023 selection split")
        gaps = (candidates.query_first - candidates.support_last).dt.days - 1
        require(gaps.ge(9).all() and pd.to_numeric(candidates.purge_intervening_days).eq(gaps).all(), "Donor pseudo-candidate purge differs from at least nine actual intervening days")
        return {"source_fit_records": int(log.stage.isin(["donor_checkpoint_selection", "donor_fixed_refit",
                                                         "donor_meta_checkpoint_selection", "donor_meta_fixed_refit"]).sum()),
                "donor_selection_candidate_records": len(candidates)}

    def meta_episode_integrity(self) -> dict:
        config = self.documents["models/engine_metadata.json"]["config"]
        tasks, query_days = int(config["meta_tasks"]), int(config["meta_query_days"])
        require(tasks > 0 and query_days > 0, "Invalid meta task/query count")
        log = self.tables["models/fit_log.csv"]
        examples = self.tables["data/forecast_examples.csv.gz"]
        total, minimum_gap, expected_paths = 0, None, set()
        for algorithm in ("reptile", "fomaml"):
            for selection, stage in ((True, "donor_meta_checkpoint_selection"), (False, "donor_meta_fixed_refit")):
                training = examples[examples.city.isin(self.donors) & examples.issue_date.ge(pd.Timestamp("2020-01-01"))
                                    & examples.date.le(pd.Timestamp("2022-12-31") if selection else CUTOFF)]
                if selection:
                    training = training[~training.city.isin(DEVELOPMENT)]
                by_city = {city: pd.DatetimeIndex(part.date.unique()).sort_values() for city, part in training.groupby("city")}
                for seed in SEEDS:
                    rows = log[log.stage.eq(stage) & log.method.eq(f"{algorithm}_gru") & log.seed.eq(seed)]
                    require(len(rows) == 1, f"Missing/duplicate meta fit record {algorithm}/{stage}/{seed}")
                    steps = int(rows.requested_steps.iloc[0])
                    require(steps >= 0, "Negative requested meta steps")
                    relative = f"models/meta_episodes/{algorithm}_{'selection' if selection else 'refit'}_s{seed}.csv.gz"
                    expected_paths.add((self.root / relative).resolve())
                    frame = self.csv(relative, empty=True)
                    require(len(frame) == steps * tasks, f"{relative}: incomplete episode log versus requested steps/tasks")
                    if frame.empty:
                        continue  # A donor-selected step zero is a legitimate frozen initialization.
                    columns(frame, ["city", "budget", "support_first", "support_last", "query_first", "query_last",
                                    "purge_intervening_days", "support_examples", "query_examples", "step", "task", "algorithm", "seed"], relative)
                    dates(frame, ["support_first", "support_last", "query_first", "query_last"], relative)
                    numbers(frame, ["budget", "purge_intervening_days", "support_examples", "query_examples", "step", "task", "seed"], relative, True)
                    require(frame.algorithm.eq(algorithm).all() and frame.seed.eq(seed).all(), f"{relative}: algorithm/seed identity differs")
                    require(set(frame.city).issubset(by_city), f"{relative}: target/development-excluded/unknown city in meta episodes")
                    require(frame.budget.isin(BUDGETS).all(), f"{relative}: unexpected support budget")
                    require(not frame.duplicated(["step", "task"]).any() and frame.step.between(1, steps).all()
                            and frame.task.between(0, tasks-1).all(), f"{relative}: incomplete/duplicate outer-step task grid")
                    gaps = (frame.query_first - frame.support_last).dt.days - 1
                    require(gaps.ge(9).all() and frame.purge_intervening_days.eq(gaps).all(), f"{relative}: support/query purge is less than nine actual intervening calendar days")
                    for row in frame.itertuples(index=False):
                        available = by_city[row.city]
                        support = available[available <= row.query_first - pd.Timedelta(days=int(config["purge_days"]) + 1)][-int(row.budget):]
                        query = available[available >= row.query_first][:query_days]
                        require(len(support) == row.budget and support[0] == row.support_first and support[-1] == row.support_last,
                                f"{relative}: support bounds are not the eligible latest purged B donor labels")
                        require(len(query) == query_days and query[0] == row.query_first and query[-1] == row.query_last,
                                f"{relative}: query bounds are not the next eligible donor labels")
                        require(row.support_examples == 3 * row.budget and row.query_examples == 3 * query_days,
                                f"{relative}: wrong three-horizon support/query counts")
                    minimum_gap = int(gaps.min()) if minimum_gap is None else min(minimum_gap, int(gaps.min()))
                    total += len(frame)
        actual_paths = {p.resolve() for p in (self.root / "models/meta_episodes").glob("*.csv.gz")}
        require(actual_paths == expected_paths, "Missing/unexpected meta episode audit files")
        require(total > 0, "No meta-training episodes were recorded")
        return {"episode_files": len(expected_paths), "episodes": total, "minimum_actual_intervening_calendar_days": minimum_gap,
                "all_episode_cities_are_allowed_donors": True, "support_and_query_bounds_reconstructed_from_source_eligible_labels": True}

    def metric_integrity(self) -> dict:
        frame = self.tables["models/predictions.csv.gz"]
        work = frame[CASE_KEY + ["date", "y_pred", "y_true"]].copy()
        error = work.pop("y_pred").to_numpy(float) - work.pop("y_true").to_numpy(float)
        work["absolute_error"], work["squared_error"], work["error"] = np.abs(error), np.square(error), error
        case = work.groupby(CASE_KEY, observed=True, as_index=False).agg(
            n_dates=("date", "nunique"), mae=("absolute_error", "mean"), mse=("squared_error", "mean"), bias=("error", "mean"))
        case["rmse"] = np.sqrt(case.mse)
        # The complete date/seed grid makes averaging per-seed city losses
        # identical to averaging seed losses at each date, without ensembling.
        city = case.groupby(CITY_KEY, observed=True, as_index=False).agg(
            n_dates=("n_dates", "first"), n_seeds=("seed", "nunique"), mae=("mae", "mean"), mse=("mse", "mean"), bias=("bias", "mean"))
        city["rmse"] = np.sqrt(city.mse)
        macro = city.groupby(MACRO_KEY, observed=True, as_index=False).agg(
            n_cities=("city", "nunique"), n_city_dates=("n_dates", "sum"), mae=("mae", "mean"), mse=("mse", "mean"),
            rmse=("rmse", "mean"), bias=("bias", "mean"))
        # Published macro RMSE is mean(city RMSE), not sqrt(macro MSE).
        comparisons = []
        for name, independent, keys in (("case", case, CASE_KEY), ("city", city, CITY_KEY), ("macro", macro, MACRO_KEY)):
            reported = self.tables[f"evaluation/{name}_metrics.csv"].copy()
            columns(reported, keys + ["mae", "rmse", "bias"], f"reported {name} metrics")
            exact_keys(reported, independent, keys, f"{name} metric grid")
            merged = independent.merge(reported, on=keys, validate="one_to_one", suffixes=("_audit", "_reported"))
            tested, largest = [], 0.0
            for metric in ("mae", "rmse", "bias", "n_dates", "n_seeds", "n_cities", "n_city_dates"):
                if metric not in independent or metric not in reported:
                    continue
                a = pd.to_numeric(merged[metric + "_audit"]).to_numpy(float)
                b = pd.to_numeric(merged[metric + "_reported"]).to_numpy(float)
                require(np.isfinite(b).all(), f"Reported {name}.{metric} has missing/nonfinite values")
                difference = float(np.max(np.abs(a - b))) if len(a) else 0.0
                require(np.allclose(a, b, atol=ATOL, rtol=0), f"Independent {name}.{metric} disagrees with evaluation CSV; max absolute difference={difference:.12g}")
                tested.append(metric)
                largest = max(largest, difference)
            # Evaluator omits MSE; verify it via its reported RMSE for each
            # city/seed and city, where RMSE really is sqrt(mean squared loss).
            if name != "macro":
                a = merged.mse.to_numpy(float) if "mse" not in reported else merged.mse_audit.to_numpy(float)
                b = np.square(pd.to_numeric(merged.rmse_reported).to_numpy(float))
                difference = float(np.max(np.abs(a - b))) if len(a) else 0.0
                require(np.allclose(a, b, atol=ATOL, rtol=0), f"Independent {name}.mse disagrees with reported RMSE squared; max absolute difference={difference:.12g}")
                tested.append("mse versus reported rmse squared")
                largest = max(largest, difference)
            if "mse" in reported:
                require(np.allclose(merged.mse_audit, merged.mse_reported, atol=ATOL, rtol=0), f"Independent {name}.mse disagrees with explicit MSE column")
                tested.append("mse")
            if name == "macro":
                reported_city = self.tables["evaluation/city_metrics.csv"].copy()
                reported_city["mse_from_reported_city_rmse"] = np.square(pd.to_numeric(reported_city.rmse))
                derived = reported_city.groupby(MACRO_KEY, observed=True, as_index=False).mse_from_reported_city_rmse.mean()
                derived = macro.merge(derived, on=MACRO_KEY, validate="one_to_one")
                difference = float(np.max(np.abs(derived.mse - derived.mse_from_reported_city_rmse)))
                require(np.allclose(derived.mse, derived.mse_from_reported_city_rmse, atol=ATOL, rtol=0),
                        f"Independent macro.mse disagrees with mean reported city RMSE squared; max absolute difference={difference:.12g}")
                tested.append("mse versus mean reported city rmse squared")
                largest = max(largest, difference)
            output = self.write_csv(f"independent_{name}_metrics.csv", independent)
            comparisons.append({"level": name, "rows": len(independent), "metrics_checked": tested,
                                "max_absolute_difference": largest, "evidence": output})
        return {"absolute_tolerance": ATOL, "relative_tolerance": 0,
                "macro_mae_definition": "unweighted mean of six city seed-averaged MAEs",
                "macro_rmse_definition": "unweighted mean of six city sqrt(seed-averaged MSE)",
                "macro_mse_definition": "unweighted mean of six city seed-averaged MSEs; separately exported",
                "comparisons": comparisons}

    def scope_limits(self) -> list[str]:
        return [
            "This is an artifact consistency and chronology audit, not proof that hardware executed every declared optimizer update.",
            "Original workbooks are checksum-verified; observed XLSX values are not independently reparsed with a second parser.",
            "Fitting logs record label counts and endpoints, not every actually consumed target label or meta support/query label.",
            "Bootstrap intervals, subgroup reports and scientific interpretation are outside this audit's metric-recalculation scope."]

    def finish(self) -> dict:
        failed = [check for check in self.checks if check["status"] == "FAIL" and check["critical"]]
        verdict = "FAIL" if failed else "PASS"
        result = {"verdict": verdict, "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "root": str(self.root), "output": str(self.output),
                  "verifier": {"path": str(self.entrypoint), "sha256": sha256(self.entrypoint),
                               "shared_audit_library": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
                               "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                               "engine_or_evaluator_imported": False, "models_rerun": False},
                  "critical_failures": len(failed), "checks": self.checks,
                  "artifacts": list(self.artifacts.values()),
                  "scope_limits": self.scope_limits()}
        (self.output / "verification.json").write_text(json.dumps(safe(result), indent=2, allow_nan=False) + "\n")
        text = [f"Independent {self.profile_label} audit: {verdict}", "", f"Input: {self.root}",
                f"Evidence: {self.output / 'verification.json'}", "",
                "No model fitting or prediction rerun was performed. The verifier imports only pandas, NumPy and the standard library.", ""]
        text.extend(f"- {check['status']}: {check['name']}" + (f" — {check['error']}" if check["status"] == "FAIL" else "") for check in self.checks)
        text.extend(["", f"Critical failures: {len(failed)}. All mandatory checks must pass before the audit verdict is PASS.", "",
                     "Independent metric CSVs include MSE even when the evaluation omits it. Macro RMSE averages city RMSEs; it is not the square root of macro MSE.", "",
                     "Scope limits:", *[f"- {limit}" for limit in result["scope_limits"]], ""])
        (self.output / "README.md").write_text("\n".join(text))
        if failed:
            raise AuditFailure(f"{len(failed)} critical audit failure(s); see {self.output / 'verification.json'}")
        return result


def verify_study(root: str | Path, output: str | Path, source: str | Path | None = None) -> dict:
    audit = StudyAudit(Path(root), Path(output), Path(source) if source else None)
    for name, operation in (("Required artifacts", audit.load), ("Original source checksums and observation inventory", audit.source_integrity),
                            ("Cohort selection, whole-city exclusions and all-station buffer", audit.cohort_integrity),
                            ("Frozen protocol, engine metadata and available code hashes", audit.protocol_integrity),
                            ("Every forecast feature, truth and eligible date against observations", audit.forecast_integrity),
                            ("Complete paired prediction grid, chronology and exact baselines", audit.prediction_integrity),
                            ("Exact nested label-budget reconstruction and fitting chronology", audit.budget_and_fitting_integrity),
                            ("Donor-only meta episodes and actual nine-day purges", audit.meta_episode_integrity),
                            ("Independent city/seed and macro metrics", audit.metric_integrity)):
        audit.run_check(name, operation)
    return audit.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Downloaded afml_final directory")
    parser.add_argument("--output", required=True, type=Path, help="Independent audit output directory")
    parser.add_argument("--source", type=Path, help="Original extracted CPCB source release; normally autodetected")
    args = parser.parse_args()
    result = verify_study(args.root, args.output, args.source)
    print(f"{result['verdict']}: {Path(args.output).resolve() / 'verification.json'}")


if __name__ == "__main__":
    main()
