"""Independent schema-1 audit for the frozen 33-method research study.

This extends verify_final_study's tested source, cohort, forecast, prediction and
metric checks. It imports neither model engine nor evaluator. Method names come
from metadata and must also match the frozen, explicit 33-arm research design.
No code or protocol is considered verified before frozen_protocol.json exists.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts import verify_final_study as audit
except ImportError:
    import verify_final_study as audit


DEEP = ("gru", "lstm", "tcn", "transformer", "mlp")
BASELINES = ("persistence", "seasonal7", "mean7", "ridge_local", "histgb_local", "randomforest_local")
RESEARCH_METHODS = (*BASELINES,
    *(f"deep_{a}_{m}" for a in DEEP for m in ("local", "pooled_zero", "finetune")),
    *(f"matched_gru_{m}" for m in ("local", "pooled_zero", "finetune")),
    *(f"{a}_gru_{o}" for a in ("maml", "fomaml", "reptile") for o in ("sgd", "adam")),
    "diversity6_pooled_zero", "diversity6_finetune", "diversity6_maml_sgd")
require, columns, exact_keys = audit.require, audit.columns, audit.exact_keys


def labels(frame: pd.DataFrame) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in sorted(frame.date.unique())]


def label_digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


class ResearchAudit(audit.StudyAudit):
    def __init__(self, root: Path, output: Path, source: Path | None = None):
        super().__init__(root, output, source)
        self.entrypoint = Path(__file__).resolve()
        self.profile_label = "research-study"

    def scope_limits(self) -> list[str]:
        return [
            "This is an artifact consistency and chronology audit, not proof that hardware executed every declared optimizer update.",
            "Original workbooks are checksum-verified; XLSX cells are not reparsed with a second parser.",
            "Explicit target support lists and meta episode label digests are checked; actual sampled minibatch identities are not logged.",
            "Schema-1 raw Reptile fitting records may label the outer optimizer outer_adam; the reviewed code applies parameter interpolation. Original records are preserved.",
            "Recorded selected-backend derivative evidence is checked; the verifier itself does not import or execute models.",
            "Bootstrap intervals, subgroup reports and scientific interpretation are outside this audit's metric-recalculation scope."]

    def read_json(self, relative: str) -> dict:
        value = json.loads(self.artifact(self.root / relative).read_text())
        self.documents[relative] = value
        return value

    def load(self) -> dict:
        details = super().load()
        engine = self.documents["models/engine_metadata.json"]
        require(engine.get("research_schema_version") == 1, "Only explicit research_schema_version 1 is supported")
        methods = engine.get("methods")
        require(isinstance(methods, list) and all(isinstance(value, str) for value in methods), "Metadata methods must be an explicit name list")
        require(len(methods) == len(set(methods)) == 33 and set(methods) == set(RESEARCH_METHODS),
                "Research metadata must contain the complete predeclared 33-method set")
        self.expected_methods = tuple(methods)
        for relative in ("models/selected_model_specs.json", "models/source_selections.json",
                         "models/diversity_donors.json", "models/frozen_adaptation_settings.json",
                         "models/frozen_methods.json", "models/progress.json", "models/second_order_gradient_check.json"):
            self.read_json(relative)
        paths = sorted((self.root / "models/fit_records").glob("*.json"))
        require(bool(paths), "Research audit requires the individual JSON fitting records")
        self.records = []
        for path in paths:
            row = json.loads(self.artifact(path).read_text())
            row["_artifact"] = str(path.resolve())
            self.records.append(row)
        require(len(self.records) == len(self.tables["models/fit_log.csv"]), "Individual JSON fit-record count differs from aggregate fit log")
        # Match every aggregate row to its journal identity without parsing Python
        # repr strings from CSV's nested JSON columns.
        identity = ["stage", "method", "seed", "city", "budget", "candidate", "source_candidate", "development_task"]
        records = pd.DataFrame(self.records).reindex(columns=identity).fillna("").astype(str)
        aggregate = self.tables["models/fit_log.csv"].reindex(columns=identity).fillna("")
        for name in ("seed", "budget", "candidate", "source_candidate", "development_task"):
            records[name] = records[name].map(lambda value: str(int(float(value))) if value else "")
            aggregate[name] = aggregate[name].map(lambda value: str(int(float(value))) if str(value) else "")
        aggregate = aggregate.astype(str)
        exact_keys(aggregate, records, identity, "Research fit journal versus aggregate log")
        self.artifact(Path(__file__), "independent research verifier")
        return {**details, "research_schema_version": 1, "method_inventory_source": "explicit metadata, corroborated by frozen design",
                "methods": methods, "individual_fit_records": len(self.records)}

    def allowed_fit_stages(self) -> set[str]:
        return {"target_fit", "donor_hpo_source", "donor_hpo_meta", "donor_fixed_refit",
                "donor_meta_fixed_refit", "donor_diversity_fixed_refit", "donor_adaptation_candidate", "donor_hpo_development"}

    def protocol_integrity(self) -> dict:
        frozen, data = self.documents["frozen_protocol.json"], self.documents["data/protocol.json"]
        engine = self.documents["models/engine_metadata.json"]
        config = engine["config"]
        for name, expected in (("budgets", audit.BUDGETS), ("seeds", audit.SEEDS), ("horizons", audit.HORIZONS)):
            require(set(frozen[name]) == set(expected) and frozen[name] == data[name], f"Frozen/data {name} differs from required grid")
            if name != "horizons":
                require(set(config[name]) == set(expected), f"Research engine {name} differs from frozen grid")
        require(set(frozen["expected_methods"]) == set(self.expected_methods)
                and len(frozen["expected_methods"]) == 33, "Frozen research methods differ from metadata")
        require(set(frozen["expected_periods"]) == set(audit.PERIODS), "Frozen research periods differ")
        require(set(frozen.get("target_cities", frozen.get("targets", []))) == set(audit.TARGETS), "Frozen targets differ")
        require(frozen["latest_fit_cutoff"] == frozen["target_fitting_cutoff"] == "2023-12-31", "Research fitting cutoff changed")
        require(frozen["source_release_sha256"] == data["source_release_sha256"], "Research/data source release hashes differ")
        require(pd.notna(pd.to_datetime(frozen["frozen_utc"], utc=True)), "Invalid research freeze timestamp")
        model_config = frozen.get("model_config", frozen.get("research_config"))
        require(isinstance(model_config, dict) and bool(model_config), "Frozen protocol must contain explicit model_config/research_config")
        for name, value in model_config.items():
            require(config.get(name) == value, f"Frozen/engine research configuration differs at {name}")
        require(not config.get("smoke", False), "A smoke run is not the final research study")
        require(set(config["architectures"]) == set(DEEP), "Missing deep architecture")
        require(all(config.get(name) is True for name in ("include_baselines", "include_meta", "include_diversity")), "A required research arm was disabled")
        require(int(config["purge_days"]) >= 9 and int(config["diversity_donors"]) == 6, "Invalid purge/diversity configuration")
        require(int(config["manual_hidden"]) == 64 and int(config["manual_layers"]) == 2, "Matched/meta backbone must be the frozen two-layer GRU64")
        for name in ("source_grid", "meta_grid", "deep_scratch_grid", "deep_finetune_grid", "native_sgd_grid", "meta_adam_grid"):
            require(len(config[name]) == 4, f"Research {name} must have four predeclared candidates")
        require(all(p["optimizer"] == "sgd" for p in config["native_sgd_grid"]), "Native matched/meta adaptation is not SGD")
        require(all(p["optimizer"] == "adam" for p in config["meta_adam_grid"]), "Secondary meta adaptation arm is not Adam")
        require(set(engine["target_cities"]) == set(audit.TARGETS) and set(engine["donor_cities"]) == set(self.donors)
                and set(engine["dev_cities"]) == set(audit.DEVELOPMENT), "Research metadata changes target/donor memberships")
        require(engine["scaling"].get("target_fitted_scalers") is False
                and engine["scaling"]["output_clipping"] is False, "Research scaling lacks explicit fixed-scale/unclipped declarations")
        require(engine["model_specs"]["meta_and_matched"] == {"architecture": "manual_gru", "hidden": 64, "layers": 2}, "Matched and meta backbones differ")
        require(self.documents["models/frozen_methods.json"]["methods"] == list(self.expected_methods), "Frozen-method artifact differs from metadata")
        progress = self.documents["models/progress.json"]
        require(progress["stage"] == "complete" and progress["completed_blocks"] == progress["total_blocks"] == 6*3*3*33,
                "Research run is incomplete")
        require(progress["prediction_rows"] == len(self.tables["models/predictions.csv.gz"]), "Progress prediction count differs")
        require(progress["fingerprint"] == engine["fingerprint"] == self.documents["models/frozen_methods.json"]["fingerprint"], "Research artifact fingerprints disagree")
        declared = frozen.get("implementation_sha256", {})
        required = {"final_model_engine.py", "research_model_engine.py", "final_evaluation.py"}
        require(required.issubset(declared), "Frozen research protocol must hash its model and evaluation dependencies")
        cache = self.cache_lineage() if "input_cache_manifest_sha256" in frozen else None
        if "final_data_pipeline.py" not in declared:
            require(cache is not None and "load_research_inputs.py" in declared,
                    "An omitted unused extraction-module hash requires explicit verified cached-input lineage and loader hash")
        code = []
        for name, digest in sorted(declared.items()):
            require(Path(name).name == name and bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"Invalid frozen implementation digest/name: {name}")
            candidates = [self.root / "scripts" / name, self.root / "source" / name, self.root / "code" / name, self.root / name,
                          self.root.parent / "afml_research_source/scripts" / name,
                          Path(__file__).resolve().parent / name]
            path = next((p for p in candidates if p.is_file()), None)
            if path:
                self.artifact(path, "frozen implementation source")
                require(audit.sha256(path) == digest, f"Research source differs from frozen hash: {path}")
            code.append({"module": name, "expected_sha256": digest, "path": str(path.resolve()) if path else None,
                         "status": "hash verified" if path else "declared frozen hash only; source unavailable"})
        for name, digest in engine["code_sha256"].items():
            require(declared.get(name) == digest, f"Engine/frozen implementation hashes differ for {name}")
        return {"research_schema_version": 1, "methods": len(self.expected_methods), "reported_device": engine["device"],
                "fingerprint": engine["fingerprint"], "frozen_utc": frozen["frozen_utc"], "implementation": code,
                "cached_input_lineage": cache,
                "optimizer_design": "Primary matched and meta final adaptation use SGD; Adam arms are secondary final-adaptation sensitivity arms, not differentiable Adam meta-training."}

    def cache_lineage(self) -> dict:
        expected = self.documents["frozen_protocol.json"]["input_cache_manifest_sha256"]
        output_path = self.artifact(self.root / "data/INPUT_MANIFEST.json")
        output = json.loads(output_path.read_text())
        upstream_path = self.root / "data/UPSTREAM_INPUT_MANIFEST.json"
        if upstream_path.is_file():
            self.artifact(upstream_path)
            require(audit.sha256(upstream_path) == expected, "Upstream input manifest differs from the frozen cache-lineage hash")
            upstream = json.loads(upstream_path.read_text())
        else:
            require(audit.sha256(output_path) == expected, "Cached output manifest changed without a frozen upstream manifest")
            upstream = output
        require(set(output) == set(upstream), "Output and upstream cached-input inventories differ")
        required_inputs = {"observations.csv.gz", "forecast_examples.csv.gz", "cohort.csv", "raw_integrity.csv", "data_summary.json", "protocol.json"}
        require(required_inputs.issubset(output), "Cache manifest omits a required source/feature/protocol artifact")
        transformed = []
        for name, entry in output.items():
            require(Path(name).name == name, "Cached-input manifest contains an unsafe path")
            path = self.artifact(self.root / "data" / name)
            require(path.stat().st_size == int(entry["bytes"]) and audit.sha256(path) == entry["sha256"], f"Cached output hash/bytes differ: {path}")
            prior = upstream[name]
            if entry["sha256"] == prior["sha256"]:
                require(int(entry["bytes"]) == int(prior["bytes"]), f"Cached unchanged input has different bytes: {name}")
                continue
            require(name.endswith(".gz") and "uncompressed_sha256" in prior and "uncompressed_bytes" in prior,
                    f"Cached input changed without an explicit gzip-content alias: {name}")
            digest, byte_count = hashlib.sha256(), 0
            with gzip.open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    digest.update(block)
                    byte_count += len(block)
            require(digest.hexdigest() == prior["uncompressed_sha256"] and byte_count == int(prior["uncompressed_bytes"]),
                    f"Recompressed cache bytes differ from frozen upstream content: {name}")
            transformed.append(name)
        return {"frozen_upstream_manifest_sha256": expected, "verified_files": len(output),
                "gzip_container_changes_with_identical_uncompressed_content": transformed,
                "raw_source_connection": "Established by the separate required workbook/support, observation, cohort and forecast-feature audit checks."}

    def donor_frames(self, examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        donor = examples[examples.city.isin(self.donors) & examples.issue_date.ge(pd.Timestamp("2020-01-01"))]
        full = donor[donor.date.le(audit.CUTOFF)]
        preliminary = donor[donor.date.le(pd.Timestamp("2022-12-31")) & ~donor.city.isin(audit.DEVELOPMENT)]
        tuning = donor[donor.date.between("2023-01-04", "2023-12-31") & donor.issue_date.ge(pd.Timestamp("2023-01-01"))]
        return full, preliminary, tuning

    def expected_task(self, city: str, budget: int, examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        config = self.documents["models/engine_metadata.json"]["config"]
        require(city in audit.DEVELOPMENT, "Adaptation candidate names a non-development/target city")
        part = examples[examples.city.eq(city)]
        query = part[part.date.between("2023-01-04", "2023-12-31") & part.issue_date.ge(pd.Timestamp("2023-01-01"))]
        if config.get("pseudo_query_days") is not None:
            query = query[query.date.isin(sorted(query.date.unique())[:int(config["pseudo_query_days"])])]
        last = query.date.min() - pd.Timedelta(days=int(config["purge_days"]) + 1)
        available = part[part.date.le(min(last, pd.Timestamp("2022-12-31"))) & part.issue_date.ge(pd.Timestamp("2020-01-01"))]
        selected = sorted(available.date.unique())[-budget:]
        require(len(selected) == budget, f"Insufficient donor support for {city}/{budget}")
        return available[available.date.isin(selected)], query

    def training_fields(self, row: dict, training: pd.DataFrame) -> None:
        expected = labels(training)
        require(row["target_validation_used"] is False, "A research fit used target validation")
        for name, value in (("train_examples", len(training)), ("train_label_dates", len(expected)), ("train_cities", training.city.nunique())):
            require(int(row[name]) == value, f"{row['_artifact']}: {name} differs from independent donor reconstruction")
        require(row["train_city_names"] == sorted(training.city.unique()), f"{row['_artifact']}: fitting cities differ from permitted donors")
        require(row["train_first_label"] == expected[0] and row["train_last_label"] == expected[-1]
                and row["train_label_dates_sha256"] == label_digest(expected), f"{row['_artifact']}: fitting label bounds/digest differ")
        require(0 < int(row["selected_step"]) <= int(row["actual_steps"]) <= int(row["requested_steps"]), "Research source fit has invalid actual/selected/requested update counts")

    def source_fitting_integrity(self, log: pd.DataFrame, examples: pd.DataFrame) -> dict:
        config = self.documents["models/engine_metadata.json"]["config"]
        full, preliminary, _ = self.donor_frames(examples)
        nondevelopment = sorted(set(self.donors) - set(audit.DEVELOPMENT))
        ordered = sorted(nondevelopment, key=lambda value: hashlib.sha256(f"diversity6:{value}".encode()).hexdigest())
        restricted = sorted(ordered[:6])
        diversity = self.documents["models/diversity_donors.json"]
        require(diversity["restricted_cities"] == restricted and set(diversity["all_cities"]) == set(self.donors), "Diversity6 membership is not the deterministic outcome-independent donor subset")
        hpo_seed = int(config["hpo_seed"])
        source = [row for row in self.records if row["stage"] != "target_fit"]
        frame = pd.DataFrame(source)
        adaptation_labels = [f"deep_{a}_{mode}" for a in DEEP for mode in ("local", "finetune")]
        adaptation_labels += ["matched_gru_local", "matched_gru_finetune"]
        adaptation_labels += [f"{algorithm}_gru_{optimizer}" for algorithm in ("maml", "fomaml", "reptile") for optimizer in ("sgd", "adam")]
        grids = {
            "donor_hpo_source": (["method", "candidate", "seed"], [(a, i, hpo_seed) for a in (*DEEP, "manual_gru") for i in range(4)]),
            "donor_hpo_development": (["method", "source_candidate", "development_task", "seed"], [(a, i, task, hpo_seed) for a in (*DEEP, "manual_gru") for i in range(4) for task in range(12)]),
            "donor_hpo_meta": (["method", "candidate", "seed"], [(a, i, hpo_seed) for a in ("maml", "fomaml", "reptile") for i in range(4)]),
            "donor_fixed_refit": (["method", "seed"], [(a, s) for a in (*[f"deep_{a}" for a in DEEP], "matched_gru") for s in audit.SEEDS]),
            "donor_meta_fixed_refit": (["method", "seed"], [(a, s) for a in ("maml", "fomaml", "reptile") for s in audit.SEEDS]),
            "donor_diversity_fixed_refit": (["method", "seed"], [(a, s) for a in ("diversity6_pooled", "diversity6_maml") for s in audit.SEEDS]),
            "donor_adaptation_candidate": (["method", "candidate", "city", "budget", "seed"], [(a, i, city, b, hpo_seed) for a in adaptation_labels for i in range(4) for city in audit.DEVELOPMENT for b in audit.BUDGETS]),
        }
        for stage, (keys, expected) in grids.items():
            actual = frame[frame.stage.eq(stage)].copy()
            exact_keys(actual, pd.DataFrame(expected, columns=keys), keys, f"Research {stage} grid")
        for row in source:
            stage = row["stage"]
            if stage in ("donor_adaptation_candidate", "donor_hpo_development"):
                city = row["city"] if stage == "donor_adaptation_candidate" else row["train_city_names"][0]
                budget = int(row["budget"]) if stage == "donor_adaptation_candidate" else int(row["train_label_dates"])
                training, query = self.expected_task(city, budget, examples)
                require(pd.Timestamp(row["query_first"]) == query.date.min() and pd.Timestamp(row["query_last"]) == query.date.max(), "Research adaptation query bounds differ from donor 2023 dates")
                gap = int((query.date.min() - training.date.max()).days) - 1
                require(gap >= 9 and int(row["purge_intervening_days"]) == gap, "Research adaptation purge is incorrect")
            elif stage in ("donor_hpo_source", "donor_hpo_meta"):
                training = preliminary
            elif stage == "donor_diversity_fixed_refit":
                training = full[full.city.isin(restricted)]
            else:
                training = full
            self.training_fields(row, training)
            if "refit" in stage:
                require(int(row["actual_steps"]) == int(row["requested_steps"])
                        and int(row["selected_step"]) == int(row["requested_steps"]), "A final donor refit stopped or selected on evaluation data")
        # Frozen all-donor update counts/settings must be reused in diversity6.
        selections = self.documents["models/source_selections.json"]
        for row in source:
            if row["stage"] not in ("donor_fixed_refit", "donor_meta_fixed_refit", "donor_diversity_fixed_refit"):
                continue
            kind = "meta" if row["method"] in ("maml", "fomaml", "reptile", "diversity6_maml") else "supervised"
            name = "maml" if row["method"] == "diversity6_maml" else "manual_gru" if row["method"] in ("matched_gru", "diversity6_pooled") else row["method"].removeprefix("deep_")
            selected = selections[kind][name]
            require(int(row["requested_steps"]) == int(selected["steps"]), "Final source/diversity fit differs from donor-selected duration")
        return {"source_records": len(source), "development_candidates": len(frame[frame.stage.eq("donor_adaptation_candidate")]),
                "diversity6_donors": restricted, "donor_train_city_lists_and_label_date_digests_checked": True}

    def target_label_and_optimizer_integrity(self) -> dict:
        examples = self.tables["data/forecast_examples.csv.gz"]
        settings = self.documents["models/frozen_adaptation_settings.json"]
        specs = self.documents["models/selected_model_specs.json"]
        for row in self.records:
            if row["stage"] != "target_fit":
                continue
            method, city, budget = row["method"], row["city"], int(row["budget"])
            part = examples[examples.city.eq(city) & examples.date.le(audit.CUTOFF)]
            expected = labels(part)[-budget:]
            require(row["support_label_dates"] == expected and len(set(expected)) == budget,
                    "Research target record does not list exactly the latest eligible B fitting labels")
            require(row["support_label_dates_sha256"] == row["train_label_dates_sha256"] == label_digest(expected), "Research target label digest differs from source reconstruction")
            require(row["train_city_names"] == [city] and int(row["train_cities"]) == 1, "Target support mixes cities")
            if method in BASELINES:
                require(row["source_key"] is None and row["model_spec"] is None and row["adaptation_preset"] is None,
                        "A baseline unexpectedly uses a neural source/adaptation preset")
                require(row["baseline_preset"] == settings.get(method), "A baseline target preset differs from frozen donor selection")
                continue
            preset = settings.get(method)
            require(row["adaptation_preset"] == preset, "Target adaptation preset differs from frozen donor settings")
            expected_spec = specs[method.split("_")[1]] if method.startswith("deep_") else specs["manual_gru"]
            require(row["model_spec"] == expected_spec, "Target model spec differs from frozen selected architecture")
            expected_source = (None if method.endswith("_local") else method.split("_")[1] if method.startswith("deep_")
                               else "manual_gru" if method.startswith("matched_gru_")
                               else "diversity6_maml" if method == "diversity6_maml_sgd"
                               else "diversity6_pooled" if method.startswith("diversity6_") else method.split("_")[0])
            require(row["source_key"] == expected_source, "Target source-initialization key differs from its research arm")
            if method.startswith("matched_gru_") or "_gru_sgd" in method or method == "diversity6_maml_sgd" or method == "diversity6_finetune":
                if preset is not None:
                    require(preset["optimizer"] == row["optimizer"] == "sgd", "A matched/primary meta target arm does not use native SGD")
            if method.endswith("_gru_adam"):
                require(preset["optimizer"] == row["optimizer"] == "adam", "A secondary meta Adam arm uses a different optimizer")
            expected_steps = int(preset["steps"]) if preset else 0
            require(int(row["requested_steps"]) == int(row["actual_steps"]) == int(row["selected_step"]) == expected_steps,
                    "Target neural update count differs from donor-frozen schedule")
            if not method.startswith("deep_"):
                hidden, layers = int(expected_spec["hidden"]), int(expected_spec["layers"])
                expected_parameters = sum(3*hidden*((6 if index == 0 else hidden)+hidden+2) for index in range(layers)) + hidden + 1
                require(int(row["parameters"]) == expected_parameters, "Matched/meta/diversity GRU parameter counts differ")
        require(settings["diversity6_finetune"] == settings["matched_gru_finetune"]
                and settings["diversity6_maml_sgd"] == settings["maml_gru_sgd"], "Diversity6 uses independently retuned adaptation settings")
        return {"target_fit_records_checked": sum(row["stage"] == "target_fit" for row in self.records),
                "exact_target_support_lists_and_digests_checked": True,
                "primary_matched_meta_optimizer": "sgd", "secondary_meta_optimizer": "adam"}

    def donor_selection_integrity(self) -> dict:
        """Recompute candidate rankings from saved donor evaluation records."""
        engine = self.documents["models/engine_metadata.json"]
        config = engine["config"]
        settings = self.documents["models/frozen_adaptation_settings.json"]
        specs = self.documents["models/selected_model_specs.json"]
        selections = self.documents["models/source_selections.json"]
        fit_by_stage = pd.DataFrame(self.records)
        weight = float(config["hpo_development_weight"])

        def selection(name: str) -> dict:
            return json.loads(self.artifact(self.root / "models/selection" / f"{name}.json").read_text())

        for architecture in (*DEEP, "manual_gru"):
            ranked = selection(f"source_{architecture}")
            trials = ranked["trials"]
            require({int(trial["candidate"]) for trial in trials} == set(range(4)) and len(trials) == 4,
                    f"Missing source candidates for {architecture}")
            fitted = fit_by_stage[fit_by_stage.stage.eq("donor_hpo_source") & fit_by_stage.method.eq(architecture)]
            development = fit_by_stage[fit_by_stage.stage.eq("donor_hpo_development") & fit_by_stage.method.eq(architecture)]
            for trial in trials:
                index = int(trial["candidate"])
                row = fitted.loc[fitted.candidate.eq(index)].iloc[0]
                scores = development.loc[development.source_candidate.eq(index), "macro_horizon_mae"]
                require(len(scores) == 12, "Incomplete held-out donor development scores")
                require(abs(scores.mean() - trial["heldout_development_adapted_mae"]) <= audit.ATOL,
                        "Held-out city mean differs from the source trial")
                score = (1 - weight) * row.donor_validation_mae + weight * scores.mean()
                require(abs(score - trial["score"]) <= audit.ATOL and int(row.selected_step) == trial["selected_steps"],
                        "Source trial score or selected updates differs from the source fitting record")
            chosen = min(trials, key=lambda trial: (trial["score"], trial["candidate"]))
            require(int(ranked["selected_candidate"]) == chosen["candidate"],
                    "Source setting was not selected by the best reconstructed donor score")
            require(specs[architecture] == chosen["model_spec"] and selections["supervised"][architecture] ==
                    {"preset": chosen["preset"], "steps": chosen["selected_steps"]},
                    "Source refit settings differ from the selected donor candidate")

        for algorithm in ("maml", "fomaml", "reptile"):
            ranked = selection(f"meta_{algorithm}")
            trials = ranked["trials"]
            require({int(trial["candidate"]) for trial in trials} == set(range(4)) and len(trials) == 4,
                    f"Missing meta candidates for {algorithm}")
            fitted = fit_by_stage[fit_by_stage.stage.eq("donor_hpo_meta") & fit_by_stage.method.eq(algorithm)]
            for trial in trials:
                row = fitted.loc[fitted.candidate.eq(trial["candidate"])].iloc[0]
                score = ((1 - weight) * row.chronological_donor_adapted_mae +
                         weight * row.heldout_development_adapted_mae)
                require(abs(score - trial["score"]) <= audit.ATOL and int(row.selected_step) == trial["selected_steps"],
                        "Meta trial score or selected updates differs from the source fitting record")
            chosen = min(trials, key=lambda trial: (trial["score"], trial["candidate"]))
            require(ranked["selected_candidate"] == chosen["candidate"] and selections["meta"][algorithm] ==
                    {"preset": chosen["preset"], "steps": chosen["selected_steps"]},
                    "Meta refit settings differ from the minimum donor score")

        lookup = {"ridge_local": "ridge_presets", "histgb_local": "histgb_presets",
                  "randomforest_local": "randomforest_presets"}
        examples = self.tables["data/forecast_examples.csv.gz"]
        for label, selected in settings.items():
            if label.startswith("diversity6_"):
                continue
            baseline = label in lookup
            result = selection(label if baseline else f"adapt_{label}")
            table = pd.DataFrame(result["scores"])
            ranking = table.groupby("candidate", sort=True).macro_horizon_mae.mean()
            chosen = int(ranking.idxmin())
            grid = config[lookup[label]] if baseline else None
            require(result["selected_candidate"] == chosen and result["selected_preset"] == selected,
                    f"Adaptation setting is inconsistent with donor-only ranking for {label}")
            size = len(grid) if baseline else 4
            expected = pd.DataFrame([(index, city, budget) for index in range(size)
                                     for city in audit.DEVELOPMENT for budget in audit.BUDGETS],
                                    columns=["candidate", "city", "budget"])
            exact_keys(table, expected, ["candidate", "city", "budget"], f"{label} donor ranking grid")
            if baseline:
                require(selected == grid[chosen], "Selected tabular preset differs from its frozen grid")
                for row in table.itertuples(index=False):
                    support, query = self.expected_task(row.city, int(row.budget), examples)
                    require(int(row.seed) == int(config["hpo_seed"]) and row.support_examples == len(support)
                            and row.query_examples == len(query) and pd.Timestamp(row.support_first) == support.date.min()
                            and pd.Timestamp(row.support_last) == support.date.max() and pd.Timestamp(row.query_first) == query.date.min()
                            and pd.Timestamp(row.query_last) == query.date.max() and row.purge_intervening_days >= 9,
                            "Tabular tuning uses an incorrect donor support or query window")
        return {"supervised_search_configurations": 24, "meta_search_configurations": 12,
                "donor_only_rankings_recomputed": True, "tabular_candidate_city_dates_reconstructed": True}

    def meta_episode_integrity(self) -> dict:
        config = self.documents["models/engine_metadata.json"]["config"]
        full, preliminary, _ = self.donor_frames(self.tables["data/forecast_examples.csv.gz"])
        restricted = self.documents["models/diversity_donors.json"]["restricted_cities"]
        expected_paths, total, minimum_gap = set(), 0, None
        for row in self.records:
            if "episodes_file" not in row:
                continue
            algorithm = "maml" if row["method"] == "diversity6_maml" else row["method"]
            require(algorithm in ("maml", "fomaml", "reptile"), "An episode file is attributed to an unknown meta algorithm")
            training = preliminary if row["stage"] == "donor_hpo_meta" else full
            if row["method"] == "diversity6_maml":
                training = training[training.city.isin(restricted)]
            by_city = {city: pd.DatetimeIndex(part.date.unique()).sort_values() for city, part in training.groupby("city")}
            relative = Path(row["episodes_file"])
            require(not relative.is_absolute() and relative.parts[0] == "meta_episodes" and ".." not in relative.parts,
                    "Meta episode path escapes the model directory")
            path = (self.root / "models" / relative).resolve()
            require(path not in expected_paths, "Multiple fitting records point at the same episode file")
            expected_paths.add(path)
            frame = self.csv(str(Path("models") / relative), empty=True)
            require(len(frame) == int(row["episode_rows"]) == int(row["actual_steps"]) * int(config["meta_tasks"]), "Research meta episode count differs from actual outer updates")
            require(not frame.empty, "A research meta fit has no episodes")
            columns(frame, ["city", "budget", "support_first", "support_last", "query_first", "query_last", "purge_intervening_days",
                            "support_examples", "query_examples", "step", "task", "algorithm", "seed", "support_label_dates_sha256",
                            "query_label_dates_sha256", "full_second_order", "inner_steps", "inner_lr", "support_batch_examples", "query_batch_examples"], "research episodes")
            audit.dates(frame, ["support_first", "support_last", "query_first", "query_last"], "research episodes")
            audit.numbers(frame, ["budget", "purge_intervening_days", "support_examples", "query_examples", "step", "task",
                                  "seed", "inner_steps", "support_batch_examples", "query_batch_examples"], "research episodes", True)
            audit.numbers(frame, ["inner_lr"], "research episodes")
            require(frame.algorithm.eq(algorithm).all() and frame.seed.eq(int(row["seed"])).all(), "Meta episode algorithm/seed identity differs")
            require(set(frame.city).issubset(by_city) and frame.budget.isin(audit.BUDGETS).all(), "Meta episode uses a target/excluded donor or an unknown budget")
            require(frame.full_second_order.astype(str).str.lower().eq(str(algorithm == "maml").lower()).all(), "Meta episode second-order declaration differs from its algorithm")
            preset = (config["meta_grid"][int(row["candidate"])] if row["stage"] == "donor_hpo_meta"
                      else self.documents["models/source_selections.json"]["meta"][algorithm]["preset"])
            require(frame.inner_steps.eq(int(preset["inner_steps"])).all()
                    and frame.inner_lr.eq(float(preset["inner_lr"])).all(), "Research episode inner-update settings differ from frozen donor selection")
            require(not frame.duplicated(["step", "task"]).any() and frame.step.between(1, int(row["actual_steps"])).all()
                    and frame.task.between(0, int(config["meta_tasks"])-1).all(), "Missing/duplicate research meta step/task grid")
            gap = (frame.query_first - frame.support_last).dt.days - 1
            require(gap.ge(9).all() and frame.purge_intervening_days.eq(gap).all(), "Research meta support/query purge is less than nine actual intervening days")
            for episode in frame.itertuples(index=False):
                available = by_city[episode.city]
                support = available[available <= episode.query_first-pd.Timedelta(days=int(config["purge_days"])+1)][-int(episode.budget):]
                query = available[available >= episode.query_first][:int(config["meta_query_days"])]
                sd = [value.strftime("%Y-%m-%d") for value in support]
                qd = [value.strftime("%Y-%m-%d") for value in query]
                require(len(sd) == int(episode.budget) and support[0] == episode.support_first and support[-1] == episode.support_last
                        and episode.support_label_dates_sha256 == label_digest(sd), "Research episode support labels/digest differ from eligible purged B dates")
                require(len(qd) == int(config["meta_query_days"]) and query[0] == episode.query_first and query[-1] == episode.query_last
                        and episode.query_label_dates_sha256 == label_digest(qd), "Research episode query labels/digest differ from eligible donor dates")
                require(episode.support_examples == 3*len(sd) and episode.query_examples == 3*len(qd), "Incorrect research episode three-horizon counts")
                require(episode.support_batch_examples == min(int(config["meta_batch_size"]), 3*len(sd))
                        and episode.query_batch_examples == min(int(config["meta_batch_size"]), 3*len(qd)), "Research meta minibatch count differs from frozen policy")
            total += len(frame)
            minimum_gap = int(gap.min()) if minimum_gap is None else min(minimum_gap, int(gap.min()))
        require(total > 0, "Research output contains no meta episodes")
        require(expected_paths == {path.resolve() for path in (self.root / "models/meta_episodes").glob("*.csv.gz")}, "Missing/unreferenced research meta episode files")
        expected_fits = 3*4 + 3*3 + 3
        require(len(expected_paths) == expected_fits, "Missing a meta HPO/refit/diversity episode log")
        return {"episode_files": len(expected_paths), "episodes": total, "minimum_actual_intervening_days": minimum_gap,
                "source_eligible_support_and_query_date_digests_checked": True,
                "full_second_order_declarations_checked": True}

    def runtime_derivative_evidence(self) -> dict:
        value = self.documents["models/second_order_gradient_check.json"]
        engine = self.documents["models/engine_metadata.json"]
        require(value["passed"] is True and value["create_graph_full_maml"] is True
                and value["create_graph_fomaml"] is False and value["cudnn_used_for_meta"] is False,
                "Recorded backend derivative check did not establish the declared meta graph")
        require(value["device"] == engine["device"], "Derivative runtime check used a different device from fitting")
        for name, lower in (("full_maml_vs_fomaml_gradient_l2", 1e-7), ("full_gradient_l2", 1e-7),
                            ("body_gradient_l2", 1e-8), ("body_parameter_absolute_change", 1e-8)):
            require(np.isfinite(float(value[name])) and float(value[name]) > lower, f"Runtime derivative check has no meaningful {name}")
        require(abs(float(value["forward_query_loss_maml"]) - float(value["forward_query_loss_fomaml"])) <= 1e-6,
                "Runtime MAML/FOMAML gradient comparison used different adapted forward losses")
        require(float(value["manual_native_gru_max_abs_difference"]) <= 1e-5, "Explicit GRU gates differ from native GRU in runtime check")
        return {"reported_backend_check_passed": True, "device": value["device"],
                "evidence_scope": "Checks a recorded synthetic test on the fitting backend; this verifier does not import or execute model code."}


def verify_research_study(root: str | Path, output: str | Path, source: str | Path | None = None) -> dict:
    instance = ResearchAudit(Path(root), Path(output), Path(source) if source else None)
    for name, operation in (("Required research artifacts and explicit method inventory", instance.load),
                            ("Original source checksums and observation inventory", instance.source_integrity),
                            ("Cohort selection, whole-city exclusions and all-station buffer", instance.cohort_integrity),
                            ("Frozen research protocol, metadata and code hashes", instance.protocol_integrity),
                            ("Every forecast feature, truth and eligible date against observations", instance.forecast_integrity),
                            ("Complete research prediction grid, chronology and exact baselines", instance.prediction_integrity),
                            ("Nested budgets and research source fitting chronology", instance.budget_and_fitting_integrity),
                            ("Explicit target labels, matched capacity and native adaptation", instance.target_label_and_optimizer_integrity),
                            ("Donor-only candidate rankings and stored fitting choices", instance.donor_selection_integrity),
                            ("Donor-only research episodes, label digests and nine-day purges", instance.meta_episode_integrity),
                            ("Recorded selected-backend derivative evidence", instance.runtime_derivative_evidence),
                            ("Independent research city/seed and macro metrics", instance.metric_integrity)):
        instance.run_check(name, operation)
    return instance.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    result = verify_research_study(args.root, args.output, args.source)
    print(f"{result['verdict']}: {args.output.resolve() / 'verification.json'}")


if __name__ == "__main__":
    main()
