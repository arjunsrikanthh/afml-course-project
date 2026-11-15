"""Paired evaluation for the fixed-cohort, daily-AQI forecasting experiment.

The unit of generalization is a target city, not a random seed or an overlapping
forecast window.  ``evaluate`` requires a complete prediction grid and refuses to
silently discard dates to make models comparable.  It does no fitting or tuning.

Required columns: city, station_id, date (target day), issue_date, horizon,
budget, seed, method, y_true, y_pred, last, period.  Recommended periods are
``secondary_2024`` and ``test_2025``; 2024 is not a tuning/validation split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVALUATION_REQUIRED = [
    "city", "station_id", "date", "issue_date", "horizon", "budget", "seed",
    "method", "y_true", "y_pred", "last", "period",
]
CASE = ["period", "city", "station_id", "budget", "horizon", "seed", "method"]
CITY = ["period", "city", "station_id", "budget", "horizon", "method"]
MACRO = ["period", "budget", "horizon", "method"]
EXPECTED_METHODS = [
    "persistence", "seasonal7", "mean7", "ridge_local", "histgb_local", "randomforest_local",
    *[f"{architecture}_{mode}" for architecture in ["gru", "lstm", "tcn", "transformer", "mlp"]
      for mode in ["local", "pooled_zero", "finetune"]],
    "reptile_gru", "fomaml_gru",
]
EXPECTED_PERIODS = ["secondary_2024", "test_2025"]
EXPECTED_TARGETS = ["Amaravati", "Guwahati", "Ahmedabad", "Pithampur", "Ludhiana", "Udaipur"]
DEFAULT_PLOT_METHODS = [
    "persistence", "seasonal7", "mean7", "ridge_local", "histgb_local",
    "randomforest_local", "gru_local", "gru_pooled_zero", "gru_finetune",
    "lstm_finetune", "tcn_finetune", "transformer_finetune", "mlp_finetune",
    "reptile_gru", "fomaml_gru",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {"rows": len(value), "columns": list(value.columns)}
    if isinstance(value, pd.Series):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def _constant(df: pd.DataFrame, keys: list[str], columns: list[str], name: str) -> None:
    counts = df.groupby(keys, dropna=False, observed=True)[columns].nunique(dropna=False)
    bad = counts.gt(1).any(axis=1)
    if bad.any():
        raise ValueError(f"{name}: inconsistent values at {counts.index[bad].tolist()[:3]}")


def _prepare(predictions: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(EVALUATION_REQUIRED) - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction columns missing: {missing}")
    if predictions.empty:
        raise ValueError("No predictions to evaluate.")
    df = predictions[EVALUATION_REQUIRED].copy()
    if df.isna().any().any():
        raise ValueError(f"Prediction values must be present: {df.isna().sum().to_dict()}")
    for col in ["date", "issue_date"]:
        values = pd.to_datetime(df[col], errors="raise")
        if getattr(values.dt, "tz", None) is not None:
            raise ValueError(f"{col} must use timezone-naive calendar dates.")
        if not values.eq(values.dt.normalize()).all():
            raise ValueError(f"{col} contains times; supply the daily calendar label explicitly.")
        df[col] = values
    for col in ["horizon", "budget", "seed"]:
        values = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(values).all() or not values.eq(np.floor(values)).all():
            raise ValueError(f"{col} must contain finite integers.")
        df[col] = values.astype("int64")
    if (df.horizon <= 0).any() or (df.budget <= 0).any():
        raise ValueError("Horizons and fitting-label budgets must be positive.")
    if not (df.date - df.issue_date).dt.days.eq(df.horizon).all():
        raise ValueError("date must equal issue_date plus horizon calendar days.")
    for col in ["y_true", "y_pred", "last"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        if not np.isfinite(df[col]).all():
            raise ValueError(f"{col} includes a nonfinite number.")
    for col in ["city", "station_id", "method", "period"]:
        df[col] = df[col].astype(str)
    identity = ["period", "city", "budget", "horizon", "seed", "method", "date"]
    if df.duplicated(identity).any():
        raise ValueError("Duplicate method/seed/city/budget/horizon/target-date predictions.")
    _constant(df, ["city"], ["station_id"], "One selected station is required per target city")
    _constant(df, ["station_id"], ["city"], "A selected station cannot name multiple cities")
    _constant(df, ["city", "date", "horizon"],
              ["station_id", "issue_date", "y_true", "last", "period"],
              "Truth and observed issue-day context must agree across methods, seeds and budgets")
    _constant(df, ["city", "date"], ["y_true", "station_id"],
              "Observed target truth must agree across forecast horizons")

    # Every forecast key must occur for every method.  Intersection-only scoring
    # can hide failed predictions and is deliberately not performed here.
    methods = sorted(df.method.unique())
    expected_methods = protocol.get("expected_methods", EXPECTED_METHODS)
    if isinstance(expected_methods, (int, np.integer)):
        if int(expected_methods) != len(EXPECTED_METHODS):
            raise ValueError("expected_methods must be an explicit name list, or 23 for the canonical method set.")
        expected_methods = EXPECTED_METHODS
    if isinstance(expected_methods, str):
        raise ValueError("expected_methods must be a list of method names, not a string.")
    expected_methods = set(map(str, expected_methods))
    if set(methods) != expected_methods:
        raise ValueError(f"Method set differs from frozen protocol; missing={sorted(expected_methods - set(methods))}, "
                         f"unexpected={sorted(set(methods) - expected_methods)}")
    expected_periods = set(map(str, protocol.get("expected_periods", EXPECTED_PERIODS)))
    if set(df.period.unique()) != expected_periods:
        raise ValueError(f"Period set differs from frozen protocol: expected {sorted(expected_periods)}, "
                         f"received {sorted(df.period.unique())}")
    cutoff = pd.Timestamp(protocol.get("latest_fit_cutoff", protocol.get("target_fitting_cutoff", "2023-12-31")))
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("latest_fit_cutoff must be a timezone-naive fitting calendar date.")
    if not df.issue_date.gt(cutoff).all():
        raise ValueError(f"Every evaluation issue_date must occur after the latest fitting cutoff {cutoff.date()}.")
    pair_keys = ["period", "city", "budget", "horizon", "seed", "date"]
    method_counts = df.groupby(pair_keys, observed=True).method.nunique()
    if not method_counts.eq(len(methods)).all():
        bad = method_counts[method_counts.ne(len(methods))]
        raise ValueError(f"Methods do not share exact evaluation keys; examples: {bad.head(3).to_dict()}")
    # A missing entire seed, budget, city or horizon must not escape the rowwise check.
    expected_seeds = set(protocol.get("seeds", df.seed.unique().tolist()))
    grids = df.groupby(["period", "city", "budget", "horizon"], observed=True).seed.agg(set)
    if not grids.map(lambda s: s == expected_seeds).all():
        raise ValueError("Each period/city/budget/horizon must have the same complete seed set.")
    expected_budgets = set(protocol.get("budgets", df.budget.unique().tolist()))
    expected_horizons = set(protocol.get("horizons", df.horizon.unique().tolist()))
    expected_cities = set(map(str, protocol.get("target_cities", protocol.get("targets", EXPECTED_TARGETS))))
    for period, part in df.groupby("period", observed=True):
        expected = pd.MultiIndex.from_product(
            [sorted(expected_cities), sorted(expected_budgets), sorted(expected_horizons)],
            names=["city", "budget", "horizon"],
        )
        actual = pd.MultiIndex.from_frame(part[["city", "budget", "horizon"]].drop_duplicates())
        if len(actual.symmetric_difference(expected)):
            raise ValueError(f"Incomplete city/budget/horizon grid in period {period}.")
    # Dates must be fixed before fitting, including across label budgets and seeds.
    date_sets = df.groupby(["period", "city", "horizon", "budget", "seed"], observed=True).date.agg(frozenset)
    for _, sets in date_sets.groupby(level=["period", "city", "horizon"]):
        if not all(s == sets.iloc[0] for s in sets):
            raise ValueError("Evaluation dates differ across fitting budgets or random seeds.")
    if protocol.get("common_dates_across_horizons", True):
        horizon_dates = df.groupby(["period", "city", "horizon"], observed=True).date.agg(frozenset)
        for _, sets in horizon_dates.groupby(level=["period", "city"]):
            if not all(s == sets.iloc[0] for s in sets):
                raise ValueError("Target dates must be common across horizons under this protocol.")

    period_years = {"secondary_2024": 2024, "test_2025": 2025,
                    **protocol.get("period_years", {})}
    for period, year in period_years.items():
        dates = df.loc[df.period.eq(period), "date"]
        if len(dates) and not dates.dt.year.eq(int(year)).all():
            raise ValueError(f"Period {period} includes target dates outside {year}.")
    persistence = protocol.get("persistence_method", "persistence")
    if persistence not in methods and "persist" in methods:
        persistence = "persist"
    if persistence not in methods:
        raise ValueError(f"The required persistence reference {persistence!r} is absent.")
    base = df[df.method.eq(persistence)]
    if not np.array_equal(base.y_pred.to_numpy(), base["last"].to_numpy()):
        raise ValueError("Persistence forecasts must equal the exact observed issue-day AQI.")

    source_checked = False
    if protocol.get("truth_reference") is not None:
        ref = pd.DataFrame(protocol["truth_reference"]).copy()
        ref_cols = ["city", "station_id", "date", "issue_date", "horizon", "y_true", "last"]
        if not set(ref_cols).issubset(ref):
            raise ValueError(f"truth_reference requires {ref_cols}")
        ref = ref[ref_cols]
        for col in ["date", "issue_date"]:
            ref[col] = pd.to_datetime(ref[col], errors="raise")
        for col in ["city", "station_id"]:
            ref[col] = ref[col].astype(str)
        join = ["city", "station_id", "date", "horizon"]
        if ref.duplicated(join).any():
            raise ValueError("truth_reference contains duplicate station/target-date/horizon keys.")
        scored = df[ref_cols].drop_duplicates()
        check = scored.merge(ref, on=join, how="outer", validate="one_to_one", suffixes=("", "_source"), indicator=True)
        if not check._merge.eq("both").all():
            raise ValueError("Evaluated dates must exactly match truth_reference; omitted or extra source-context keys were found.")
        for col in ["issue_date", "y_true", "last"]:
            if not check[col].eq(check[col + "_source"]).all():
                raise ValueError(f"Evaluated {col} differs from the exact source-context reference.")
        source_checked = True

    df["error"] = df.y_pred - df.y_true
    df["absolute_error"] = df.error.abs()
    df["squared_error"] = df.error ** 2
    checks = {
        "prediction_rows": len(df), "cities": sorted(df.city.unique()),
        "stations": sorted(df.station_id.unique()), "methods": methods,
        "seeds": sorted(map(int, expected_seeds)), "budgets": sorted(map(int, expected_budgets)),
        "horizons": sorted(map(int, expected_horizons)), "periods": sorted(df.period.unique()),
        "expected_methods": sorted(expected_methods), "expected_periods": sorted(expected_periods),
        "latest_fit_cutoff": str(cutoff.date()), "all_issue_dates_after_fit_cutoff": True,
        "complete_method_pairing": True, "constant_truth_and_context": True,
        "common_dates_across_budgets_and_seeds": True,
        "common_target_dates_across_horizons": bool(protocol.get("common_dates_across_horizons", True)),
        "exact_source_reference_checked": source_checked, "persistence_method": persistence,
        "prediction_scale": "Unclipped original AQI units", "tuning_performed_by_evaluator": False,
    }
    return df.sort_values(identity).reset_index(drop=True), checks


def _add_skill(table: pd.DataFrame, keys: list[str], persistence: str) -> pd.DataFrame:
    ref = table.loc[table.method.eq(persistence), keys + ["mae", "rmse"]].rename(
        columns={"mae": "persistence_mae", "rmse": "persistence_rmse"})
    result = table.merge(ref, on=keys, how="left", validate="many_to_one")
    result["skill_mae"] = 1 - result.mae / result.persistence_mae.replace(0, np.nan)
    result["skill_rmse"] = 1 - result.rmse / result.persistence_rmse.replace(0, np.nan)
    return result


def _metrics(df: pd.DataFrame, persistence: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case = df.groupby(CASE, observed=True, as_index=False).agg(
        n_dates=("date", "nunique"), mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"), bias=("error", "mean"))
    case["rmse"] = np.sqrt(case.pop("mse"))
    case = _add_skill(case, [c for c in CASE if c != "method"], persistence)
    # Average losses, not predictions, over seeds before any city/date inference.
    date_keys = ["period", "city", "station_id", "budget", "horizon", "date", "method"]
    daily = df.groupby(date_keys, observed=True, as_index=False).agg(
        issue_date=("issue_date", "first"), y_true=("y_true", "first"),
        last=("last", "first"), mean_prediction=("y_pred", "mean"),
        n_seeds=("seed", "nunique"), absolute_error=("absolute_error", "mean"),
        squared_error=("squared_error", "mean"), error=("error", "mean"))
    city = daily.groupby(CITY, observed=True, as_index=False).agg(
        n_dates=("date", "nunique"), n_seeds=("n_seeds", "min"),
        mae=("absolute_error", "mean"), mse=("squared_error", "mean"), bias=("error", "mean"))
    city["rmse"] = np.sqrt(city.pop("mse"))
    city = _add_skill(city, [c for c in CITY if c != "method"], persistence)
    macro = city.groupby(MACRO, observed=True, as_index=False).agg(
        n_cities=("city", "nunique"), n_city_dates=("n_dates", "sum"),
        mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"),
        mean_city_skill_mae=("skill_mae", "mean"), mean_city_skill_rmse=("skill_rmse", "mean"))
    macro = _add_skill(macro, [c for c in MACRO if c != "method"], persistence)
    wins = city.assign(
        improved=city.mae.lt(city.persistence_mae), worse=city.mae.gt(city.persistence_mae),
        tied=city.mae.eq(city.persistence_mae)).groupby(MACRO, observed=True, as_index=False).agg(
            cities_improved_vs_persistence=("improved", "sum"),
            cities_worse_vs_persistence=("worse", "sum"),
            cities_tied_vs_persistence=("tied", "sum"))
    macro = macro.merge(wins, on=MACRO, validate="one_to_one")
    return case, daily, city, macro


def _comparison_specs(methods: list[str], persistence: str, protocol: dict) -> list[dict]:
    specs = [{"method": m, "reference": persistence, "kind": "method_minus_persistence"}
             for m in methods if m != persistence]
    for method in methods:
        if method.endswith("_pooled_zero"):
            ref = method.removesuffix("_pooled_zero") + "_local"
            if ref in methods:
                specs.append({"method": method, "reference": ref, "kind": "pooled_zero_minus_local"})
        if method.endswith("_finetune"):
            ref = method.removesuffix("_finetune") + "_pooled_zero"
            if ref in methods:
                specs.append({"method": method, "reference": ref, "kind": "finetune_minus_pooled_zero"})
            local = method.removesuffix("_finetune") + "_local"
            if local in methods:
                specs.append({"method": method, "reference": local, "kind": "finetune_minus_local"})
    for method in ["reptile_gru", "fomaml_gru"]:
        if method in methods and "gru_finetune" in methods:
            specs.append({"method": method, "reference": "gru_finetune", "kind": "meta_minus_finetune"})
    extras = protocol.get("comparisons", [])
    if isinstance(extras, dict):
        extras = [{"method": k, "reference": v} for k, v in extras.items()]
    for spec in extras:
        if not isinstance(spec, dict):
            spec = {"method": spec[0], "reference": spec[1]}
        specs.append({"kind": "requested_comparison", **spec})
    unique = {}
    for spec in specs:
        if spec["method"] not in methods or spec["reference"] not in methods:
            raise ValueError(f"Unknown method in comparison: {spec}")
        unique.setdefault((spec["method"], spec["reference"]), spec)
    return list(unique.values())


def _calendar_bootstrap_weights(dates: pd.DatetimeIndex, n: int, block: int, seed: int) -> np.ndarray:
    """Resample overlapping contiguous calendar blocks, retaining paired gaps.

    Missing calendar days occupy their real positions and contribute no loss.
    The same observed-date weights are used for every method in a comparison.
    Blocks stay within the city's evaluation calendar (no wraparound).
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    length = int((dates[-1] - dates[0]).days) + 1
    width = min(block, length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, length - width + 1, size=(n, math.ceil(length / width)))
    sampled = (starts[:, :, None] + np.arange(width)).reshape(n, -1)[:, :length]
    flat = sampled + np.arange(n)[:, None] * length
    counts = np.bincount(flat.ravel(), minlength=n * length).reshape(n, length)
    positions = (dates - dates[0]).days.to_numpy()
    weights = counts[:, positions].astype(float)
    den = weights.sum(axis=1)
    if np.any(den == 0):
        raise ValueError("A calendar-block bootstrap sample contains no observed dates; the cohort is too sparse.")
    return weights / den[:, None]


def _paired_comparisons(daily: pd.DataFrame, city: pd.DataFrame, specs: list[dict], protocol: dict) -> pd.DataFrame:
    n = int(protocol.get("n_bootstrap", 1000))
    block = int(protocol.get("bootstrap_block_days", 14))
    seed = int(protocol.get("bootstrap_seed", 20261114))
    if n < 100 or block < 1:
        raise ValueError("Use at least 100 bootstrap resamples and a positive calendar block length.")
    methods = sorted(daily.method.unique())
    col = {m: i for i, m in enumerate(methods)}
    city_rank = {c: i for i, c in enumerate(sorted(daily.city.unique()))}
    period_rank = {p: i for i, p in enumerate(sorted(daily.period.unique()))}
    cache: dict[tuple, np.ndarray] = {}
    rows = []
    for (period, budget, horizon), group in daily.groupby(["period", "budget", "horizon"], observed=True, sort=True):
        cities = sorted(group.city.unique())
        boot_mae = np.zeros((n, len(methods)))
        boot_rmse = np.zeros_like(boot_mae)
        for city_name, part in group.groupby("city", observed=True, sort=True):
            ae = part.pivot(index="date", columns="method", values="absolute_error").reindex(columns=methods).sort_index()
            se = part.pivot(index="date", columns="method", values="squared_error").reindex(columns=methods).sort_index()
            if ae.isna().any().any() or se.isna().any().any():
                raise ValueError("Unpaired method/date losses reached the bootstrap.")
            key = (period, city_name, tuple(ae.index.asi8), n, block)
            if key not in cache:
                # No budget/method/seed term: every comparison uses the same paired resampling.
                cache[key] = _calendar_bootstrap_weights(
                    ae.index, n, block, seed + 1009 * period_rank[period] + 9176 * city_rank[city_name])
            weights = cache[key]
            boot_mae += weights @ ae.to_numpy() / len(cities)
            boot_rmse += np.sqrt(np.maximum(weights @ se.to_numpy(), 0)) / len(cities)
        point = city[(city.period == period) & (city.budget == budget) & (city.horizon == horizon)]
        for spec in specs:
            method, reference = spec["method"], spec["reference"]
            paired = point.loc[point.method.eq(method), ["city", "mae", "rmse"]].merge(
                point.loc[point.method.eq(reference), ["city", "mae", "rmse"]],
                on="city", validate="one_to_one", suffixes=("", "_reference"))
            delta = paired.mae - paired.mae_reference
            mae_draws = boot_mae[:, col[method]] - boot_mae[:, col[reference]]
            rmse_draws = boot_rmse[:, col[method]] - boot_rmse[:, col[reference]]
            lo, hi = np.quantile(mae_draws, [0.025, 0.975])
            rlo, rhi = np.quantile(rmse_draws, [0.025, 0.975])
            rows.append({
                "period": period, "budget": budget, "horizon": horizon, **spec,
                "delta_mae": delta.mean(), "delta_mae_ci_low": lo, "delta_mae_ci_high": hi,
                "delta_rmse": (paired.rmse - paired.rmse_reference).mean(),
                "delta_rmse_ci_low": rlo, "delta_rmse_ci_high": rhi,
                "cities_improved": int(delta.lt(0).sum()), "cities_tied": int(delta.eq(0).sum()),
                "cities_worse": int(delta.gt(0).sum()), "n_cities": len(paired),
                "ci_excludes_zero_in_favor_of_method": bool(hi < 0),
                "bootstrap_resamples": n, "block_calendar_days": block,
                "uncertainty_scope": "Temporal uncertainty conditional on these selected cities; cities not resampled",
            })
    return pd.DataFrame(rows)


def _subgroups(daily: pd.DataFrame, persistence: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    month = daily.date.dt.month
    seasons = np.select([month.isin([6, 7, 8, 9]), month.isin([12, 1, 2])], ["monsoon", "winter"], default="other")
    expanded = pd.concat([
        daily.assign(subgroup_type="season", subgroup=seasons),
        daily.assign(subgroup_type="observed_target_aqi", subgroup=np.where(daily.y_true.ge(200), "AQI >= 200", "AQI < 200")),
    ], ignore_index=True)
    keys = CITY + ["subgroup_type", "subgroup"]
    city = expanded.groupby(keys, observed=True, as_index=False).agg(
        n_dates=("date", "nunique"), mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"), bias=("error", "mean"))
    city["rmse"] = np.sqrt(city.pop("mse"))
    city = _add_skill(city, [c for c in keys if c != "method"], persistence)
    macro_keys = MACRO + ["subgroup_type", "subgroup"]
    macro = city.groupby(macro_keys, observed=True, as_index=False).agg(
        n_cities=("city", "nunique"), n_city_dates=("n_dates", "sum"),
        mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"))
    macro = _add_skill(macro, [c for c in macro_keys if c != "method"], persistence)
    return city, macro


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "No eligible rows."
    view = df[columns].head(max_rows)
    def fmt(x: Any) -> str:
        if isinstance(x, (float, np.floating)):
            return f"{x:.3f}" if np.isfinite(x) else "NA"
        return str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    if len(df) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(df)} rows; complete results are in CSV.")
    return "\n".join(lines)


def _primary_period(periods: list[str], protocol: dict) -> str:
    primary = protocol.get("primary_period", "test_2025")
    return primary if primary in periods else sorted(periods)[-1]


def _cost_summary(protocol: dict) -> pd.DataFrame:
    """Summarize recorded fitting work without multiplying shared donor costs.

    Engine fit_seconds records the fitting routine; source checkpoint selection
    can include validation work. Target block_seconds additionally contains
    prediction/artifact work. The two durations are never added together.
    """
    value = protocol.get("fit_costs")
    if value is None:
        return pd.DataFrame()
    try:
        logs = pd.DataFrame(value).copy()
    except (TypeError, ValueError):
        return pd.DataFrame()
    if logs.empty:
        return pd.DataFrame()
    runtime = next((c for c in ["fit_seconds", "runtime_seconds", "elapsed_seconds", "seconds", "duration_seconds"] if c in logs), None)
    parameters = next((c for c in ["parameters", "trainable_parameters", "parameter_count", "n_params", "params"] if c in logs), None)
    if runtime is None and parameters is None:
        return pd.DataFrame()
    group = [c for c in ["stage", "method", "architecture"] if c in logs]
    if not group:
        group = [c for c in ["phase", "model", "name"] if c in logs]
    if not group:
        logs["stage"] = "all recorded fits"
        group = ["stage"]
    for col in group:
        logs[col] = logs[col].fillna("not applicable").astype(str)
    rows = []
    for key, part in logs.groupby(group, observed=True, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = {**dict(zip(group, key)), "fit_records": len(part)}
        if runtime is not None:
            values = pd.to_numeric(part[runtime], errors="coerce")
            row.update(fit_seconds_recorded=int(values.notna().sum()),
                       total_fit_seconds=values.sum(min_count=1), median_fit_seconds=values.median())
        if "block_seconds" in part:
            values = pd.to_numeric(part.block_seconds, errors="coerce")
            row["total_block_seconds"] = values.sum(min_count=1)
        if parameters is not None:
            values = pd.to_numeric(part[parameters], errors="coerce")
            row.update(parameters_min=values.min(), parameters_max=values.max())
        if "city" in part:
            row["logged_cities"] = part.city.dropna().nunique()
        if "seed" in part:
            row["logged_seeds"] = part.seed.dropna().nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def _plots(daily: pd.DataFrame, city: pd.DataFrame, macro: pd.DataFrame, out: Path, protocol: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160, "figure.facecolor": "white"})
    primary = _primary_period(macro.period.unique().tolist(), protocol)
    horizons = sorted(macro.horizon.unique())
    budget = int(max(macro.budget))
    available = set(macro.method)
    selected = [m for m in protocol.get("plot_methods", DEFAULT_PLOT_METHODS) if m in available]
    if not selected:
        selected = sorted(available)
    palette = plt.get_cmap("tab20")
    colors = {m: palette(i % 20) for i, m in enumerate(selected)}
    charts = []

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5), squeeze=False, sharey=True)
    for ax, horizon in zip(axes.ravel(), horizons):
        part = macro[(macro.period == primary) & (macro.horizon == horizon)]
        for method in selected:
            values = part[part.method.eq(method)].sort_values("budget")
            ax.plot(values.budget, values.skill_mae, marker="o", label=method, color=colors[method], linewidth=1.4)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(title=f"{horizon}-day horizon", xlabel="Unique target fitting-label dates")
        ax.set_xticks(sorted(macro.budget.unique()))
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("MAE skill vs persistence (higher is better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(selected)), fontsize=8)
    fig.suptitle(f"{primary}: sparse-label learning curves; equal-city macro errors")
    fig.tight_layout(rect=(0, .18, 1, .94))
    filename = "learning_curves.png"
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)
    charts.append(filename)

    fig, ax = plt.subplots(figsize=(max(12, len(selected) * .75), 6))
    x = np.arange(len(selected))
    width = .8 / len(horizons)
    for i, horizon in enumerate(horizons):
        values = macro[(macro.period == primary) & (macro.budget == budget) & (macro.horizon == horizon)].set_index("method")
        ax.bar(x - .4 + width * (i + .5), values.reindex(selected).mae, width, label=f"{horizon} day(s)")
    ax.set_xticks(x, selected, rotation=55, ha="right", fontsize=8)
    ax.set(ylabel="Equal-city macro MAE (AQI units)", title=f"{primary}: horizon comparison at B={budget} fitting-label dates")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    filename = "horizon_errors.png"
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)
    charts.append(filename)

    cities = sorted(city.city.unique())
    heat_methods = sorted(set(city.method) - {protocol.get("persistence_method", "persistence"), "persist"})
    if heat_methods:
        fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), max(5, .36 * len(heat_methods))), squeeze=False, sharey=True)
        matrices = []
        for horizon in horizons:
            part = city[(city.period == primary) & (city.budget == budget) & (city.horizon == horizon)]
            matrices.append(part.pivot(index="method", columns="city", values="skill_mae").reindex(index=heat_methods, columns=cities))
        finite = np.concatenate([m.to_numpy().ravel() for m in matrices])
        finite = finite[np.isfinite(finite)]
        limit = max(.1, float(np.max(np.abs(finite)))) if len(finite) else 1.0
        for ax, horizon, matrix in zip(axes.ravel(), horizons, matrices):
            im = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
            ax.set_xticks(np.arange(len(cities)), cities, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(np.arange(len(heat_methods)), heat_methods, fontsize=8)
            ax.set_title(f"{horizon}-day horizon")
            for r in range(len(heat_methods)):
                for c in range(len(cities)):
                    value = matrix.iloc[r, c]
                    if np.isfinite(value):
                        ax.text(c, r, f"{value:+.0%}", ha="center", va="center", fontsize=7,
                                color="white" if abs(value) > .6 * limit else "black")
        fig.suptitle(f"{primary}: per-city MAE skill vs persistence at B={budget}")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="MAE skill (positive is better)", fraction=.025, pad=.03)
        fig.subplots_adjust(left=.14, bottom=.18, top=.91, right=.88, wspace=.15)
        filename = "per_city_skill.png"
        fig.savefig(out / filename, bbox_inches="tight")
        plt.close(fig)
        charts.append(filename)

    trace_horizon = min(horizons)
    trace_methods = [m for m in [protocol.get("persistence_method", "persistence"), "gru_pooled_zero", "gru_finetune", "reptile_gru", "fomaml_gru"] if m in available]
    if not trace_methods:
        trace_methods = selected[:4]
    part = daily[(daily.period == primary) & (daily.budget == budget) & (daily.horizon == trace_horizon)]
    start = pd.Timestamp(protocol.get("trace_start_date", part.date.min()))
    stop = start + pd.Timedelta(days=int(protocol.get("trace_calendar_days", 90)))
    part = part[part.date.ge(start) & part.date.lt(stop)]
    fig, axes = plt.subplots(math.ceil(len(cities) / 2), 2, figsize=(16, 3 * math.ceil(len(cities) / 2)), squeeze=False)
    for ax, city_name in zip(axes.ravel(), cities):
        g = part[part.city.eq(city_name)]
        truth = g[["date", "y_true"]].drop_duplicates().sort_values("date")
        ax.plot(truth.date, truth.y_true, color="black", linewidth=1.5, label="observed AQI")
        for method in trace_methods:
            values = g[g.method.eq(method)].sort_values("date")
            ax.plot(values.date, values.mean_prediction, linewidth=1, alpha=.8, label=method)
        ax.set(title=city_name, ylabel="AQI")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(alpha=.2)
    for ax in axes.ravel()[len(cities):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{primary}: fixed first {int(protocol.get('trace_calendar_days', 90))}-calendar-day window, h={trace_horizon}, B={budget}\nSeed-mean ensemble traces for visualization only; scores average individual-seed losses")
    fig.tight_layout(rect=(0, .09, 1, .95))
    filename = "forecast_traces.png"
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)
    charts.append(filename)
    return charts


def _write_report(out: Path, checks: dict, macro: pd.DataFrame, contrasts: pd.DataFrame,
                  subgroups: pd.DataFrame, protocol: dict, charts: list[str], costs: pd.DataFrame) -> str:
    primary = _primary_period(checks["periods"], protocol)
    main = macro[macro.period.eq(primary)]
    best = main.sort_values(["budget", "horizon", "mae", "method"]).groupby(["budget", "horizon"], as_index=False).head(1)
    persistence = checks["persistence_method"]
    # All budgets/horizons remain visible; the winner is descriptive, never a new tuning decision.
    rows = [
        "# Fixed-cohort AQI forecasting results", "",
        f"The primary held-out result period is **{primary}**, covering **{len(checks['cities'])} selected target cities**, "
        f"one station per city, horizons {checks['horizons']}, fitting-label budgets {checks['budgets']} and seeds {checks['seeds']}. "
        "All values below are calculated from the supplied, unclipped forecasts. This report does not select or train a model.", "",
        "## Evaluation integrity", "",
        f"- {checks['prediction_rows']:,} prediction rows; {len(checks['methods'])} methods. "
        "Exact prediction keys are paired across methods; truth, issue-day AQI and dates agree across methods, budgets and seeds.",
        f"- The method and period sets match the frozen protocol. Every forecast issue date is after the latest fitting cutoff, {checks['latest_fit_cutoff']}.",
        "- Target dates are fixed across label budgets and seeds" +
        (" and across horizons." if checks["common_target_dates_across_horizons"] else ". Horizons may have different eligible target dates."),
        "- Exact comparison with the supplied source truth/context table: " +
        ("passed." if checks["exact_source_reference_checked"] else "not supplied; internal prediction consistency was checked. Source provenance must be established by the data-building audit."),
        "- MAE, RMSE and signed bias use AQI units. Bias is prediction minus observation. City errors receive equal macro weight; a long station series does not become another city.",
        "- Losses are averaged across seeds at each city/target-date before inference. Seeds are repeated fits, not independent locations. RMSE is the square root of seed-and-date mean squared loss within a city, then averaged over cities; it is not an ensemble-prediction score.",
        f"- Paired 95% percentile intervals use {int(protocol.get('n_bootstrap', 1000)):,} moving-block resamples of "
        f"{int(protocol.get('bootstrap_block_days', 14))} contiguous calendar days within each city, followed by a fixed-city macro average. "
        "Missing calendar days retain their positions. This addresses temporal dependence and overlapping windows, conditional on these cities; cities are not resampled.", "",
        "## Primary held-out performance", "",
        "The lowest observed macro MAE in each budget/horizon cell is descriptive. Comparing many methods after seeing test results does not make the winner a separately validated deployment choice.", "",
        _markdown_table(best, ["budget", "horizon", "method", "mae", "rmse", "bias", "skill_mae", "cities_improved_vs_persistence", "n_cities"]), "",
    ]
    improvements = main[main.method.ne(persistence)]
    positive = int(improvements.skill_mae.gt(0).sum())
    rows += [
        f"Across the {len(improvements)} method/budget/horizon primary comparisons, {positive} have lower macro MAE than persistence. "
        "The cell-level comparison tables are the evidence; this count is not a sample of independent experiments.", "",
        "### Paired differences", "",
        "Differences are **method minus reference**, so negative MAE differences favor the method. "
        "An interval crossing zero leaves the direction unresolved under this temporal resampling procedure. "
        "Intervals are descriptive and unadjusted for the many comparisons.", "",
    ]
    focus = contrasts[(contrasts.period == primary) & contrasts.kind.isin(["finetune_minus_pooled_zero", "meta_minus_finetune"])]
    rows.append(_markdown_table(focus, ["budget", "horizon", "method", "reference", "delta_mae", "delta_mae_ci_low", "delta_mae_ci_high", "cities_improved", "n_cities"], max_rows=63))
    rows += ["", "## What the project questions can support", ""]
    scratch = contrasts[(contrasts.period == primary) & contrasts.kind.isin(["pooled_zero_minus_local", "finetune_minus_local"])]
    if len(scratch):
        rows.append("**Does pretraining beat training from scratch?** The direct paired comparisons below use the same architecture, "
                    "target label budget, input information and evaluation dates. Pooled zero-shot has donor fitting labels only; "
                    "fine-tuning adds the permitted target budget. Local scratch uses that target budget without pretraining. "
                    "These compare fitting strategies, not equal total numbers of source and target examples.")
        for kind, label in [("pooled_zero_minus_local", "Pooled zero-shot"), ("finetune_minus_local", "Fine-tuning")]:
            part = scratch[scratch.kind.eq(kind)]
            if len(part):
                rows += ["", f"{label} has lower observed macro MAE than its matched local scratch model in "
                         f"{int(part.delta_mae.lt(0).sum())}/{len(part)} architecture/budget/horizon cells; "
                         f"{int(part.delta_mae_ci_high.lt(0).sum())} have entirely negative temporal intervals. "
                         "A gain over scratch alone does not establish a useful gain over persistence."]
        scratch_summary = scratch.groupby(["kind", "method", "reference"], as_index=False, observed=True).agg(
            cells=("delta_mae", "size"), mean_delta_mae_across_cells=("delta_mae", "mean"),
            cells_with_lower_mae=("delta_mae", lambda v: int(v.lt(0).sum())),
            cells_with_entirely_negative_interval=("delta_mae_ci_high", lambda v: int(v.lt(0).sum())))
        rows += ["", _markdown_table(scratch_summary, ["kind", "method", "reference", "cells", "mean_delta_mae_across_cells",
                                                       "cells_with_lower_mae", "cells_with_entirely_negative_interval"]),
                 "", "The across-cell mean is a descriptive summary of related budget/horizon settings; "
                 "it is not a new independent-city estimate. Complete paired cell intervals are in `paired_comparisons.csv`."]
    ft = focus[focus.kind.eq("finetune_minus_pooled_zero")]
    if len(ft):
        n_better = int(ft.delta_mae.lt(0).sum())
        n_supported = int(ft.delta_mae_ci_high.lt(0).sum())
        rows.append(f"**Does fine-tuning help over the pooled model?** Fine-tuning has lower observed macro MAE in {n_better}/{len(ft)} "
                    f"architecture/budget/horizon cells; {n_supported} have an entirely negative temporal interval. "
                    "The individual cell and city results show where adaptation helps or harms; a common advantage is not assumed.")
    meta = focus[focus.kind.eq("meta_minus_finetune")]
    if len(meta):
        for method, part in meta.groupby("method", observed=True):
            rows += ["", f"**Does {method} add value over ordinary GRU fine-tuning?** It has lower observed macro MAE in "
                     f"{int(part.delta_mae.lt(0).sum())}/{len(part)} budget/horizon cells, with "
                     f"{int(part.delta_mae_ci_high.lt(0).sum())} entirely negative temporal intervals. "
                     "These first-order meta-learning implementations are tested directly; a positive meta-learning conclusion requires these comparisons, not the algorithm name."]
    rows += [
        "", "**Can transfer help when target fitting labels are scarce?** The budgets count unique target fitting-label dates, "
        "with each selected date used for the three horizon labels. Models may use the complete observed recent input context at forecast time. "
        "This simulates limited fitting labels, not a monitoring station with no recent observations. "
        "The most recent B fitting dates also change recency and seasonal composition, so budget differences are not a causal effect of label count alone.",
        "", "**Are these 24–72-hour forecasts?** Horizons 1–3 predict future daily AQI labels and are approximately 24–72 hours "
        "relative to the source's nominal 4 PM daily convention. Exact operational issue latency and real-time availability are not established. "
        "Inputs are past observed AQI; no pollutant or weather timing claim is implied.",
        "", "**Do models forecast an unseen place?** Pooled zero-shot omits target fitting labels, but still consumes the target station's recent observed AQI. "
        "Fine-tuning and local models use target labels. This is held-out-city transfer conditional on recent target inputs, not a forecast for a place with no monitoring history.",
        "", "**Can this be generalized nationally or causally?** The selected cities are a coverage-based study cohort, not a representative national sample. "
        "One station does not measure a city's full spatial exposure. Temporal intervals condition on these locations. "
        "Coverage selection, source missingness, overlapping windows, shared donor pretraining and budget recency limit causal and national claims.", "",
        "## Seasonal and high-AQI checks", "",
        "Winter is December–February; monsoon is June–September; other months form a third group. "
        "The high-AQI subgroup uses the **observed target AQI >= 200**, not a forecast-triggered alert. "
        "Subgroup macro errors average only cities with eligible observations; city/date counts accompany every row. "
        "Seasonal differences are descriptive and do not identify a transfer mechanism.", "",
    ]
    subgroup_view = subgroups[(subgroups.period == primary) & (subgroups.budget == max(checks["budgets"])) &
                             (subgroups.horizon == min(checks["horizons"])) &
                             subgroups.method.isin([persistence, "gru_pooled_zero", "gru_finetune", "reptile_gru", "fomaml_gru"])]
    rows.append(_markdown_table(subgroup_view, ["method", "subgroup_type", "subgroup", "n_cities", "n_city_dates", "mae", "rmse", "bias", "skill_mae"], max_rows=25))
    others = macro[macro.period.ne(primary)]
    if len(others):
        rows += ["", "## Secondary periods", "",
                 "The following periods are separate robustness evaluations. In particular, **secondary_2024 is not used for tuning, early stopping, architecture selection or budget selection**. "
                 "Do not pool it with the primary 2025 test when reporting the main result.", "",
                 _markdown_table(others.sort_values(["period", "budget", "horizon", "mae", "method"]).groupby(["period", "budget", "horizon"], as_index=False).head(1),
                                 ["period", "budget", "horizon", "method", "mae", "rmse", "skill_mae"])]
    rows += ["", "## Eligibility, implementation costs and reproducibility", "",
             "Eligibility and gap exclusions come from the preprocessing audit, not from model success. "
             "See `eligibility_counts.csv` / `eligibility_counts.json` when supplied. "
             "A missing record is not proof of monitoring downtime or naturally scarce labels. "
             "No missing AQI value is imputed by this evaluator; it scores only the common observed targets.", "",
             "Training durations, trainable parameter counts and fit status are recorded in `fit_costs` and `parameter_counts` artifacts when supplied. "
             "Repeated deterministic baseline seeds do not add independent evidence. "
             "Protocol/source hashes and execution environment belong in the run metadata alongside these results.", "",
             "The forecast traces average predictions across seeds for an ensemble visualization only; "
             "the reported errors instead average each seed's losses and do not score that ensemble.", ""]
    if not costs.empty:
        cost_columns = [c for c in ["stage", "method", "architecture", "fit_records", "total_fit_seconds",
                                    "median_fit_seconds", "parameters_min", "parameters_max", "total_block_seconds"] if c in costs]
        rows += ["### Recorded implementation costs", "",
                 "Durations sum the recorded fits within each stage/method/architecture. Shared donor fitting stays in its own stage "
                 "and is not multiplied by the number of target forecasts. Parameter counts describe a model, not a quantity to sum across fits. "
                 "`fit_seconds` records fitting routines, including checkpoint evaluation during donor selection; "
                 "target `block_seconds` additionally includes prediction/artifact work and is reported separately, never added to fitting time. "
                 "These are hardware- and run-specific measurements, not an algorithmic efficiency guarantee. "
                 "Missing timing or parameter entries are unknown rather than zero.", "",
                 _markdown_table(costs, cost_columns, max_rows=60), "",
                 "Complete stage summaries are in `cost_summary.csv`; individual recorded fits are in `fit_costs.csv`.", ""]
    rows += [
             "Complete outputs: `case_metrics.csv`, `seed_averaged_date_losses.csv.gz`, `city_metrics.csv`, "
             "`macro_metrics.csv`, `paired_comparisons.csv`, `subgroup_city_metrics.csv`, `subgroup_metrics.csv`, "
             "`evaluation_checks.json`, `evaluation_protocol.json`, and `summary.json`.", "",
             "## Figures", ""]
    for chart in charts:
        rows += [f"![{chart.removesuffix('.png').replace('_', ' ')}]({chart})", ""]
    text = "\n".join(rows).rstrip() + "\n"
    (out / "RESULTS.md").write_text(text)
    return text


def evaluate(predictions: pd.DataFrame, outdir: str | Path, protocol: dict | None = None) -> dict:
    """Validate, score, plot and report a complete paired prediction table.

    Optional protocol fields include targets/target_cities, budgets, horizons, seeds,
    expected_methods (canonical 23 by default), expected_periods (2024/2025),
    latest_fit_cutoff (2023-12-31; target_fitting_cutoff is also accepted),
    truth_reference (long DataFrame), n_bootstrap (default 1000),
    bootstrap_block_days (14), bootstrap_seed, primary_period, plot_methods,
    comparisons, eligibility_counts, fit_costs and parameter_counts.
    """
    protocol = dict(protocol or {})
    df, checks = _prepare(predictions, protocol)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    persistence = checks["persistence_method"]
    protocol["persistence_method"] = persistence
    case, daily, city, macro = _metrics(df, persistence)
    specs = _comparison_specs(checks["methods"], persistence, protocol)
    contrasts = _paired_comparisons(daily, city, specs, protocol)
    subgroup_city, subgroup_macro = _subgroups(daily, persistence)
    costs = _cost_summary(protocol)
    tables = {
        "case_metrics.csv": case, "seed_averaged_date_losses.csv.gz": daily,
        "city_metrics.csv": city, "macro_metrics.csv": macro,
        "paired_comparisons.csv": contrasts,
        "subgroup_city_metrics.csv": subgroup_city, "subgroup_metrics.csv": subgroup_macro,
    }
    if not costs.empty:
        tables["cost_summary.csv"] = costs
    for filename, table in tables.items():
        table.to_csv(out / filename, index=False)
    _write_json(out / "evaluation_checks.json", checks)
    _write_json(out / "evaluation_protocol.json", {k: v for k, v in protocol.items() if k != "truth_reference"})
    optional_artifacts = []
    for key in ["eligibility_counts", "fit_costs", "parameter_counts"]:
        if key not in protocol or protocol[key] is None:
            continue
        value = protocol[key]
        _write_json(out / f"{key}.json", value)
        optional_artifacts.append(f"{key}.json")
        if isinstance(value, pd.DataFrame):
            frame = value
        elif isinstance(value, list) and all(isinstance(r, dict) for r in value):
            frame = pd.DataFrame(value)
        elif isinstance(value, dict) and all(np.isscalar(v) or v is None for v in value.values()):
            frame = pd.DataFrame([{"item": k, "value": v} for k, v in value.items()])
        else:
            frame = None
        if frame is not None:
            frame.to_csv(out / f"{key}.csv", index=False)
            optional_artifacts.append(f"{key}.csv")
    charts = _plots(daily, city, macro, out, protocol)
    _write_report(out, checks, macro, contrasts, subgroup_macro, protocol, charts, costs)
    primary = _primary_period(checks["periods"], protocol)
    primary_rows = macro[macro.period.eq(primary)]
    best = primary_rows.sort_values(["budget", "horizon", "mae", "method"]).groupby(["budget", "horizon"], as_index=False).head(1)
    summary = {
        "outdir": str(out.resolve()), "primary_period": primary, "checks": checks,
        "uncertainty": {"method": "paired moving-block calendar bootstrap within each fixed city",
                        "resamples": int(protocol.get("n_bootstrap", 1000)),
                        "block_calendar_days": int(protocol.get("bootstrap_block_days", 14)),
                        "seed": int(protocol.get("bootstrap_seed", 20261114)),
                        "scope": "Conditional temporal uncertainty for the selected cities; not national population uncertainty"},
        "best_observed_primary_cells": best.to_dict(orient="records"),
        "artifacts": list(tables) + charts + optional_artifacts +
                     ["RESULTS.md", "evaluation_checks.json", "evaluation_protocol.json", "summary.json"],
    }
    summary = _json_safe(summary)
    _write_json(out / "summary.json", summary)
    return summary


def display_results(result: dict | str | Path, max_table_rows: int = 30) -> None:
    """Display compact tables, the generated interpretation and figures in a notebook.

    Pass either the dict returned by evaluate or its output directory.  This
    helper does not require seaborn, tabulate or a notebook-specific runtime.
    """
    out = Path(result["outdir"] if isinstance(result, dict) else result)
    summary = json.loads((out / "summary.json").read_text())
    table = pd.read_csv(out / "macro_metrics.csv")
    table = table[table.period.eq(summary["primary_period"])].sort_values(["budget", "horizon", "mae"])
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        print(_markdown_table(table, ["budget", "horizon", "method", "mae", "rmse", "skill_mae"], max_table_rows))
        print((out / "RESULTS.md").read_text())
        return
    display(Markdown(f"### Primary results: {summary['primary_period']}"))
    display(table.head(max_table_rows))
    # The report contains image links for portable Markdown; inline notebook
    # images are emitted separately so they work regardless of notebook cwd.
    report = (out / "RESULTS.md").read_text().split("## Figures")[0]
    display(Markdown(report))
    for filename in summary["artifacts"]:
        if filename.endswith(".png"):
            display(Image(filename=str(out / filename)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, help="JSON protocol metadata; no training is performed")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text()) if args.protocol else {}
    predictions = pd.read_csv(args.predictions)
    summary = evaluate(predictions, args.outdir, protocol)
    print(json.dumps({"outdir": summary["outdir"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
