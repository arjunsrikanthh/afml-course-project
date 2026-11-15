#!/usr/bin/env python3
"""Independently audit cross-architecture trajectories from saved forecast rows."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd


FROZEN = Path("delivery/submission_20261115/repository/data/frozen")
ORIGINAL = Path("results/final_study_20261114/mac_research/afml_research/models")
OUTPUT = Path("results/cross_architecture_followup_20261115")
FAMILIES = ("deep_lstm", "deep_mlp", "deep_tcn", "deep_transformer")
BUDGETS = (30, 90, 180)
SEEDS = (17, 43, 71)
VARIANTS = ("original", "smaller_lr")
STEPS = (0, 1, 3, 10, 30, 100)
KEYS = ["city", "station_id", "date", "issue_date", "horizon", "period"]
TRUTH = ["last", "y_true"]
IDENTITY = ["city", "budget", "seed", "family", "method"]
STATE = ["step", "period", "horizon"]
FUTURE = ("secondary_2024", "test_2025")
ARCHIVE_TOLERANCE_AQI = .002
STEP_ZERO_TOLERANCE_AQI = .00002


def check(condition, reason):
    if not condition:
        raise AssertionError(reason)


def digest(path):
    with path.open("rb") as stream:
        result = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    return pd.read_csv(path, dtype={"station_id": str})


def frozen_windows(frozen):
    """Reconstruct analysis windows directly from the released pre-model inputs."""
    windows = read_csv(frozen / "forecast_examples.csv.gz").rename(columns={"y": "y_true"})
    cities = read_json(frozen / "protocol.json")["targets"]
    cohort = read_csv(frozen / "cohort.csv")
    donors = cohort.loc[cohort.role.isin(["donor", "donor_development"]), "city"]
    candidates = windows[
        windows.city.isin(donors)
        & windows.date.between("2023-01-04", "2023-12-31")
        & (windows.issue_date >= "2023-01-01")
    ]
    panel = []
    for city, group in candidates.groupby("city"):
        dates = sorted(group.date.unique(), key=lambda date: sha256(f"{city}|{date}".encode()).hexdigest())[:6]
        panel.append(group.loc[group.date.isin(dates)])
    retain = pd.concat(panel, ignore_index=True).assign(period="seen_donor_retention_2023")
    check(len(retain) == 23 * 6 * 3, "released retention panel size")

    queries, supports = {}, {}
    for city in cities:
        rows = windows.loc[windows.city == city]
        second = rows[
            rows.date.between("2024-01-04", "2024-12-31") & (rows.issue_date >= "2024-01-01")
        ].assign(period=FUTURE[0])
        main = rows[
            rows.date.between("2025-01-01", "2025-12-31") & (rows.issue_date >= "2024-01-01")
        ].assign(period=FUTURE[1])
        queries[city] = pd.concat([second, main], ignore_index=True)

        eligible = rows.loc[rows.date <= "2023-12-31"]
        for budget in BUDGETS:
            dates = sorted(eligible.date.unique())[-budget:]
            fitting = eligible.loc[eligible.date.isin(dates)].assign(period="support")
            check(len(fitting) == 3 * budget, (city, budget, "released support row count"))
            supports[city, budget] = fitting
    return cities, queries, supports, retain


def aligned(actual, expected, context):
    ordered = actual.sort_values(KEYS, ignore_index=True)
    check(not ordered.duplicated(KEYS).any(), (context, "duplicate forecast key"))
    check(ordered[KEYS].equals(expected[KEYS]), (context, "released forecast keys"))
    check(np.array_equal(ordered[TRUTH], expected[TRUTH]), (context, "released truth or persistence"))
    check(np.isfinite(ordered.y_pred).all(), (context, "non-finite forecasts"))
    return ordered


def summaries(rows):
    initial_error = rows.y_initial.to_numpy() - rows.y_true.to_numpy()
    correction = rows.y_pred.to_numpy() - rows.y_initial.to_numpy()
    error = rows.y_pred.to_numpy() - rows.y_true.to_numpy()
    work = rows[STATE].assign(
        absolute=np.abs(error),
        squared=error ** 2,
        absolute_initial=np.abs(initial_error),
        squared_initial=initial_error ** 2,
        correction=correction,
        correction_squared=correction ** 2,
        correction_alignment=2 * initial_error * correction,
    )
    result = work.groupby(STATE, as_index=False).agg(
        n=("absolute", "size"),
        mae=("absolute", "mean"),
        mse=("squared", "mean"),
        initial_mae=("absolute_initial", "mean"),
        initial_mse=("squared_initial", "mean"),
        mean_correction=("correction", "mean"),
        mean_correction_squared=("correction_squared", "mean"),
        correction_alignment=("correction_alignment", "mean"),
    )
    result["rmse"] = np.sqrt(result.mse)
    result["residual_mse"] = result.mse / 10_000
    result["rms_correction"] = np.sqrt(result.mean_correction_squared)
    check(np.allclose(result.mse - result.initial_mse,
                      result.mean_correction_squared + result.correction_alignment,
                      rtol=1e-9, atol=1e-8), "squared error correction identity")
    return result


def compare_metrics(actual, expected, keys, context):
    actual = actual.sort_values(keys, ignore_index=True)
    expected = expected.sort_values(keys, ignore_index=True)
    check(actual[keys].equals(expected[keys]), (context, "metric dimensions"))
    columns = ["n", "mae", "rmse", "initial_mae", "residual_mse", "mean_correction", "rms_correction"]
    check(np.allclose(actual[columns], expected[columns], rtol=1e-10, atol=1e-10),
          (context, "metrics differ from raw forecasts"))


def verify_source_digests(root, frozen, protocol):
    script_dir = root / "scripts/investigation_20261115"
    for field, filename in (("wrapper_sha256", "cross_architecture_controls.py"),
                            ("investigation_script_sha256", "adaptation_controls.py")):
        check(digest(script_dir / filename) == protocol[field], (filename, "frozen script digest"))
    for source, expected in protocol["source_states_sha256"].items():
        check(digest(root / source) == expected, (source, "source checkpoint digest"))
    for name, expected in protocol["frozen_engine_sha256"].items():
        check(digest(frozen.parent.parent / "scripts" / name) == expected, (name, "released engine digest"))
    check(digest(frozen / "INPUT_MANIFEST.json") == protocol["input_manifest_sha256"], "input manifest digest")
    input_hashes = read_json(frozen / "INPUT_MANIFEST.json")
    for name, metadata in input_hashes.items():
        check(digest(frozen / name) == metadata["sha256"], (name, "released input digest"))


def full_audit(root, output, allow_partial):
    frozen = root / FROZEN
    original = root / ORIGINAL
    protocol_path = output / "protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = digest(protocol_path)
    verify_source_digests(root, frozen, protocol)
    check(set(protocol["families"]) == set(FAMILIES), "frozen family coverage")
    check(tuple(protocol["budgets"]) == BUDGETS and tuple(protocol["seeds"]) == SEEDS,
          "frozen budget and seed coverage")
    check(protocol["variants"] == {
        "original": {"lr_factor": 1, "head_only": False, "anchor": 0},
        "smaller_lr": {"lr_factor": .1, "head_only": False, "anchor": 0},
    }, "frozen update controls")
    cities, queries, supports, panel = frozen_windows(frozen)
    expected_identities = set(product(cities, BUDGETS, SEEDS, FAMILIES, VARIANTS))
    check(protocol["expected_blocks"] == len(expected_identities), "frozen expected block count")
    receipts = sorted((output / "receipts").glob("*.json"))
    check(len(receipts) > 0, "no completed prediction receipts")
    check(allow_partial or len(receipts) == len(expected_identities), "incomplete prediction receipts")
    seen = set()
    baselines = {}
    blocks = []
    daily_endpoints = []
    max_parity = max_step_zero = 0.0
    forecast_rows = archived_replays = 0

    for path in receipts:
        receipt = read_json(path)
        forecasts = output / "prediction_blocks" / path.with_suffix(".csv.gz").name
        check(receipt["complete"] and receipt["protocol_sha256"] == protocol_hash
              and receipt["prediction_sha256"] == digest(forecasts), (path.name, "receipt and forecast digests"))
        rows = read_csv(forecasts)
        target = rows.loc[rows.period.isin(FUTURE)]
        identity = tuple(target[key].iat[0] for key in IDENTITY)
        city, budget, seed, family, variant = identity
        check(identity in expected_identities and identity not in seen, (path.name, "unexpected block identity"))
        seen.add(identity)
        for key, value in zip(IDENTITY[1:], identity[1:]):
            check(rows[key].eq(value).all(), (path.name, key, "mixed block identity"))
        support = supports[city, budget]
        expected = pd.concat([queries[city], support, panel], ignore_index=True).sort_values(KEYS, ignore_index=True)
        states = rows.groupby("step")
        check(set(states.indices) == set(STEPS), (path.name, "trajectory checkpoints"))
        step_zero = aligned(states.get_group(0), expected, (path.name, 0)).y_pred.to_numpy()
        paired = baselines.setdefault(identity[:-1], step_zero)
        cross_variant_delta = float(np.max(np.abs(step_zero - paired)))
        check(cross_variant_delta <= STEP_ZERO_TOLERANCE_AQI, (path.name, "cross-variant start differs"))
        max_step_zero = max(max_step_zero, cross_variant_delta)
        for step, state in states:
            ordered = aligned(state, expected, (path.name, step))
            check(np.array_equal(ordered.y_initial, step_zero), (path.name, step, "within-trajectory baseline"))
            if variant == "original" and step in (0, 100):
                original_method = family + ("_pooled_zero" if step == 0 else "_finetune")
                prefix = path.stem.removesuffix(f"_{family}_{variant}")
                archive = read_csv(original / "prediction_blocks" / f"{prefix}_{original_method}.csv.gz")
                archive = archive.sort_values(KEYS, ignore_index=True)
                result = aligned(state.loc[state.period.isin(FUTURE)], archive, (path.name, step, "archived replay"))
                delta = float(np.max(np.abs(result.y_pred - archive.y_pred)))
                check(delta <= ARCHIVE_TOLERANCE_AQI, (path.name, step, "archived replay differs", delta))
                max_parity = max(max_parity, delta)
                archived_replays += 1
        raw = summaries(rows)
        compare_metrics(raw, pd.DataFrame(receipt["summaries"]), STATE, path.name)
        blocks.append(raw.assign(city=city, budget=budget, seed=seed, family=family, variant=variant))
        endpoint = target.loc[target.step == 100]
        daily_endpoints.append(endpoint[["city", "date", "period", "horizon", "family"]].assign(
            variant=variant, absolute=(endpoint.y_pred - endpoint.y_true).abs(),
            absolute_initial=(endpoint.y_initial - endpoint.y_true).abs(),
        ))
        check(len(rows) == receipt["rows"], (path.name, "prediction row count"))
        forecast_rows += len(rows)

    metrics = pd.concat(blocks, ignore_index=True)
    complete = seen == expected_identities
    if complete:
        dimensions = ["city", "budget", "seed", "family", "variant", *STATE]
        compare_metrics(metrics, read_csv(output / "metrics.csv"), dimensions, "released metric table")
        verify_marginal_metrics(metrics, output / "macro_descriptive.csv")
        check(archived_replays == 2 * len(expected_identities) // len(VARIANTS), "incomplete archived replays")
        endpoint_table(metrics).to_csv(output / "independent_endpoints.csv", index=False)
        intervals(pd.concat(daily_endpoints, ignore_index=True)).to_csv(output / "endpoint_block_intervals.csv", index=False)

    return {
        "passed": True,
        "complete": complete,
        "blocks": len(seen),
        "expected_blocks": len(expected_identities),
        "forecast_rows_verified": forecast_rows,
        "independent_metric_rows": len(metrics),
        "archived_replay_blocks_verified": archived_replays,
        "max_archived_prediction_difference_aqi": max_parity,
        "max_cross_variant_step_zero_difference_aqi": max_step_zero,
        "cross_variant_step_zero_tolerance_aqi": STEP_ZERO_TOLERANCE_AQI,
        "protocol_sha256": protocol_hash,
    }


def endpoint_table(metrics):
    endpoint = metrics.loc[metrics.step == 100].copy()
    endpoint["mae_change"] = endpoint.mae - endpoint.initial_mae
    endpoint["mse_change"] = endpoint.mse - endpoint.initial_mse
    endpoint["rmse_change"] = endpoint.rmse - np.sqrt(endpoint.initial_mse)
    paired_keys = ["family", "variant", "city", "budget", "seed", "horizon"]
    support = endpoint.loc[endpoint.period == "support", paired_keys + ["mae_change", "mse_change"]]
    compared = endpoint.merge(
        support.rename(columns={"mae_change": "support_mae_change", "mse_change": "support_mse_change"}),
        on=paired_keys, validate="many_to_one",
    )
    compared["support_mse_down_this_mae_up"] = (compared.support_mse_change < 0) & (compared.mae_change > 0)
    table = compared.groupby(["family", "variant", "period"], as_index=False).agg(
        cases=("mae", "size"),
        mae=("mae", "mean"),
        initial_mae=("initial_mae", "mean"),
        mae_change=("mae_change", "mean"),
        rmse_change=("rmse_change", "mean"),
        mse_change=("mse_change", "mean"),
        support_mae_change=("support_mae_change", "mean"),
        support_mse_change=("support_mse_change", "mean"),
        correction_alignment_aqi_squared=("correction_alignment", "mean"),
        correction_size_aqi_squared=("mean_correction_squared", "mean"),
        support_mse_down_this_mae_up_cases=("support_mse_down_this_mae_up", "sum"),
    )
    original = table.loc[table.variant == "original", ["family", "period", "mae", "rmse_change"]]
    table = table.merge(original.rename(columns={"mae": "original_mae", "rmse_change": "original_rmse_change"}),
                        on=["family", "period"], validate="many_to_one")
    table["mae_minus_original"] = table.mae - table.original_mae
    table["rmse_minus_original"] = table.rmse_change - table.original_rmse_change
    return table


def verify_marginal_metrics(metrics, reported):
    keys = ["family", "variant", "step", "period"]
    columns = ["mae", "rmse", "initial_mae", "residual_mse"]
    macro = metrics.groupby(keys, as_index=False)[columns].mean()
    macro["mae_delta"] = macro.mae - macro.initial_mae
    actual = read_csv(reported).sort_values(keys, ignore_index=True)
    check(macro[keys].equals(actual[keys]), "macro metric dimensions")
    check(np.allclose(macro[columns + ["mae_delta"]], actual[columns + ["mae_delta"]],
                      rtol=1e-10, atol=1e-10), "macro metrics differ from raw forecast errors")


def intervals(daily):
    """Conditional 95% resampling intervals, sharing calendar draws across cases."""
    keys = ["family", "period", "city", "horizon", "date"]
    values = daily.groupby(keys + ["variant"])[["absolute", "absolute_initial"]].mean().unstack("variant")
    changes = pd.DataFrame({
        "original_minus_pooled": values["absolute", "original"] - values["absolute_initial", "original"],
        "smaller_lr_minus_pooled": values["absolute", "smaller_lr"] - values["absolute_initial", "smaller_lr"],
        "smaller_lr_minus_original": values["absolute", "smaller_lr"] - values["absolute", "original"],
    }).reset_index()
    records = []
    for (family, period), group in changes.groupby(["family", "period"]):
        year = int(period[-4:])
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        days = (pd.to_datetime(group.date) - start).dt.days
        year_days = (end - start).days
        check(days.between(0, year_days - 1).all(), (family, period, "calendar-block date"))
        value_cols = [column for column in group.columns if column not in keys]
        for contrast in value_cols:
            results = {"family": family, "period": period, "contrast": contrast}
            cell_means = group.groupby(["city", "horizon"])[contrast].mean()
            city_means = cell_means.groupby("city").mean()
            check(len(cell_means) == 18 and len(city_means) == 6, (family, period, "interval case counts"))
            results["macro_mae_difference"] = cell_means.mean()
            results["cities_above_zero"] = int((city_means > 0).sum())
            for width in (14, 28):
                blocks = group.assign(block=(days // width)).groupby(["block", "city", "horizon"])[contrast]
                labels = range((year_days + width - 1) // width)
                sums = blocks.sum().unstack(["city", "horizon"], fill_value=0).reindex(labels, fill_value=0)
                counts = blocks.count().unstack(["city", "horizon"], fill_value=0).reindex(labels, fill_value=0)
                rng = np.random.default_rng(107)
                weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=4000)
                denominators = weights @ counts.to_numpy()
                include = (denominators > 0).all(axis=1)
                check(include.sum() > 3900, (family, period, width, "too many empty bootstrap draws"))
                distribution = ((weights @ sums.to_numpy())[include] / denominators[include]).mean(axis=1)
                low, high = np.quantile(distribution, [.025, .975])
                results[f"low_{width}d"], results[f"high_{width}d"] = low, high
            records.append(results)
    return pd.DataFrame.from_records(records)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Audit only completed receipts.")
    args = parser.parse_args()
    output = args.output or args.root / OUTPUT
    suffix = "_partial" if args.allow_partial else ""
    destination = output / f"independent_verification{suffix}.json"
    try:
        report = full_audit(args.root, output, args.allow_partial)
    except (AssertionError, OSError, ValueError, KeyError, IndexError) as exc:
        destination.write_text(json.dumps({"passed": False, "error": str(exc)}, indent=2) + "\n")
        raise
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
