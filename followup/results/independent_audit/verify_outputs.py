#!/usr/bin/env python3
"""Verify follow-up blocks against released inputs and recompute comparison tables.

This audit uses only released raw forecast examples, the frozen analysis protocol,
and emitted predictions/receipts. It does not import either production evaluator.
"""
from argparse import ArgumentParser
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE.parent
FROZEN = ROOT / "delivery/submission_20261115/repository/data/frozen"
ORIGINAL = ROOT / "results/final_study_20261114/mac_research/afml_research/models"
SEEDS = (17, 43, 71)
BUDGETS = (30, 90, 180)
META = ("pooled", "maml", "fomaml", "reptile")
FAMILIES = ("deep_gru", "matched_gru")
VARIANTS = ("original", "smaller_lr", "head_only", "anchor_0p01", "anchor_0p1")
FORECAST_KEY = ["city", "station_id", "date", "issue_date", "horizon", "period"]
TRUTH = ["last", "y_true"]
TARGET_PERIODS = ("secondary_2024", "test_2025")
META_STATES = {(0.0, 0), *product((0.01, 0.03), (1, 3, 10, 30))}
STEP_ZERO_ATOL_AQI = 2e-5


def read_csv(path):
    return pd.read_csv(path, dtype={"station_id": str})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def check(condition, context):
    if not condition:
        raise AssertionError(context)


def align(actual, expected, key, truth, context):
    """Compare ordered keys and truth values without any outcome-based filtering."""
    actual = actual.sort_values(key, ignore_index=True)
    expected = expected.sort_values(key, ignore_index=True)
    check(not actual.duplicated(key).any(), (context, "duplicate observed key"))
    check(actual[key].equals(expected[key]), (context, "released window keys differ"))
    check(np.array_equal(actual[truth], expected[truth]), (context, "truth differs"))
    check(np.isfinite(actual.y_pred).all(), (context, "nonfinite predictions"))
    return actual


def single(frame, column):
    possibilities = frame[column].unique()
    check(len(possibilities) == 1, ("mixed block identity", column, possibilities.tolist()))
    return possibilities[0]


def released_views(donor_dates_per_city):
    windows = read_csv(FROZEN / "forecast_examples.csv.gz").rename(columns={"y": "y_true"})
    cities = read_json(FROZEN / "protocol.json")["targets"]
    cohorts = read_csv(FROZEN / "cohort.csv")
    donors = cohorts.loc[cohorts.role.isin(["donor", "donor_development"]), "city"]
    donor_candidates = windows[
        windows.city.isin(donors)
        & windows.date.between("2023-01-04", "2023-12-31")
        & (windows.issue_date >= "2023-01-01")
    ]
    panels = []
    for city, group in donor_candidates.groupby("city"):
        dates = sorted(group.date.unique(), key=lambda date: hashlib.sha256(
            f"{city}|{date}".encode()).hexdigest())[:donor_dates_per_city]
        panels.append(group[group.date.isin(dates)])
    panel = pd.concat(panels, ignore_index=True).assign(period="seen_donor_retention_2023")
    check(len(panel) == 23 * donor_dates_per_city * 3, "incomplete donor retention panel")
    queries, supports = {}, {}
    for city in cities:
        rows = windows[windows.city == city]
        secondary = rows[
            rows.date.between("2024-01-04", "2024-12-31") & (rows.issue_date >= "2024-01-01")
        ].assign(period="secondary_2024")
        test = rows[
            rows.date.between("2025-01-01", "2025-12-31") & (rows.issue_date >= "2024-01-01")
        ].assign(period="test_2025")
        queries[city] = pd.concat([secondary, test], ignore_index=True)
        eligible = rows[rows.date <= "2023-12-31"]
        for budget in BUDGETS:
            dates = sorted(eligible.date.unique())[-budget:]
            support = eligible[eligible.date.isin(dates)].assign(period="support")
            check(len(support) == 3 * budget, (city, budget, "bad support count"))
            supports[city, budget] = support
    return cities, queries, supports, panel


def original_parity(rows, path, tolerance):
    original = read_csv(path)
    got = align(rows, original, FORECAST_KEY, TRUTH, path.name)
    original = original.sort_values(FORECAST_KEY, ignore_index=True)
    delta = np.abs(got.y_pred - original.y_pred)
    check(delta.max() <= tolerance, (path.name, "original parity", float(delta.max())))
    return float(delta.max())


def compare_metrics(actual, expected, index, columns, context):
    actual = actual.sort_values(index, ignore_index=True)
    expected = expected.sort_values(index, ignore_index=True)
    check(len(actual) == len(expected), (context, "different metric row count"))
    check((actual[index] == expected[index]).all().all(), (context, "metric keys differ"))
    check(np.allclose(actual[columns], expected[columns], rtol=1e-10, atol=1e-10),
          (context, "recomputed metric differs"))


def error_metrics(rows, group_keys):
    error = rows.y_pred.to_numpy() - rows.y_true.to_numpy()
    return rows.assign(mae=np.abs(error), rmse=error ** 2).groupby(
        group_keys, as_index=False
    ).agg(n=("y_true", "size"), mae=("mae", "mean"), rmse=("rmse", "mean")).assign(
        rmse=lambda metrics: np.sqrt(metrics.rmse)
    )


def audit_meta():
    output = OUTPUT / "meta_initialization"
    protocol_file = output / "PROTOCOL.json"
    protocol_hash = sha(protocol_file)
    protocol = read_json(protocol_file)
    for filename, digest in protocol["source_sha256"].items():
        check(sha(ROOT / filename) == digest, (filename, "meta protocol source changed"))
    cities, queries, supports, panel = released_views(7)
    receipts = sorted((output / "blocks").glob("*.json"))
    expected_identities = set(product(cities, BUDGETS, SEEDS, META))
    seen = set()
    rows_checked, donor_rows_checked, original_replays = 0, 0, 0
    max_parity = 0.0
    raw_metrics = []
    metric_dimensions = ["lr", "step", "period", "horizon"]
    identity_keys = ["city", "budget", "seed", "initialization"]
    donor_key = ["city", "date", "horizon"]
    for path in receipts:
        rec = read_json(path)
        pred_path = path.with_suffix(".csv.gz")
        donor_path = path.with_name(f"{path.stem}_donor.csv.gz")
        check(rec["protocol_sha256"] == protocol_hash, (path.name, "protocol digest"))
        check(rec["prediction_sha256"] == sha(pred_path), (path.name, "target digest"))
        check(rec["donor_prediction_sha256"] == sha(donor_path), (path.name, "donor digest"))
        pred, donor = read_csv(pred_path), read_csv(donor_path)
        identity = tuple(single(pred, key) for key in identity_keys)
        city, budget, seed, init = identity
        check(identity not in seen and identity in expected_identities, (path.name, identity))
        seen.add(identity)
        for key, value in zip(["target_city", *identity_keys[1:]], identity):
            check(single(donor, key) == value, (path.name, "donor identity", key))
        state_keys = ["lr", "step"]
        pred_states = pred.groupby(state_keys)
        donor_states = donor.groupby(state_keys)
        check(set(pred_states.indices) == set(donor_states.indices) == META_STATES,
              (path.name, "state keys"))
        trajectories = {(r["lr"], r["step"]): r for r in rec["support_and_drift"]}
        check(len(rec["support_and_drift"]) == 9 and set(trajectories) == META_STATES,
              (path.name, "support state count"))
        support = supports[city, budget]
        dates = sorted(support.date.unique())
        dates_digest = hashlib.sha256("\n".join(dates).encode()).hexdigest()
        applicable = {(0.0, 0), (0.01, 3)} if init == "pooled" else {(0.03, 10)}
        actual_replays = {(replay["lr"], replay["step"]): replay for replay in rec["original_replays"]}
        check(set(actual_replays) == applicable, (path.name, "unexpected original replay states"))
        for state, rows in pred_states:
            align(rows, queries[city], FORECAST_KEY, TRUTH, path.name)
            donor_rows = donor_states.get_group(state)
            align(donor_rows, panel, donor_key, ["y_true"], path.name)
            trace = trajectories[state]
            check(trace["support_n"] == 3 * budget and trace["support_first_label"] == dates[0]
                  and trace["support_last_label"] == dates[-1]
                  and trace["support_dates_sha256"] == dates_digest
                  and trace["support_first_feature_date"] == support.feature_date_min.min()
                  and trace["query_rows"] == len(rows)
                  and trace["query_first_issue_date"] == queries[city].issue_date.min()
                  and trace["query_first_feature_date"] == queries[city].feature_date_min.min()
                  and np.isfinite(trace["support_mse"]),
                  (path.name, state, "support/query trace"))
            error = donor_rows.y_pred - donor_rows.y_true
            check(np.isclose(np.abs(error).mean(), trace["donor_retention_mae"], atol=1e-10)
                  and np.isclose(np.mean((error / 100) ** 2),
                                 trace["donor_retention_residual_mse"], atol=1e-10),
                  (path.name, "donor diagnostics"))
            if state in applicable:
                method = (("matched_gru_pooled_zero" if state[1] == 0 else "matched_gru_finetune")
                          if init == "pooled" else f"{init}_gru_sgd")
                original = ORIGINAL / "prediction_blocks" / f"{path.stem.rsplit('_', 1)[0]}_{method}.csv.gz"
                replay = actual_replays[state]
                check(sha(original) == replay["original_prediction_sha256"], (path.name, "original digest"))
                difference = original_parity(rows, original, 0.01)
                check(np.isclose(difference, replay["prediction_max_abs_diff"], rtol=0, atol=1e-12),
                      (path.name, "replay delta"))
                max_parity = max(max_parity, difference)
                original_replays += 1
        metrics = error_metrics(pred, metric_dimensions)
        compare_metrics(metrics, pd.DataFrame(rec["metrics"]), metric_dimensions,
                        ["n", "mae", "rmse"], path.name)
        raw_metrics.append(metrics.assign(city=city, budget=budget, seed=seed, initialization=init))
        rows_checked += len(pred)
        donor_rows_checked += len(donor)
    check(seen == expected_identities, "missing meta block identities")
    metrics = pd.concat(raw_metrics, ignore_index=True)
    dims = identity_keys + metric_dimensions
    compare_metrics(metrics, read_csv(output / "city_seed_metrics.csv"), dims,
                    ["n", "mae", "rmse"], "meta city-seed summaries")
    paired_keys = [k for k in dims if k != "initialization"]
    pooled = metrics[metrics.initialization == "pooled"]
    paired = metrics[metrics.initialization != "pooled"].merge(
        pooled[paired_keys + ["mae", "rmse"]], on=paired_keys,
        suffixes=("", "_pool"), validate="many_to_one")
    paired["mae_minus_pooled"] = paired.mae - paired.mae_pool
    paired["rmse_minus_pooled"] = paired.rmse - paired.rmse_pool
    compare_metrics(paired, read_csv(output / "paired_meta_minus_pooled.csv"), dims,
                    ["mae_minus_pooled"], "meta matched-schedule contrast")
    keys = [k for k in dims if k not in ["lr", "step"]]
    zero = metrics[metrics.step == 0]
    adaptation = metrics[metrics.step > 0].merge(zero[keys + ["mae", "rmse"]], on=keys,
                                                suffixes=("", "_zero"), validate="many_to_one")
    adaptation["mae_change_from_zero"] = adaptation.mae - adaptation.mae_zero
    adaptation["rmse_change_from_zero"] = adaptation.rmse - adaptation.rmse_zero
    compare_metrics(adaptation, read_csv(output / "paired_adaptation_effects.csv"), dims,
                    ["mae_change_from_zero", "rmse_change_from_zero"], "meta adaptation contrast")
    return {"blocks": len(seen), "target_prediction_rows": rows_checked,
            "donor_prediction_rows": donor_rows_checked, "independent_metric_rows": len(metrics),
            "independent_reproductions": original_replays, "max_original_difference_aqi": max_parity,
            "raw_recomputed_every_block": True, "contrast_rows_verified": len(paired) + len(adaptation)}


def audit_adaptation():
    output = OUTPUT / "adaptation"
    protocol_path = output / "protocol.json"
    protocol_hash = sha(protocol_path)
    protocol = read_json(protocol_path)
    sources = protocol["source_states_sha256"]
    for filename, digest in sources.items():
        check(sha(ROOT / filename) == digest, (filename, "adaptation protocol source changed"))
    check(sha(ROOT / "scripts/investigation_20261115/adaptation_controls.py")
          == protocol["investigation_script_sha256"], "adaptation script digest")
    cities, queries, supports, panel = released_views(6)
    settings = read_json(ORIGINAL / "frozen_adaptation_settings.json")
    receipts = sorted((output / "receipts").glob("*.json"))
    expected_identities = set(product(cities, BUDGETS, SEEDS, FAMILIES, VARIANTS))
    complete = len(receipts) == len(expected_identities)
    seen = set()
    initial = {}
    rows_checked, original_replays = 0, 0
    max_parity = 0.0
    max_step_zero_difference = 0.0
    raw_metrics = []
    group_keys = ["step", "period", "horizon"]
    identity_keys = ["city", "budget", "seed", "family", "method"]
    appearances = Counter()
    for path in receipts:
        rec = read_json(path)
        prediction_path = output / "prediction_blocks" / path.with_suffix(".csv.gz").name
        check(rec["complete"] and rec["protocol_sha256"] == protocol_hash
              and rec["prediction_sha256"] == sha(prediction_path), (path.name, "receipt digests"))
        pred = read_csv(prediction_path)
        target = pred[pred.period.isin(TARGET_PERIODS)]
        identity = tuple(single(target, key) for key in identity_keys)
        city, budget, seed, family, variant = identity
        for key, value in zip(identity_keys[1:], identity[1:]):
            check(single(pred, key) == value, (path.name, "mixed state", key))
        check(identity not in seen and identity in expected_identities, (path.name, identity))
        seen.add(identity)
        appearances[city] += 1
        _, ft, pooled, steps = protocol["families"][family]
        states = pred.groupby("step")
        check(set(states.indices) == set(steps), (path.name, "step keys"))
        expected = pd.concat([queries[city], supports[city, budget], panel], ignore_index=True)
        original = states.get_group(0).sort_values(FORECAST_KEY, ignore_index=True).y_pred.to_numpy()
        if identity[:-1] in initial:
            compare_to = initial[identity[:-1]]
            check(compare_to.shape == original.shape, (path.name, "step zero shape differs"))
            difference = float(np.max(np.abs(compare_to - original)))
            check(difference <= STEP_ZERO_ATOL_AQI, (path.name, "step zero differs", difference))
            max_step_zero_difference = max(max_step_zero_difference, difference)
        else:
            initial[identity[:-1]] = original
        for step, state in states:
            ordered = align(state, expected, FORECAST_KEY, TRUTH, (path.name, step))
            check(np.array_equal(ordered.y_initial, original), (path.name, step, "paired baseline"))
            if variant == "original" and step in (0, settings[ft]["steps"]):
                method = pooled if step == 0 else ft
                original_path = ORIGINAL / "prediction_blocks" / f"{path.stem.split('_b', 1)[0]}_b{budget}_s{seed}_{method}.csv.gz"
                difference = original_parity(state[state.period.isin(TARGET_PERIODS)], original_path, .002)
                max_parity = max(max_parity, difference)
                original_replays += 1
        metrics = error_metrics(pred, group_keys)
        initial_metrics = pred.assign(initial_mae=np.abs(pred.y_initial - pred.y_true)).groupby(
            group_keys, as_index=False).initial_mae.mean()
        metrics = metrics.merge(initial_metrics, on=group_keys, validate="one_to_one")
        compare_metrics(metrics, pd.DataFrame(rec["summaries"]), group_keys,
                        ["n", "mae", "rmse", "initial_mae"], path.name)
        raw_metrics.append(metrics.assign(city=city, budget=budget, seed=seed, family=family, variant=variant))
        check(rec["rows"] == len(pred), (path.name, "prediction row count"))
        rows_checked += len(pred)
    check(seen <= expected_identities and (not complete or seen == expected_identities),
          "invalid adaptation trajectory identities")
    metrics = pd.concat(raw_metrics, ignore_index=True)
    if complete:
        dims = ["city", "budget", "seed", "family", "variant", *group_keys]
        compare_metrics(metrics, read_csv(output / "metrics.csv"), dims,
                        ["n", "mae", "rmse", "initial_mae"], "adaptation metrics")
        macro_dims = ["family", "variant", "step", "period"]
        metrics["mae_delta"] = metrics.mae - metrics.initial_mae
        macro = metrics.groupby(macro_dims, as_index=False)[["mae", "initial_mae", "mae_delta"]].mean()
        compare_metrics(macro, read_csv(output / "macro_descriptive.csv"), macro_dims,
                        ["mae", "initial_mae", "mae_delta"], "adaptation macro baseline contrasts")
    return {"complete": complete, "blocks": len(seen), "expected_blocks": len(expected_identities),
            "prediction_rows": rows_checked, "independent_metric_rows": len(metrics),
            "original_reproductions": original_replays, "max_original_difference_aqi": max_parity,
            "max_cross_variant_step_zero_difference_aqi": max_step_zero_difference,
            "cross_variant_step_zero_tolerance_aqi": STEP_ZERO_ATOL_AQI,
            "blocks_per_city": dict(appearances), "raw_recomputed_every_receipt": True}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("meta", "adaptation"))
    args = parser.parse_args()
    dest = HERE / f"{args.arm}_output_verification.json"
    try:
        report = (audit_meta if args.arm == "meta" else audit_adaptation)()
        report["passed"] = True
    except (AssertionError, OSError, ValueError, KeyError) as exc:
        dest.write_text(json.dumps({"passed": False, "error": str(exc)}, indent=2) + "\n")
        raise
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
