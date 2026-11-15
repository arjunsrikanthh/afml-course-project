"""Audit the objective and donor query sampling experiments from their saved rows."""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd

from summarize_explanation import uncertainty
from verify_cross_architecture import (
    FROZEN, ORIGINAL, BUDGETS, KEYS, SEEDS, aligned, digest, frozen_windows, read_csv, read_json,
)


DEFAULT_OUTPUT = Path("results/explanatory_extension_20261115/maml_objective")
ARMS = {
    "episodic": {"zero_query_objective", "after_five_query_objective", "released_pooled", "released_maml"},
    "broad": {"zero_broad_query_objective"},
}
METRIC_KEYS = ["period", "city", "horizon", "budget", "seed", "source_stage", "objective", "rate", "steps"]
STAGES = (300, 1350)


def check_protocol(root, output):
    main = output / "PROTOCOL.json"
    broad = output / "broad_query/PROTOCOL.json"
    for name, expected in read_json(main)["sha256"].items():
        assert digest(root / name) == expected, name
    broad_protocol = read_json(broad)
    assert broad_protocol["paired_protocol_sha256"] == digest(main)
    assert digest(root / "scripts/investigation_20261115/maml_query_sampling_control.py") == broad_protocol["script_sha256"]


def check_receipt(root, output, arm, seed):
    receipt = read_json(output / f"receipt_s{seed}.json")
    assert receipt["seed"] == seed and receipt["archived_final_rngs_exact"]
    assert receipt["protocol_sha256"] == digest(output / "PROTOCOL.json")
    for name, expected in receipt["sha256"].items():
        assert digest(output / name) == expected, name
    journal = read_csv(root / ORIGINAL / "meta_episodes" / f"refit_meta_maml_s{seed}.csv.gz")
    assert len(journal) == 5400
    if arm == "episodic":
        assert receipt["episodes_exact"] == len(journal)
        return
    draws = read_csv(output / f"query_draws_s{seed}.csv.gz")
    draw_fields = ["episodic_query_draws", "broad_query_draws"]
    by_city = draws.groupby("city")[draw_fields].sum()
    assert (by_city.iloc[:, 0] == by_city.iloc[:, 1]).all()
    assert draws.date.max() <= "2023-12-31" and (draws.issue_date < draws.date).all()
    assert (by_city.sum() == 172800).all()
    assert (draws["episodic_query_draws"] > 0).sum() == receipt["episodic_distinct_query_rows"]
    assert (draws["broad_query_draws"] > 0).sum() == receipt["broad_distinct_query_rows"]


def score_frames(raw, expected, expected_arms, seed, stage):
    assert set(raw.seed) == {seed} and set(raw.source_stage) == {stage}
    assert set(raw.method) == expected_arms and set(raw.budget) == set(BUDGETS)
    assert set(raw.steps) == {0, 5}
    assert set(zip(raw.rate, raw.steps)) == {(0., 0), (.03, 5)}
    assert set(raw.city) == set(expected)
    for (city, budget, objective, steps), sample in raw.groupby(["city", "budget", "method", "steps"]):
        aligned(sample, expected[city], (city, budget, objective, steps))
    assert len(raw) == 2 * len(BUDGETS) * len(expected_arms) * sum(len(rows) for rows in expected.values())
    error = raw.y_pred - raw.y_true
    scores = raw.assign(objective=raw.method, absolute=error.abs(), squared=error**2, error=error)
    result = scores.groupby(METRIC_KEYS, as_index=False).agg(
        n=("absolute", "size"), mae=("absolute", "mean"), mse=("squared", "mean"), bias=("error", "mean")
    )
    result["rmse"] = np.sqrt(result.mse)
    return result


def summarize(root, output):
    check_protocol(root, output)
    _, queries, _, _ = frozen_windows(root / FROZEN)
    expected = {city: rows.sort_values(KEYS, ignore_index=True) for city, rows in queries.items()}
    metrics, paired_raw = [], []
    rows_checked = 0
    maximum_reference_difference = 0.
    for arm, expected_arms in ARMS.items():
        path = output if arm == "episodic" else output / "broad_query"
        for seed in SEEDS:
            check_receipt(root, path, arm, seed)
            for stage in STAGES:
                raw = read_csv(path / f"forecasts_s{seed}_step{stage}.csv.gz")
                saved = read_csv(path / f"metrics_s{seed}_step{stage}.csv")
                result = score_frames(raw, expected, expected_arms, seed, stage)
                raw_scored = saved[saved.period.isin(["secondary_2024", "test_2025"]) & saved.steps.isin([0, 5])]
                compared = result.merge(raw_scored, on=METRIC_KEYS, suffixes=("_raw", "_saved"), validate="one_to_one")
                assert len(compared) == len(result) == len(raw_scored)
                for name in ("n", "mae", "mse", "bias", "rmse"):
                    np.testing.assert_allclose(compared[f"{name}_raw"], compared[f"{name}_saved"], atol=1e-9, rtol=1e-10)
                if stage == 1350:
                    columns = ["city", "budget", "date", "horizon", "period", "seed", "steps", "method"]
                    paired_raw.append(raw[columns].assign(ae=(raw.y_pred - raw.y_true).abs()))
                    if arm == "episodic":
                        report = read_json(path / f"receipt_s{seed}.json")
                        maximum_reference_difference = max(
                            maximum_reference_difference,
                            max(r["max_abs_aqi"] for r in report["cpu_replay_vs_released_mps_maml_prediction_difference"]),
                        )
                metrics.append(saved)
                rows_checked += len(raw)
    table = pd.concat(metrics, ignore_index=True)
    macro = table.groupby(["source_stage", "period", "objective", "rate", "steps"], as_index=False)[["mae", "rmse", "bias"]].mean()
    macro.to_csv(output / "objective_macro_metrics.csv", index=False)
    by_seed = table.groupby(["source_stage", "period", "objective", "rate", "steps", "seed"], as_index=False)[["mae", "rmse"]].mean()
    by_seed.to_csv(output / "objective_seed_metrics.csv", index=False)

    raw = pd.concat(paired_raw, ignore_index=True)
    keys = ["city", "budget", "date", "horizon", "period", "seed", "steps"]
    wide = raw.pivot(index=keys, columns="method", values="ae")
    assert wide.notna().all().all()
    contrasts = [
        ("after_five_query_objective", "zero_query_objective"),
        ("zero_query_objective", "zero_broad_query_objective"),
        ("released_maml", "after_five_query_objective"),
    ]
    changes = []
    for left, right in contrasts:
        comparison = (wide[left] - wide[right]).rename("abs_change").reset_index()
        comparison["contrast"] = left + " minus " + right + " at " + comparison.steps.astype(str)
        changes.append(comparison)
    uncertainty(pd.concat(changes, ignore_index=True)).to_csv(output / "objective_intervals.csv", index=False)
    return {
        "passed": True, "raw_forecast_rows_verified": rows_checked,
        "primary_schedules_verified_from_forecasts": [0, 5],
        "other_schedules_reported_from_run_metrics": [3, 10],
        "reconstructed_archived_episode_records": 5400 * len(SEEDS),
        "max_cpu_vs_mps_reference_forecast_difference_aqi": maximum_reference_difference,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / DEFAULT_OUTPUT
    result = summarize(args.root, output)
    text = json.dumps(result, indent=2)
    (output / "independent_verification.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
