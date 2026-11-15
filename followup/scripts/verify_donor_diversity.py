#!/usr/bin/env python3
"""Independent panel/episode/forecast checks for the post-hoc donor-count comparison."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
YEARS = {2020: 552, 2021: 828, 2022: 1104, 2023: 828}
CHAINS = {"a": 17, "b": 43, "c": 71}
COUNTS = (6, 12, 23)
SCHEDULES = ("zero", "sgd01x3", "sgd03x10")
KEYS = ["city", "station_id", "date", "issue_date", "period", "horizon"]


def validate_episode_data(episodes, labels):
    assert len(episodes) == 1200
    assert episodes.full_second_order.all() and (episodes.inner_steps == 5).all()
    assert (episodes.inner_lr == .03).all()
    assert (episodes.budget == 90).all() and (episodes.purge_intervening_days >= 9).all()
    assert (episodes.support_examples == 270).all() and (episodes.query_examples == 42).all()
    assert (episodes.support_batch_examples == 32).all() and (episodes.query_batch_examples == 32).all()
    assert np.array_equal(np.sort(episodes.step.unique()), np.arange(1, 301))
    assert (episodes.groupby("step").task.nunique() == 4).all()
    spans = {"support": [], "query": []}
    dates = {city: pd.DatetimeIndex(g.date.sort_values()) for city, g in labels.groupby("city")}
    for row in episodes.itertuples():
        city_dates = dates[row.city]
        assert (row.query_first - row.support_last).days - 1 == row.purge_intervening_days
        for part, size in (("support", 90), ("query", 14)):
            first, last = getattr(row, f"{part}_first"), getattr(row, f"{part}_last")
            chosen = city_dates[(city_dates >= first) & (city_dates <= last)]
            assert len(chosen) == size
            expected = hashlib.sha256("\n".join(chosen.strftime("%Y-%m-%d")).encode()).hexdigest()
            assert expected == getattr(row, f"{part}_label_dates_sha256")
            spans[part].append((last - first).days + 1)
    return {part: {"min": min(vals), "median": float(np.median(vals)), "max": max(vals)}
            for part, vals in spans.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "delivery/submission_20261115/repository/data/frozen")
    parser.add_argument("--output", type=Path, default=ROOT / "results/explanatory_investigation_20261115/donor_diversity")
    parser.add_argument("--panels-only", action="store_true")
    args = parser.parse_args()
    out = args.output
    protocol_file = out / "PROTOCOL.json"
    assert protocol_file.is_file()
    rows = pd.read_csv(args.inputs / "forecast_examples.csv.gz", dtype={"station_id": str},
                       parse_dates=["date", "issue_date", "feature_date_min", "feature_date_max"])
    cohort = pd.read_csv(args.inputs / "cohort.csv")
    donors = set(cohort.loc[cohort.role.isin(["donor", "donor_development"]), "city"])
    targets = set(json.loads((args.inputs / "protocol.json").read_text())["targets"])
    assert not donors & targets and len(donors) == 23 and len(targets) == 6
    eligible = rows[rows.city.isin(donors) & rows.issue_date.ge("2020-01-01") & rows.date.le("2023-12-31")]
    query = rows[rows.city.isin(targets) & rows.issue_date.ge("2024-01-01") &
                 ((rows.date.dt.year == 2025) | rows.date.between("2024-01-04", "2024-12-31"))].copy()
    query["period"] = np.where(query.date.dt.year.eq(2024), "secondary_2024", "test_2025")
    reference = query.set_index(KEYS).y.sort_index()
    assert reference.index.is_unique
    scores = []
    panel_audit = []
    for chain, seed in CHAINS.items():
        block = out / f"chain_{chain}"
        previous = set()
        previous_dates = {}
        for count in COUNTS:
            labels = pd.read_csv(block / f"labels_{count}.csv", parse_dates=["date"])
            names = set(labels.city)
            assert previous.issubset(names) and names.issubset(donors) and len(names) == count
            previous = names
            current_dates = {city: set(g.date) for city, g in labels.groupby("city")}
            assert all(current_dates[city].issubset(prior) for city, prior in previous_dates.items())
            previous_dates = current_dates
            assert not labels.duplicated().any() and len(labels) == 3312
            assert labels.groupby(labels.date.dt.year).size().to_dict() == YEARS
            per_city = labels.groupby(["city", labels.date.dt.year]).size()
            assert all(n == YEARS[y] // count for (_, y), n in per_city.items())
            panel_rows = labels.merge(eligible, on=["city", "date"], how="left", validate="one_to_many")
            assert len(panel_rows) == 9936 and not panel_rows.horizon.isna().any()
            assert (panel_rows.groupby(["city", "date"]).horizon.nunique() == 3).all()
            source_lag_dates = pd.concat([panel_rows[["city", "issue_date"]].assign(
                issue_date=lambda df: df.issue_date - pd.to_timedelta(lag, unit="D")) for lag in range(7)], ignore_index=True)
            information_dates = pd.concat([source_lag_dates.rename(columns={"issue_date": "date"}), labels])
            panel_record = {"chain": chain, "donor_count": count, "city_label_dates": len(labels),
                            "unique_source_city_lag_dates": len(source_lag_dates.drop_duplicates()),
                            "unique_source_city_lag_or_label_dates": len(information_dates.drop_duplicates())}
            panel_audit.append(panel_record)
            if args.panels_only:
                continue
            episodes = pd.read_csv(block / f"episodes_{count}.csv.gz",
                                   parse_dates=["support_first", "support_last", "query_first", "query_last"])
            panel_record["episode_calendar_spans"] = validate_episode_data(episodes, labels)
            for method, updates in (("pooled", 1200), ("maml", 300)):
                stem = f"{method}_{count}"
                receipt = json.loads((block / f"{stem}.json").read_text())
                path = block / f"predictions_{stem}.csv.gz"
                assert receipt["steps"] == updates and receipt["seed"] == seed
                assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["prediction_sha256"]
                state_path = block / f"{stem}.pt"
                assert hashlib.sha256(state_path.read_bytes()).hexdigest() == receipt["checkpoint_sha256"]
                frame = pd.read_csv(path, dtype={"station_id": str}, parse_dates=["date", "issue_date"])
                assert len(frame) == receipt["prediction_rows"] == 3 * len(reference)
                assert (frame.chain == chain).all() and (frame.seed == seed).all()
                assert (frame.method == method).all() and (frame.donor_count == count).all()
                assert set(frame.budget) == {30, 90, 180}
                zero_reference = None
                for budget, selected in frame.groupby("budget"):
                    actual = selected.set_index(KEYS).actual.sort_index()
                    pd.testing.assert_series_equal(actual, reference, check_names=False, check_dtype=False)
                    zero = selected.set_index(KEYS).zero.sort_index()
                    if zero_reference is None:
                        zero_reference = zero
                    else:
                        pd.testing.assert_series_equal(zero, zero_reference)
                    for (city, period, horizon), bucket in selected.groupby(["city", "period", "horizon"]):
                        for schedule in SCHEDULES:
                            err = bucket[schedule].to_numpy() - bucket.actual.to_numpy()
                            assert np.isfinite(err).all()
                            scores.append((chain, seed, count, method, budget, city, horizon, period,
                                           schedule, len(err), np.abs(err).mean(), np.sqrt(np.square(err).mean())))
    audit = {"passed": True, "panel_checks": len(panel_audit), "panels": panel_audit,
             "protocol_sha256": hashlib.sha256(protocol_file.read_bytes()).hexdigest(), "panels_only": args.panels_only}
    if not args.panels_only:
        order = ["chain", "seed", "donor_count", "method", "budget", "city", "horizon",
                 "period", "adaptation", "n", "mae", "rmse"]
        recalculated = pd.DataFrame(scores, columns=order).sort_values(order[:9]).reset_index(drop=True)
        reported = pd.read_csv(out / "city_metrics.csv")[order].sort_values(order[:9]).reset_index(drop=True)
        pd.testing.assert_frame_equal(recalculated, reported, atol=1e-10, rtol=0)
        audit.update({"scored_arms": 18, "city_metric_rows": len(reported), "original_query_rows": len(reference)})
    suffix = "_panels_only" if args.panels_only else ""
    (out / f"INDEPENDENT_AUDIT{suffix}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print({k: v for k, v in audit.items() if k != "panels"})


if __name__ == "__main__":
    main()
