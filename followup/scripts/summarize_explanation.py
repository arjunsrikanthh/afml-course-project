"""Summarize conditional post-hoc effects without selecting target schedules."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("results/explanatory_investigation_20261115")
TARGET_KEYS = ["city", "budget", "seed", "horizon"]
META_KEYS = TARGET_KEYS + ["period", "date", "y_true"]
PERIODS = ["secondary_2024", "test_2025"]
ENDPOINTS = {"deep_gru": 100, "matched_gru": 3}


def adaptation_summary(path):
    values = pd.read_csv(path / "metrics.csv")
    initial = values[values.step == 0]
    keys = TARGET_KEYS + ["family", "variant", "period"]
    paired = values.merge(
        initial[keys + ["mae", "residual_mse"]],
        on=keys, suffixes=("", "_zero"), validate="many_to_one",
    )
    paired["mae_change"] = paired.mae - paired.mae_zero
    paired["support_mse_change"] = paired.residual_mse - paired.residual_mse_zero
    support_keys = TARGET_KEYS + ["family", "variant", "step"]
    support = paired.loc[paired.period == "support", support_keys + ["support_mse_change"]]
    targets = paired.loc[paired.period.isin(PERIODS)].drop(columns="support_mse_change")
    targets = targets.merge(support, on=support_keys, validate="many_to_one")
    targets["support_fell_query_rose"] = (targets.support_mse_change < 0) & (targets.mae_change > 0)
    summary = targets.groupby(["family", "variant", "step", "period"], as_index=False).agg(
        target_mae=("mae", "mean"), target_mae_change=("mae_change", "mean"),
        support_residual_mse_change=("support_mse_change", "mean"),
        support_fell_query_rose=("support_fell_query_rose", "sum"),
        cases=("mae_change", "size"), correction_rms=("rms_correction", "mean"),
        relative_parameter_drift=("relative_parameter_l2", "mean"),
    )
    # Retention outcomes are recorded once per adaptation case. Donor horizons
    # repeat across cases because each target adaptation changes the weights.
    donor = paired.loc[paired.period == "seen_donor_retention_2023"]
    donor = donor.groupby(["family", "variant", "step"], as_index=False).agg(
        seen_donor_mae_change=("mae_change", "mean"),
    )
    return summary.merge(donor, on=["family", "variant", "step"], validate="many_to_one")


def meta_summary(path):
    matched = pd.read_csv(path / "paired_meta_minus_pooled.csv")
    matched["beats_pooled"] = matched.mae_minus_pooled < 0
    keys = ["initialization", "lr", "step", "period"]
    compared = matched.groupby(keys, as_index=False).agg(
        target_mae=("mae", "mean"), pooled_mae=("pooled_mae", "mean"),
        mae_minus_pooled=("mae_minus_pooled", "mean"), wins=("beats_pooled", "sum"),
        cases=("mae_minus_pooled", "size"),
    )
    diagnostic = pd.read_csv(path / "support_query_diagnostics.csv")
    diagnostic["support_fell_query_rose"] = (
        (diagnostic.support_mse_change < 0) & (diagnostic.mae_change_from_zero > 0)
    )
    updates = diagnostic.groupby(keys, as_index=False).agg(
        mae_change_from_self_zero=("mae_change_from_zero", "mean"),
        support_residual_mse_change=("support_mse_change", "mean"),
        support_fell_query_rose=("support_fell_query_rose", "sum"),
        cases=("mae_change_from_zero", "size"),
    )
    return compared, updates


def compute_day_changes(adaptation, meta):
    columns = TARGET_KEYS + ["date", "period", "step", "y_pred", "y_true", "y_initial"]
    changes = []
    for family, step in ENDPOINTS.items():
        frames = []
        for path in sorted(adaptation.glob(f"*_{family}_original.csv.gz")):
            frame = pd.read_csv(path, usecols=columns)
            frame = frame.loc[frame.period.isin(PERIODS) & (frame.step == step)].copy()
            frame["abs_change"] = abs(frame.y_pred - frame.y_true) - abs(frame.y_initial - frame.y_true)
            correction = frame.y_pred - frame.y_initial
            frame["direction_term"] = 2 * (frame.y_initial - frame.y_true) * correction
            frame["size_term"] = correction**2
            frames.append(frame)
        assert len(frames) == 54
        batch = pd.concat(frames)
        batch["contrast"] = f"{family}: original endpoint minus zero"
        changes.append(batch)

    cols = META_KEYS + ["initialization", "lr", "step", "y_pred"]
    meta_frames = []
    for path in sorted(meta.glob("*.csv.gz")):
        if path.name.endswith("_donor.csv.gz"):
            continue
        frame = pd.read_csv(path, usecols=cols)
        frame = frame.loc[(frame.step == 0) | ((frame.lr == 0.03) & (frame.step == 10))]
        if frame.initialization.iat[0] in ("pooled", "maml"):
            meta_frames.append(frame)
    assert len(meta_frames) == 108
    all_meta = pd.concat(meta_frames, ignore_index=True)
    for step in (0, 10):
        frame = all_meta[all_meta.step == step]
        left = frame.loc[frame.initialization == "maml", META_KEYS + ["y_pred"]]
        right = frame.loc[frame.initialization == "pooled", META_KEYS + ["y_pred"]]
        paired = left.merge(right, on=META_KEYS, validate="one_to_one", suffixes=("_maml", "_pooled"))
        assert len(paired) == len(left) == len(right)
        paired["abs_change"] = abs(paired.y_pred_maml - paired.y_true) - abs(paired.y_pred_pooled - paired.y_true)
        paired["contrast"] = f"MAML minus pooled: {'zero' if step == 0 else '0.03 x 10'}"
        changes.append(paired)
    return pd.concat(changes, ignore_index=True)


def uncertainty(changes):
    """Resample calendar blocks jointly across all cities; keep cities fixed."""
    rows = []
    for (contrast, period), frame in changes.groupby(["contrast", "period"]):
        data = frame.groupby(["city", "date", "horizon"]).abs_change.mean().reset_index()
        values = data.groupby(["city", "horizon"]).abs_change.mean().unstack().mean(axis=1)
        assert len(values) == 6
        result = {
            "contrast": contrast, "period": period, "mae_difference": values.mean(),
            "positive_cities": int((values > 0).sum()),
            "leave_one_city_out_min": min(values.drop(city).mean() for city in values.index),
            "leave_one_city_out_max": max(values.drop(city).mean() for city in values.index),
        }
        dates = pd.to_datetime(data.date)
        days = (dates - pd.Timestamp(f"{period[-4:]}-01-01")).dt.days.to_numpy()
        assert days.min() >= 0 and days.max() <= 365
        rng = np.random.default_rng(107)
        for width in (14, 28):
            n_blocks = 365 // width + 1
            data["block"] = days // width
            blocks = data.groupby(["block", "city", "horizon"]).abs_change.agg(["sum", "count"])
            sums = blocks["sum"].unstack(["city", "horizon"], fill_value=0).reindex(range(n_blocks), fill_value=0)
            counts = blocks["count"].unstack(["city", "horizon"], fill_value=0).reindex(range(n_blocks), fill_value=0)
            draws = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=4000)
            denominators = draws @ counts.to_numpy()
            eligible = (denominators > 0).all(axis=1)
            assert eligible.sum() > 3900
            bootstrap = ((draws[eligible] @ sums.to_numpy()) / denominators[eligible]).mean(axis=1)
            result[f"block_{width}d_low"], result[f"block_{width}d_high"] = np.quantile(bootstrap, [0.025, 0.975])
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=BASE)
    args = parser.parse_args()
    root = args.results
    adaptation = root / "adaptation"
    meta = root / "meta_initialization"
    adaptation_summary(adaptation).to_csv(root / "adaptation_contrasts.csv", index=False)
    matched, effects = meta_summary(meta)
    matched.to_csv(root / "meta_contrasts.csv", index=False)
    effects.to_csv(root / "meta_adaptation_effects.csv", index=False)
    daily = compute_day_changes(adaptation / "prediction_blocks", meta / "blocks")
    uncertainty(daily).to_csv(root / "contrasts.csv", index=False)
    terms = daily.dropna(subset=["direction_term"])
    terms = terms.groupby(["contrast", "period", "city", "budget", "seed", "horizon"], as_index=False)[
        ["direction_term", "size_term"]
    ].mean()
    decomposition = terms.groupby(["contrast", "period"])[["direction_term", "size_term"]].mean()
    decomposition["mse_change"] = decomposition.direction_term + decomposition.size_term
    decomposition.to_csv(root / "correction_decomposition.csv")


if __name__ == "__main__":
    main()
