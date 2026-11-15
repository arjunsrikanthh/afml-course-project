"""Calculate endpoint changes in squared forecast error from paired predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results/cross_architecture_followup_20261115"
KEYS = ["city", "budget", "seed", "horizon", "family", "method", "period"]
TERMS = ["direction_term", "correction_squared", "mse_change"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    blocks = sorted((args.output / "prediction_blocks").glob("*.csv.gz"))
    if len(blocks) != 432:
        raise ValueError(f"Expected all 432 prediction blocks, found {len(blocks)}")

    results = []
    for block in blocks:
        frame = pd.read_csv(block, usecols=KEYS + ["step", "y_pred", "y_initial", "y_true"])
        changed = frame[(frame.step == 100) & frame.period.isin(["secondary_2024", "test_2025"])].copy()
        pooled_error = changed.y_initial - changed.y_true
        correction = changed.y_pred - changed.y_initial
        changed["direction_term"] = 2 * pooled_error * correction
        changed["correction_squared"] = correction**2
        changed["mse_change"] = (changed.y_pred - changed.y_true)**2 - pooled_error**2
        np.testing.assert_allclose(
            changed.direction_term + changed.correction_squared, changed.mse_change, rtol=1e-10, atol=1e-9
        )
        results.append(changed.groupby(KEYS, as_index=False)[TERMS].mean())

    cells = pd.concat(results, ignore_index=True)
    cells.to_csv(args.output / "correction_decomposition_cells.csv", index=False)
    summary = cells.groupby(["family", "method", "period"], as_index=False)[TERMS].mean()
    summary.to_csv(args.output / "correction_decomposition.csv", index=False)


if __name__ == "__main__":
    main()
