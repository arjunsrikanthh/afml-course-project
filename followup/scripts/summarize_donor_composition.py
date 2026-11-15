#!/usr/bin/env python3
"""Compact descriptive tables from verified donor-composition scores."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    base = args.results
    macro = pd.read_csv(base / "macro_metrics.csv")
    dims = ["period", "method", "adaptation", "chain"]
    counts = macro.groupby([*dims, "donor_count"], as_index=False).mae.mean()
    pairing = counts.pivot(index=dims, columns="donor_count", values="mae").reset_index()
    pairing.columns = [*dims, "mae_6", "mae_12", "mae_23"]
    pairing["delta_12_6"] = pairing.mae_12 - pairing.mae_6
    pairing["delta_23_6"] = pairing.mae_23 - pairing.mae_6
    pairing["delta_23_12"] = pairing.mae_23 - pairing.mae_12
    pairing.to_csv(base / "chain_average_mae.csv", index=False)
    compact = pairing.groupby(dims[:3], as_index=False).agg(
        mae_6=("mae_6", "mean"), mae_12=("mae_12", "mean"), mae_23=("mae_23", "mean"),
        delta_23_6=("delta_23_6", "mean"), min_chain_delta_23_6=("delta_23_6", "min"),
        max_chain_delta_23_6=("delta_23_6", "max"), n_chains=("chain", "size"))
    compact.to_csv(base / "descriptive_mae_summary.csv", index=False)
    print(compact.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
