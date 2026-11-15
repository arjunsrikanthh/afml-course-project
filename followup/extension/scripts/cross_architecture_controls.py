"""Extend the frozen GRU adaptation controls to four released neural models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import adaptation_controls as controls


RELEASED_ARCHITECTURES = ("lstm", "mlp", "tcn", "transformer")
FAMILIES = {
    f"deep_{name}": (name, f"deep_{name}_finetune", f"deep_{name}_pooled_zero", [0, 1, 3, 10, 30, 100])
    for name in RELEASED_ARCHITECTURES
}
VARIANTS = {name: controls.VARIANTS[name] for name in ("original", "smaller_lr")}
DEFAULT_OUTPUT = controls.ROOT / "results/cross_architecture_followup_20261115"
EXPECTED_BLOCKS = 6 * 3 * 3 * len(FAMILIES) * len(VARIANTS)


def freeze():
    path = controls.OUT / "protocol.json"
    if not path.exists():
        protocol = controls.protocol()
        protocol["wrapper_sha256"] = controls.sha(__file__)
        protocol["question"] = (
            "Test whether the released 100-step adaptation degrades forecasts and "
            "whether a tenfold learning-rate reduction mitigates that degradation "
            "for LSTM, MLP, TCN, and Transformer."
        )
        protocol["expected_blocks"] = EXPECTED_BLOCKS
        protocol["analysis"] = (
            "Primary 2025 endpoint comparisons: original minus zero and smaller-LR "
            "minus original, reporting residual support MSE and future MAE/RMSE. "
            "2024 repeated as a secondary evaluation. Score every frozen step and "
            "variant without target selection."
        )
        controls.write(path, protocol)
    protocol = json.loads(path.read_text())
    if protocol["wrapper_sha256"] != controls.sha(__file__):
        raise ValueError("Wrapper changed after protocol freeze")
    if protocol["investigation_script_sha256"] != controls.sha(controls.__file__):
        raise ValueError("Adaptation engine changed after protocol freeze")
    for name, expected in protocol["source_states_sha256"].items():
        if controls.sha(controls.ROOT / name) != expected:
            raise ValueError(f"Source checkpoint changed: {name}")
    return protocol


def summarize():
    receipts = [json.loads(path.read_text()) for path in sorted((controls.OUT / "receipts").glob("*.json"))]
    if len(receipts) != EXPECTED_BLOCKS:
        raise ValueError(f"Expected {EXPECTED_BLOCKS} complete blocks, got {len(receipts)}")
    metrics = pd.DataFrame([row for receipt in receipts for row in receipt["summaries"]])
    metrics.to_csv(controls.OUT / "metrics.csv", index=False)
    keys = ["family", "variant", "step", "period"]
    macro = metrics.groupby(keys, as_index=False)[["mae", "rmse", "initial_mae", "residual_mse"]].mean()
    macro["mae_delta"] = macro.mae - macro.initial_mae
    macro.to_csv(controls.OUT / "macro_descriptive.csv", index=False)
    controls.write(controls.OUT / "completion.json", {
        "complete": True,
        "blocks": len(receipts),
        "prediction_rows": sum(receipt["rows"] for receipt in receipts),
        "max_original_prediction_difference": max(receipt["original_max_abs"] or 0 for receipt in receipts),
        "all_metrics_finite": bool(np.isfinite(metrics.select_dtypes("number")).all().all()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "parity", "run", "summarize"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    device = torch.device(args.device)
    controls.OUT = args.output
    controls.FAMILIES = FAMILIES
    original_variants = controls.VARIANTS
    controls.VARIANTS = VARIANTS
    controls.freeze = freeze
    controls.summarize = summarize

    # The original synthetic smoke includes a head-freeze assertion. Keep it
    # during smoke while excluding that variant from the actual comparison.
    original_smoke = controls.smoke

    def smoke(smoke_device):
        controls.VARIANTS = original_variants
        try:
            original_smoke(smoke_device)
        finally:
            controls.VARIANTS = VARIANTS

    controls.smoke = smoke
    freeze()
    if args.action == "parity":
        torch.set_num_threads(2)
        controls.actual_parity(controls.inputs(), controls.config(), device)
    elif args.action == "run":
        controls.run(device)
    elif args.action == "summarize":
        summarize()


if __name__ == "__main__":
    main()
