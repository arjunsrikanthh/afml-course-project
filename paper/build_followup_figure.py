"""Render the deep-GRU follow-up figure from its complete recorded trajectories."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "figure_data" / "adaptation_contrasts.csv"
STEPS = [0, 1, 3, 10, 30, 100]
ARMS = {
    "original": ("Original", "#26394e", "o"),
    "smaller_lr": ("0.1× rate", "#238876", "s"),
    "head_only": ("Head only", "#8052a0", "^"),
    "anchor_0p01": ("Anchor 0.01", "#c96b39", "D"),
    "anchor_0p1": ("Anchor 0.1", "#ae4167", "P"),
}


def main():
    scores = pd.read_csv(DATA).query("family == 'deep_gru' and period == 'test_2025'")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.48, 2.76), sharex=True, layout="constrained")
    for arm, (label, color, marker) in ARMS.items():
        data = scores[scores.variant == arm].sort_values("step")
        assert data.step.tolist() == STEPS and (data.cases == 162).all()
        style = {"label": label, "color": color, "marker": marker, "ms": 3, "lw": 1.2}
        axes[0].plot(range(len(STEPS)), data.support_residual_mse_change, **style)
        axes[1].plot(range(len(STEPS)), data.target_mae_change, **style)
    for axis in axes:
        axis.axhline(0, color="#73787f", lw=0.7, zorder=0)
        axis.grid(axis="y", color="#dce1e7", lw=0.55, zorder=0)
    axes[0].set_ylabel("Support MSE change")
    axes[1].set_ylabel("2025 MAE change")
    axes[1].set(xticks=range(len(STEPS)), xticklabels=STEPS, xlabel="Target update count")
    axes[0].legend(loc="lower left", ncols=2, frameon=False, columnspacing=0.8)
    fig.savefig(HERE / "figures" / "adaptation_diagnostics.pdf")


if __name__ == "__main__":
    main()
