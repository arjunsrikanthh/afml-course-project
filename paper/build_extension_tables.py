"""Regenerate both submission tables from the archived diagnostic results."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_data"


def rows(filename):
    with (DATA / filename).open(newline="") as source:
        return list(csv.DictReader(source))


def write_table(filename, alignment, headings, table):
    contents = "% Generated from the verified CSV files in figure_data/.\n"
    contents += rf"\begin{{tabular}}{{{alignment}}}\toprule" + "\n"
    contents += " & ".join(headings) + r" \\\midrule" + "\n"
    contents += "".join(" & ".join(row) + r" \\" + "\n" for row in table)
    contents += "\\bottomrule\n\\end{tabular}\n"
    (ROOT / filename).write_text(contents)


def adaptation_table():
    gru = {
        row["variant"]: row for row in rows("adaptation_contrasts.csv")
        if row["family"] == "deep_gru" and row["period"] == "test_2025" and row["step"] == "100"
    }
    table = [[
        "Wider GRU",
        f"{float(gru['original']['target_mae']) - float(gru['original']['target_mae_change']):.2f}",
        f"{float(gru['original']['target_mae_change']):+.2f}",
        f"{float(gru['smaller_lr']['target_mae_change']):+.2f}",
    ]]
    endpoints = {
        (row["family"], row["variant"]): row for row in rows("independent_endpoints.csv")
        if row["period"] == "test_2025"
    }
    for model, label in (("lstm", "LSTM"), ("mlp", "MLP"), ("tcn", "TCN"), ("transformer", "Transformer")):
        original = endpoints[f"deep_{model}", "original"]
        smaller = endpoints[f"deep_{model}", "smaller_lr"]
        table.append([
            label,
            f"{float(original['initial_mae']):.2f}",
            f"{float(original['mae_change']):+.2f}",
            f"{float(smaller['mae_change']):+.2f}",
        ])
    write_table("extension_architecture_rows.tex", "lrrr",
                ["Architecture", "Pooled MAE", r"Original $\Delta$", r"$0.1\times$ rate $\Delta$"], table)


def objective_table():
    scores = {
        (row["objective"], row["steps"]): float(row["mae"])
        for row in rows("objective_macro_metrics.csv")
        if row["source_stage"] == "1350" and row["period"] == "test_2025"
    }
    methods = (
        ("released_pooled", "Released pooled GRU"),
        ("zero_query_objective", "Immediate loss, paired tasks"),
        ("after_five_query_objective", "Post-update loss (MAML)"),
        ("zero_broad_query_objective", "Immediate loss, broader dates"),
    )
    write_table("extension_objective_rows.tex", "lrr", ["Source training", "Zero steps", "Five steps"], [
        [label, f"{scores[method, '0']:.2f}", f"{scores[method, '5']:.2f}"]
        for method, label in methods
    ])


if __name__ == "__main__":
    adaptation_table()
    objective_table()
