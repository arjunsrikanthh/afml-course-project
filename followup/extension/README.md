# Cross-architecture and MAML-objective controls

These post-hoc controls reuse the original CPCB donor/target split, fitting-label budgets, and previously inspected 2024 and 2025 test windows. The cross-architecture experiment compares the released GRU, LSTM, MLP, TCN, and Transformer adaptation evidence. The MAML experiment changes whether the source training objective rewards forecasts before or after local practice, with archived donor tasks held fixed.

## Reproduce

Create the environment using the [repository instructions](../../README.md), then run:

```sh
python reproduce_extension.py prepare
python reproduce_extension.py cross-smoke
python reproduce_extension.py cross-run --device mps
python reproduce_extension.py cross-audit
python reproduce_extension.py maml-freeze
python reproduce_extension.py maml-run --seed 17
python reproduce_extension.py maml-run --seed 43
python reproduce_extension.py maml-run --seed 71
python reproduce_extension.py maml-broad
python reproduce_extension.py maml-audit
```

`prepare` downloads and verifies the original study archive and previous replay supplement. It stages the exact saved weights, rebuilds archived comparison blocks from the consolidated public forecast table, and pins the protocols. Use `--study-archive` and `--replay-archive` to reuse local copies of the same ZIP files. Apple GPU is the original cross-architecture device. CPU is the default and is also supported for the parity smoke check. MAML uses CPU and the three seeds can run concurrently. The standard run directories are under `runs/followup/results/`. The inherited cross-architecture runner prints the older progress denominator in its per-block log. The frozen protocol and completion check require 432 blocks.

The `results/` directory contains the compact comparison tables, reports, and audit receipts. The new runs recreate the date-level forecasts.
