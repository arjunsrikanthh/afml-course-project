# Explanatory follow-up

This analysis reuses the original six held-out cities, three fitting-label budgets, three seeds, and one-to-three-day horizons. It examines the original fine-tuning and meta-learning results after the 2024 and 2025 forecasts had already been inspected. The original [frozen study](https://github.com/arjunsrikanthh/afml-course-project/releases/tag/study-v1.0.0) provides the cohort, common forecast masks, source data and most starting weights. The [follow-up release](https://github.com/arjunsrikanthh/afml-course-project/releases/tag/followup-v1.0.0) supplies the exact additional checkpoints and original per-block forecast files needed for replay.

Read the [findings and project comparison](results/README.md) for the experimental conclusions, metric definitions and limitations.

## Reproduce

Use Python 3.12 and install the repository [dependencies](../requirements.txt) in a virtual environment as described in the [main README](../README.md). Run these commands from the checkout root.

```bash
python reproduce_followup.py prepare
python reproduce_followup.py smoke
python reproduce_followup.py audit-inputs
python reproduce_followup.py adaptation
python reproduce_followup.py meta
python reproduce_followup.py summarize
python reproduce_followup.py audit
```

The preparation step downloads and verifies the 225 MB original release archive and the 23 MB replay supplement, then extracts the exact inputs needed by the frozen scripts. An existing verified download can be passed with `--study-archive PATH` or `--replay-archive PATH`. Both archive files and every replay member are checked against pinned SHA-256 digests. The frozen inputs, original fit settings and original model engines are reused. The script stages the directory structure expected by the original runners in `runs/followup/`. All newly generated predictions, receipts and reports are written under `runs/followup/results/explanatory_investigation_20261115/`.

CPU works for the two checkpoint-based runs. The original adaptation run used Apple MPS. Use `--device mps` on both `smoke` and `adaptation` to replay with that backend. Meta initialization replay runs on CPU. Hardware differences may produce small floating-point differences. The smoke command checks synthetic optimization parity, verifies that head-only updates freeze the body, and checks real pooled and target-adapted forecasts from both GRUs against the original study. The full runs check original endpoint parity across the relevant city, seed and budget combinations.

The independent audit validates the input chronology, target labels and archived fitting journals. It recomputes error metrics and parity from the generated prediction blocks after both runs complete. Receipts from the research run are in [results/independent_audit](results/independent_audit/).

Across adaptation variants, the independent audit allows an absolute difference of 0.00002 AQI in separately computed step-zero forecasts. Forecast keys, observed values, input digests and the use of each trajectory's exact step-zero forecasts as its paired baseline are still checked exactly.

## Donor panel

This experiment fits fresh pooled and MAML models. Three nested donor schedules fix total city-label dates and training-update counts within each method. The selected city identities and available dates per city vary across donor counts. Each chain uses one paired training seed. The MAML training budget is shorter than in the frozen study. The comparison estimates sensitivity under this design.

```bash
python reproduce_followup.py donor-check
python reproduce_followup.py donor-run --chain a
python reproduce_followup.py donor-run --chain b
python reproduce_followup.py donor-run --chain c
python reproduce_followup.py donor-summarize
```

Each chain writes to an independent directory. Run them sequentially on one host or in parallel with enough CPU and memory. The preflight verifies the 6/12/23-city nested panels, matching label totals and years, eligible support and query episodes, the nine-day episode purge, and the second-order gradient check. The final command independently checks the episode and forecast ledgers, recomputes errors against the canonical target masks and writes the descriptive summaries. See the [donor-panel results and audit](results/donor_diversity/).

## Protocols

The [adaptation](protocols/adaptation.json), [matched-initialization](protocols/meta_initialization.json) and [balanced donor-panel](protocols/donor_diversity.json) protocols give the exact schedules, scope and training-source hashes. The checkpoint experiments vary the target-update rule or starting weights while reusing the same chronological support and query examples. The source-retention panel uses previously seen donor training examples. The forecasts and post-hoc comparisons describe the selected cohort and years.
