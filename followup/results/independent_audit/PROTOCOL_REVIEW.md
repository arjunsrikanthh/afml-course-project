# Independent protocol review — 15 November 2026

Scope: explanatory follow-up to the frozen six-target AQI experiment, with no seasonal extension. Original published source/results remain unchanged. This note reviews identifiability before new outcome inspection; it does not predeclare a positive finding or prove a mechanism.

## Required comparisons

1. Reproduce original target adaptation from the saved original initializations. Original deep GRU uses Adam LR .0003, 100 steps, weight decay .0001; matched pooled GRU uses SGD LR .01, three steps, no decay; MAML/FOMAML/Reptile use SGD LR .03, ten steps, no decay. Preserve clipping at norm 1, CitySampler seed + 991, Adam batch 256 / SGD batch 32, original support order and fixed /100 residual scale. Original CitySampler samples with replacement, using min(batch size, support length) examples per update.
2. Compare all matched initializations at zero steps and under identical SGD schedules and batch sequences. This estimates differences conditional on the saved initializations and frozen tuning. It does not remove differences in pretraining objectives, source training updates, or hyperparameter selection.
3. Lower learning rate, head-only, and parameter-anchor arms should share support examples and batch sequences with the original adaptation. Freeze intervention settings before target outcomes; select any deployment recommendation using donors only. Head-only also changes the trainable parameter count. An anchor against initial parameters is a prescribed regularization intervention, not a measurement of forgetting.
4. Record support loss, parameter displacement, and performance on a fixed donor retention panel at identical trajectory steps. Improved support loss with worse held-out target predictions is consistent with poor transfer generalization; worse donor retention supplies a separate forgetting diagnostic. Neither correlation alone establishes the unique reason adaptation fails.

## Data and analysis guards

- All target fitting labels end by 2023-12-31. B is the number of distinct target label dates shared across three horizons, producing 3B fitting pairs; preceding input context is additional. Verify every support digest against the original fit records.
- Use precisely paired station, city, label date, issue date, horizon, budget and seed keys across arms; verify truth and persistence predictions coincide. Preserve the original chronological issue-date guard and common eligible evaluation dates.
- Both 2024 and 2025 were inspected historically. Trajectory curves are descriptive diagnostic evaluations, not pristine validation or an unbiased choice of the best adaptation step. Do not use their optimum as an operationally selected method.
- Aggregate forecasts to per-city/horizon/budget/seed metrics first, then seed-average and macro-average cities. Forecast rows are dependent and are not independent experimental replicates. Report the six-city scope and avoid national claims.
- Avoid defining success as a mitigation beating only the original failing arm: retain zero-shot and persistence anchors at every matched setting.

## Optional donor-diversity follow-up

The old six-donor ablation changes both task variety and unique source-data volume. A useful new sensitivity fixes the number of unique source examples and the update count, uses balanced city/example sampling, excludes all targets and their geography guard, and repeats independently sampled nested 6/12/23-city subsets with paired seeds. One nested chain confounds diversity with city identities. At least three chains are useful; use subset identity as the experimental unit and report uncertainty accordingly. A fixed number of windows does not necessarily fix unique label dates or information, so disclose all three counts. Meta-episode support/query feasibility and per-city availability must remain comparable. This is a sensitivity to the source composition protocol rather than a universal causal effect of diversity. Defer it if it compromises exact adaptation/meta replay or the runtime bound.

## Completion standard

Freeze protocol/source hashes and original checkpoint hashes before runs; retain progress and completion receipts; independently recompute reported metrics; compare exact original-arm forecasts numerically using a declared backend tolerance. Backend differences can produce a numeric discrepancy without establishing a model bug. No external publication or original-artifact replacement is authorized by this review.
