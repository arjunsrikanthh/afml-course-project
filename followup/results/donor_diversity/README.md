# Equal-volume donor-panel sensitivity

The original 6- versus 23-donor comparison used 19,404 versus 71,829 horizon rows. This follow-up assigns 9,936 horizon rows (3,312 city–label dates) to every arm, with the same yearly totals from 2020–2023. The three donor panels are nested within each chain. The narrower panels give each donor more labels per year. Donor identities, input-window overlap and meta-task calendar span can therefore vary with panel width.

Three independently hash-ranked chains use paired training seeds 17, 43 and 71. Within each chain we compare 6, 12 and all 23 donors with the same explicit two-layer GRU. Pooled training uses 1,200 updates. Full second-order MAML uses 300 outer updates, four episodes per update, and 90 support plus 14 query label dates separated by at least nine full intervening calendar days. The original MAML refit used 1,350 updates and variable support budgets. These fixed shorter runs measure sensitivity under the declared compute budget.

Each initialization is tested before target adaptation and after the same two target SGD schedules (0.01 for 3 updates or 0.03 for 10). All schedules use the original 30/90/180 target support budgets through 2023 and the original six-city 2024/2025 evaluation masks. No schedule or step is selected from the results. Each MAE summary first averages absolute error over dates within city, then weights the six cities and the three horizons/budgets equally. The reported range is the minimum and maximum across the three paired chains.

## Results

| 2025 adaptation | Pooled, 6 / 12 / 23 donors | MAML, 6 / 12 / 23 donors |
|---|---:|---:|
| Zero-shot | 27.267 / 26.788 / 26.734 | 29.009 / 28.786 / 29.338 |
| SGD, .01 × 3 | 27.247 / 26.771 / 26.775 | 28.976 / 28.769 / 29.228 |
| SGD, .03 × 10 | 28.797 / 28.092 / 27.890 | 29.077 / 28.943 / 29.263 |

At all three matched target schedules, every chain's 2025 pooled MAE fell from 6 to 23 donors and every chain's MAML MAE rose. At 12 donors the three-chain MAML mean was lower than at 6 or 23 under each schedule. On 2024, the 23-versus-6 comparison had mixed chain directions for pooled under all three schedules and for MAML at the longer schedule. The shortened episodic MAML run therefore shows sensitivity to how donors and dates are distributed at equal horizon-row volume. These runs provide no evidence that increasing donor count from 6 to 23 rescues MAML at this source volume and compute budget.

At 6, 12, and 23 donors, a 90-label support episode spanned a median of 223–224, 416–417, and 811–816 calendar days across chains, respectively. The underlying unique city-dates used as labels or lags were 6,947–7,815, 12,871–13,632, and 19,180–19,445. The wider panels changed both episode duration and the overlap among lag windows while equalizing horizon rows and within-method optimization budgets. The results cannot identify which difference drove the forecast changes.

## Reproduction

Run the commands under [Donor panel](../../README.md#donor-panel) from a repository checkout.

Chains a and b were run on Mac CPU, and chain c on Instabox 1 CPU. Comparisons are paired within each chain on the same hardware. [The frozen protocol](../../protocols/donor_diversity.json) records hashes and frozen choices. `PREFLIGHT.json` audits panel and episode feasibility. `INDEPENDENT_AUDIT.json` checks source volumes, MAML episode date hashes, forecast masks, artifact hashes, and metrics. `city_metrics.csv`, `macro_metrics.csv`, `paired_donor_count_changes.csv`, `chain_average_mae.csv` and `descriptive_mae_summary.csv` carry the results.
