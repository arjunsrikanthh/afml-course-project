# Why the released MAML model started behind

The released MAML GRU produced more accurate forecasts after target updates than it did before updating. Its unadapted 2025 forecasts had a mean absolute error (MAE) of 28.27 AQI points, compared with 26.68 for the ordinary pooled GRU. The first follow-up showed that the starting gap mattered. This experiment tests how the starting gap arose.

## Train two copies of the same model

Each pair starts from identical weights and sees the same donor-city tasks in the same order. A task contains an earlier set of dates for practice and a later set for checking the forecast. Both arms see the same checked examples, take 1,350 training updates, use the same optimizer, and repeat across the same three seeds.

- **Train for accuracy right away.** Measure the forecast error on those later examples without first adjusting the model to that city.
- **Train for accuracy after practice.** Take five practice updates on that city's earlier examples, then measure error on the later examples. This is the released full-MAML objective.

The experiment replayed all 16,200 archived donor-task records and matched both archived random-number generator states for each seed. The reproduced MAML forecasts differed from the released forecasts by at most 0.000026 AQI points. This makes the direct comparison informative about the released MAML result.

## The training goal explains most of the starting gap

| Training condition | 2025 MAE before any local updates | 2025 MAE after the same five local updates |
|---|---:|---:|
| Released pooled GRU | 26.68 | 27.14 |
| Train for accuracy right away, same donor episodes | 26.94 | 27.45 |
| Train for accuracy after practice, same donor episodes (MAML replay) | 28.27 | 27.56 |
| Train for accuracy right away, spread examples across all donor dates | 26.82 | 27.47 |

Changing the training goal from after-practice to immediate accuracy reduced starting error by **1.33 of the original 1.59-point gap**, about **83%**. The 2025 fixed-city 14-day block interval for the difference was **0.72 to 1.98 AQI points**, with a similar result using 28-day blocks. The 2024 starting gap in the paired comparison was **1.24 points**. At the 300-update checkpoint the paired starting gap was much smaller, so the full-length result depends on training long enough for the objectives to diverge.

Giving both copies the same five updates on the target city made much of the difference disappear. MAML improved by **0.70** points in 2025, whereas the matched immediate-accuracy copy worsened by **0.51**. Their resulting 0.11-point gap had a block interval spanning zero. In 2024 the paired five-update comparison favored MAML by **0.16**, also with an interval spanning zero. The MAML training goal made it more responsive to the intended practice updates. Its advantage in improvement from the start did not yield a clear post-update forecast advantage over the tightly matched control.

As a separate check, both immediate-accuracy controls trained on the same number of examples per city. Drawing from the full available history of that donor instead of from the episode's later checking dates lowered 2025 starting error by **0.12** points. The date-block interval crossed zero. The original pooled model's training procedure made other choices too, so a small part of the original gap remains unresolved.

These are controlled, post-hoc explanations for this GRU, task setup and evaluation cohort. The intervals resample calendar blocks with the six cities fixed. The experiment does not estimate a minimum sufficient number of cities or a result for other tasks.

## Evidence

`objective_macro_metrics.csv` includes both stages, both evaluation years and all four target update schedules. `objective_intervals.csv` records the paired intervals and leave-one-city-out sensitivity. The independent check recomputed MAE and RMSE from 1,631,880 stored forecasts at zero and five local updates. The three- and ten-update rows are original run summaries. `PROTOCOL.json` and `broad_query/PROTOCOL.json` pin code and inputs.
