# Fine-tuning across the remaining neural models

The earlier follow-up showed that the wider GRU fit its local practice data more closely while making later forecasts worse. This extension repeats the same learning-rate experiment for the remaining four neural architectures in the expanded benchmark. A separate [MAML objective experiment](../maml_objective/README.md) explains how MAML's starting deficit arose.

## What changed

For each saved LSTM, MLP, TCN, and Transformer checkpoint, we reproduced the released pooled forecast and the original 100 Adam updates at learning rate 0.0003. Then we repeated the 100 updates at 0.00003, starting with the same weights and using the same training batch stream. We recorded steps 0, 1, 3, 10, 30, and 100 for six cities, 30/90/180 local fitting dates, three seeds, and three forecast horizons. The fitting examples end in 2023. The original 2024 and 2025 dates and actuals are unchanged. The new analysis is post hoc, and every fixed setting is reported.

## Lower rates reduced future forecast harm in every architecture

2025 mean absolute errors are in AQI points. Deltas are relative to the same architecture's pooled starting forecast. The support MAE change summarizes the old local examples used for the updates.

| Architecture | Pooled MAE | Original 100-update MAE | One-tenth-rate MAE | Original change on old local examples | Original change on 2025 forecasts | One-tenth-rate change on 2025 forecasts |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 29.17 | 33.55 | 29.26 | −1.99 | +4.38 | +0.08 |
| MLP | 27.10 | 37.55 | 29.19 | −7.84 | +10.45 | +2.09 |
| TCN | 26.65 | 42.38 | 27.65 | −12.18 | +15.73 | +1.01 |
| Transformer | 26.60 | 39.72 | 27.98 | −7.68 | +13.12 | +1.38 |
| Wider GRU (earlier identical-rate control) | 26.67 | 37.63 | 27.28 | −5.56 | +10.96 | +0.61 |

The original 100 updates improved the fit to the old local examples and raised future MAE for each of the five architectures. Smaller updates reduced the 2025 penalty by about **80% to 98%** for the four newly tested models. Their original 2024 penalties ranged from **+3.40 to +13.34** points and dropped to **−0.04 to +1.45** with the lower rate. On average, the pooled starting forecast still had lower 2025 MAE for all five architectures.

The date-level squared-error accounting supports the same explanation for all four. If the original error is `e` and an update produces a forecast correction `d`, the squared forecast error changes by `2ed + d²`. All four original settings had a negative average `2ed`, meaning the direction of the correction was helpful on average. The corrections were large enough that `d²` dominated. Reducing the learning rate shrank that size term and most of the harm. This gives evidence for corrections that were too large in these settings. The experiments do not establish a universally optimal rate or why a particular station's air quality changed.

## Evidence

`independent_endpoints.csv` contains the complete endpoint summaries, support/query case counts, and MAE/RMSE changes. `endpoint_block_intervals.csv` gives paired 14- and 28-day calendar-block intervals conditional on the six cities. `macro_descriptive.csv` covers all steps and years. `correction_decomposition.csv` records the paired direction and correction-size terms. `independent_verification.json` records raw forecast, archived replay, and input integrity checks.
