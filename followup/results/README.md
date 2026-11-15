# Why target adaptation helped or hurt AQI forecasts

## Original proposal, TA feedback, and implementation

| Stage | What it called for | What the study delivered |
|---|---|---|
| September one-pager | Forecast Indian station AQI 24–72 hours ahead. Pretrain on data-rich metros, fine-tune with limited target labels, and compare with local scratch and persistence on MAE and RMSE. The team considered LSTM, GRU, Transformer, and MAML. The initial candidate was a Kaggle compilation. A clearly explained negative result counted. | Direct 1-, 2-, and 3-day station AQI forecasts from official CPCB daily files. Available donors were selected by coverage and geographic exclusion, so the 23-donor pool includes cities beyond the proposed metros. Across two benchmark configurations, no fine-tuning cell met the full joint success criterion (0/99). |
| September TA feedback | Use official government measurements. Compare scratch, pooled pretraining and fine-tuning, and meta-learning. Explain the results and tradeoffs. The TA expected ordinary transfer might do better than MAML and flagged the small number of city tasks. | Compared pooled inference with zero target updates, pooled inference after adaptation, local training, and full MAML, FOMAML, and Reptile. Measured which part of the original comparison came from the starting weights and which part came from the target update schedule. |
| Refined implemented study | Ask whether borrowing data from other cities helps, whether updating those weights with local labels helps further, and whether meta-training improves those updates. | Same six target cities and forecast dates, three label budgets, three horizons, and three seeds. Strong simple baselines and shared-backbone controls. Follow-up tests of adaptation rate, trainable layers, anchoring to source weights, and donor composition. The seasonal extension was dropped at the user's direction. |

The two benchmark runs provide the original scientific comparison. The controls below reuse the already inspected 2024 and 2025 periods and test explanations for those results. [Original benchmark release](https://github.com/arjunsrikanthh/afml-course-project/releases/tag/study-v1.0.0)

## Data and training in concrete terms

The original source archive contains 3,820 station-year Excel exports and 769,129 observed station-dates. The modeling cohort uses one station per city. The six targets are Amaravati, Guwahati, Ahmedabad, Pithampur, Ludhiana, and Udaipur. Each retained example has seven consecutive observed AQI days ending on the issue date, calendar features from that date (cyclic day of year and day of week), and a horizon value for one, two, or three days. An issue date of December 20 and horizon 2 uses AQI from December 14–20 to forecast AQI on December 22. The neural model learns the scaled change from the most recent AQI and adds the predicted change back at inference. Inputs use fixed scaling and gaps break windows. Target labels are never imputed. The AQI study uses no weather or separate pollutant measurements. [Official CPCB AQI repository](https://airquality.cpcb.gov.in/ccr/#/repository/aqi)

- **Simple rules.** Persistence predicts the most recent AQI. The seven-day mean smooths recent levels. The weekly lag rule is a fixed reference. These rules require zero parameter fitting.
- **Local tabular baselines.** Ridge, histogram gradient boosting, and random forest test whether a single station's limited fitting data support a useful linear or nonlinear forecast without neural pretraining.
- **Supervised neural models.** MLP, GRU, LSTM, TCN, and Transformer test feed-forward, recurrent, convolutional, and attention-based modeling. Each can be fit from scratch on the target, trained on donors and applied as saved, or trained on donors and then updated on target labels. The original protocol selects candidates on donor-side data, refits source models through 2023, and uses the latest eligible 30, 90, or 180 target label dates through 2023 for local or adaptation fitting. Each date supplies three horizon examples with overlapping inputs. Forecasts roll forward using available past AQI, with weights fixed during evaluation.
- **Matched GRU and meta-learning.** A manual GRU supports the second derivatives used by full MAML. Its architecture is held constant across ordinary pooled training, MAML, FOMAML, and Reptile. MAML simulates within-city transfer. It updates on an earlier support block and trains starting weights by backpropagating the error on a later query block. FOMAML omits derivatives through the inner update. Reptile moves the initialization toward the weights obtained after training on a task. Each method uses the same target label dates when adapting. The original expanded run also compared SGD and Adam adaptation.

The compact and expanded studies cover 23 and 33 methods, respectively. Their counts include baseline and training-configuration combinations. In the original main test, pooled neural training beat the corresponding local neural models in all 99 reported MAE cells. That comparison shows value from borrowing donor data under the label cap. The compact pooled Transformer also beat persistence at all three horizons (MAE 22.24/27.15/29.51 versus 23.12/30.36/34.30). The strict proposal required the full comparison against scratch and persistence to succeed in both MAE and RMSE. [Original protocol and results](https://github.com/arjunsrikanthh/afml-course-project/tree/48b8247)

## Deep GRU updates fit local history and overcorrected future forecasts

We held the starting checkpoint, target dates, sampled training batches, and forecast rows fixed. We replayed the released setting and intervened on the learning rate, which layers could change, and how strongly weights were penalized for moving from the starting checkpoint. The experiment scored every frozen schedule. Each row below is an equal-city average across the same budgets, horizons, and seeds. Positive MAE changes mean worse forecasts.

| Deep GRU at 100 Adam updates | 2025 MAE | Change from pooled | Seen-donor panel MAE change |
|---|---:|---:|---:|
| Original, learning rate 0.0003 | 37.63 | +10.96 | +9.36 |
| One-tenth learning rate | 27.28 | +0.61 | +0.52 |
| Output head only | 28.52 | +1.85 | +1.41 |
| Anchor to initial weights, penalty 0.01 | 33.69 | +7.02 | +6.00 |
| Anchor to initial weights, penalty 0.10 | 30.04 | +3.37 | +2.97 |

The unchanged source model had 2025 MAE 26.67. After the original 100 updates, target support MAE fell from 30.51 to 24.96 while future MAE rose in every city's aggregate. In 149 of 162 city–budget–seed–horizon cases, support residual MSE fell and future MAE rose. This count summarizes correlated comparisons. The same update worsened 2024 MAE by 8.58. The 2025 increase was +8.81 to +12.99 across the central 95% of 14-day calendar-block resamples, and +8.36 to +13.47 with 28-day blocks. These resamples keep the six cities fixed and preserve shared calendar blocks across methods, seeds, and horizons.

The forecast errors also show how the loss arose. If the pooled error is $e_0$ and adaptation changes the forecast by $\delta$, the squared error changes by $2e_0\delta+\delta^2$. For the deep GRU at step 100 in 2025, the average first term was **−132 AQI²** and the squared correction term was **+1,225 AQI²**. The corrections had some alignment with the original errors on average, but the size of the corrections outweighed that benefit. The net increase was **+1,093 AQI²**. The fact that slowing, restricting, or anchoring updates reduced the damage supports a mechanism of overly aggressive adaptation under the released setting. Donor-panel degradation is consistent with loss of previously useful behavior. That panel comes from the donor refit period and measures retention on seen data.

The small matched GRU used a much shorter original update (SGD, 0.01, three steps). Its 2025 change was **+0.008 AQI** and its 2024 change was **−0.230**. Extending to 30 steps at the same rate raised 2025 MAE by **0.724**. Moving further in weight space can hurt even when the original short update is essentially neutral. These are fixed diagnostic settings. The study did not select an adaptation setting on either evaluation year. [Weight anchoring precedent](https://proceedings.mlr.press/v80/li18a.html)

## MAML began behind pooled training and improved from its own start

The original MAML setting used SGD at 0.03 for ten steps. Ordinary matched GRU fine-tuning used SGD at 0.01 for three steps. To separate starting weights from adaptation, all four saved initializations used the same support sample stream, both learning rates, and 1/3/10/30 updates. We also compared their zero-update forecasts.

| Saved initialization | 2025 MAE before target updates | 2025 MAE after identical SGD 0.03 × 10 | Change from own zero |
|---|---:|---:|---:|
| Pooled | 26.677 | 27.498 | +0.821 |
| Full MAML | 28.267 | 27.819 | −0.448 |
| FOMAML | 27.406 | 27.765 | +0.359 |
| Reptile | 29.054 | 29.300 | +0.246 |

MAML started **1.590 AQI** behind pooled and improved against its own start at **seven of eight** schedules in 2025. The published gap between original MAML and original ordinary fine-tuning was **1.134 AQI**. At identical 0.03 × 10 it shrank to **0.321**. At identical 0.01 × 3 it was **1.054**. Across the nine common schedules, pooled had lower broad 2025 mean MAE than each meta initialization. MAML at identical 0.03 × 10 beat pooled in four of six individual city aggregates, while its larger deficits in the other two cities kept its six-city average higher. In 2024 the same matched MAML comparison instead favored MAML by **0.145**, with a 14-day block interval spanning zero. The 2025 MAML shortfall reflects both the weaker initial forecast and the schedule used in the comparison. [Original MAML paper](https://proceedings.mlr.press/v70/finn17a.html)

## What was predictable and what we measured

The TA anticipated that ordinary transfer might beat MAML. That expectation was reasonable given the low number of independent cities. It did not specify the sign or size of the effect for each initialization, horizon, year, and update schedule. Prior work applied transfer to Indian PM₂.₅ and to limited-data daily PM₁₀. A different cold-start pollution study reported a 5.2% MAE advantage for its meta-learning method under its own 200-hour experiment, which illustrates why the expected direction needed testing. The empirical contribution here is the measured comparison for these official overall-AQI windows, including a specific, testable explanation for the deep GRU deterioration and the distinction between MAML's starting disadvantage and its ability to adapt. This work makes no priority claim. [Indian PM₂.₅ thesis](https://zenodo.org/records/20790070) · [Daily PM₁₀ transfer study](https://www.nature.com/articles/s41598-025-86550-6) · [Cold-start meta-learning study](https://onlinelibrary.wiley.com/doi/10.1155/2023/3734557)

## What changing the number of donors showed

The original six-versus-23 donor comparison also changed the amount of donor training data. This control used three nested donor panels (6, 12, and 23 cities) per chain, with three independent panel chains and a matched training seed within each chain. Each arm used 3,312 city-label dates, 9,936 horizon examples, and the same year totals. Supervised models received 1,200 updates in each arm. Full MAML received 300 outer updates with 90-date support and 14-date later query blocks. We shortened meta-training for the expedited investigation. The original MAML used 1,350 outer updates and mixed task budgets.

| 2025 six-city macro MAE | 6 donors | 12 donors | 23 donors |
|---|---:|---:|---:|
| Pooled, no update | 27.267 | 26.788 | 26.734 |
| Pooled, SGD 0.01 × 3 | 27.247 | 26.771 | 26.775 |
| Pooled, SGD 0.03 × 10 | 28.797 | 28.092 | 27.890 |
| MAML, no update | 29.009 | 28.786 | 29.338 |
| MAML, SGD 0.01 × 3 | 28.976 | 28.769 | 29.228 |
| MAML, SGD 0.03 × 10 | 29.077 | 28.943 | 29.263 |

Going from six to 23 cities improved pooled MAE and worsened short-run MAML MAE in all three paired chains and all three target schedules in 2025. Moving from six to 12 donors often helped MAML, so the response was nonmonotonic. As donor count grew, per-city labels became sparser, support episodes stretched across more calendar days, and the windows touched more distinct historical AQI readings. The fixed count of labels therefore controls one part of exposure. These data show sensitivity to how the same label budget is distributed among cities, and leave the pure effect of donor count unresolved. They do not establish that 23 donors are too few, that more tasks would fix MAML, or that the original 1,350-step result arose from donor count alone. [Task-diversity experiment in prior work](https://ojs.aaai.org/index.php/AAAI/article/view/26012)

## Boundaries and remaining challenges

- **Original goal.** The project completed the forecast and comparison requested in the proposal. The original accuracy target failed, and the deep-GRU and matched-initialization controls characterize that negative result. The six-city study cannot establish how often the result holds nationally.
- **Data scarcity.** Budgets simulate few fitting labels on stations that have adequate evaluation coverage. Moving from 30 to 180 labels also changes how far back the support history reaches. The result concerns this exact label-selection policy. Establishing performance in naturally sparse cities requires a separately sampled evaluation cohort.
- **Data access and aging.** The CPCB archive identifies daily 4 PM observations. It provides no verified live publication lag here. The experiment compares frozen 2023 weights under 2024 and 2025 conditions and does not estimate the benefit of routine operational retraining.
- **Underlying AQI dynamics.** The controlled results identify how parameter updates affected these forecasts. The input data cannot resolve physical drivers of local pollution or which omitted covariates caused particular forecast errors.

## Reproduction and audit

The frozen protocols fix every tested variant. Independent checks recomputed metrics and verified exact query and support rows across 540 adaptation blocks and 216 meta blocks, totaling 11,085,336 forecasts. Across 486 archived baseline block comparisons, maximum prediction difference was 0.000030 AQI. All 31 original artifact hashes matched. The donor study completed all 18 fits, two chains on the Mac and one on Instabox 1. Its independent audit checked all nine input panels and recomputed 5,832 metric rows. See `adaptation_contrasts.csv`, `meta_contrasts.csv`, `meta_adaptation_effects.csv`, `contrasts.csv`, `correction_decomposition.csv`, `donor_diversity/`, and `independent_audit/`.
