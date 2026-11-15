# Expanded research results

This study was specified before the compact reference target scores were inspected. It expands computational effort to answer capacity/optimization, meta-gradient, adaptation-optimizer and donor-pool questions. All conclusions below are calculated from saved observed-target forecasts. The target periods did not tune any model.

## Larger models and donor-only search

Five deeper architectures use the same legal seven-day inputs and target fitting-label budgets as the compact reference. Their training and search budgets are larger. The comparison describes the combined capacity/optimization package; it cannot isolate parameter count as a causal factor. A compact model remains a serious benchmark.

| budget | horizon | method | mae | rmse | skill_mae | cities_improved_vs_persistence | n_cities |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | deep_tcn_pooled_zero | 22.325 | 31.943 | 0.035 | 4 | 6 |
| 30 | 1 | deep_transformer_pooled_zero | 22.434 | 32.068 | 0.030 | 4 | 6 |
| 30 | 1 | deep_gru_pooled_zero | 22.450 | 32.036 | 0.029 | 4 | 6 |
| 30 | 1 | matched_gru_pooled_zero | 22.483 | 32.106 | 0.028 | 4 | 6 |
| 30 | 1 | deep_mlp_pooled_zero | 22.543 | 32.275 | 0.025 | 4 | 6 |
| 30 | 2 | deep_transformer_pooled_zero | 27.525 | 38.093 | 0.093 | 6 | 6 |
| 30 | 2 | matched_gru_pooled_zero | 27.542 | 38.156 | 0.093 | 6 | 6 |
| 30 | 2 | deep_gru_pooled_zero | 27.594 | 38.085 | 0.091 | 6 | 6 |
| 30 | 2 | deep_tcn_pooled_zero | 27.596 | 38.086 | 0.091 | 6 | 6 |
| 30 | 2 | matched_gru_finetune | 27.674 | 38.242 | 0.089 | 6 | 6 |
| 30 | 3 | deep_transformer_pooled_zero | 29.851 | 40.778 | 0.130 | 6 | 6 |
| 30 | 3 | deep_gru_pooled_zero | 29.964 | 40.863 | 0.127 | 6 | 6 |
| 30 | 3 | matched_gru_pooled_zero | 30.006 | 41.058 | 0.125 | 6 | 6 |
| 30 | 3 | deep_tcn_pooled_zero | 30.017 | 41.204 | 0.125 | 6 | 6 |
| 30 | 3 | matched_gru_finetune | 30.119 | 41.142 | 0.122 | 6 | 6 |
| 90 | 1 | deep_tcn_pooled_zero | 22.325 | 31.943 | 0.035 | 4 | 6 |
| 90 | 1 | deep_transformer_pooled_zero | 22.434 | 32.068 | 0.030 | 4 | 6 |
| 90 | 1 | deep_gru_pooled_zero | 22.450 | 32.036 | 0.029 | 4 | 6 |
| 90 | 1 | matched_gru_finetune | 22.454 | 32.069 | 0.029 | 4 | 6 |
| 90 | 1 | matched_gru_pooled_zero | 22.483 | 32.106 | 0.028 | 4 | 6 |
| 90 | 2 | matched_gru_finetune | 27.484 | 38.107 | 0.095 | 6 | 6 |
| 90 | 2 | deep_transformer_pooled_zero | 27.525 | 38.093 | 0.093 | 6 | 6 |
| 90 | 2 | matched_gru_pooled_zero | 27.542 | 38.156 | 0.093 | 6 | 6 |
| 90 | 2 | deep_gru_pooled_zero | 27.594 | 38.085 | 0.091 | 6 | 6 |
| 90 | 2 | deep_tcn_pooled_zero | 27.596 | 38.086 | 0.091 | 6 | 6 |
| 90 | 3 | deep_transformer_pooled_zero | 29.851 | 40.778 | 0.130 | 6 | 6 |
| 90 | 3 | matched_gru_finetune | 29.923 | 40.992 | 0.128 | 6 | 6 |
| 90 | 3 | deep_gru_pooled_zero | 29.964 | 40.863 | 0.127 | 6 | 6 |
| 90 | 3 | matched_gru_pooled_zero | 30.006 | 41.058 | 0.125 | 6 | 6 |
| 90 | 3 | deep_tcn_pooled_zero | 30.017 | 41.204 | 0.125 | 6 | 6 |
| 180 | 1 | deep_tcn_pooled_zero | 22.325 | 31.943 | 0.035 | 4 | 6 |
| 180 | 1 | deep_transformer_pooled_zero | 22.434 | 32.068 | 0.030 | 4 | 6 |
| 180 | 1 | matched_gru_finetune | 22.443 | 32.068 | 0.029 | 4 | 6 |
| 180 | 1 | deep_gru_pooled_zero | 22.450 | 32.036 | 0.029 | 4 | 6 |
| 180 | 1 | matched_gru_pooled_zero | 22.483 | 32.106 | 0.028 | 4 | 6 |
| 180 | 2 | matched_gru_finetune | 27.495 | 38.117 | 0.094 | 6 | 6 |
| 180 | 2 | deep_transformer_pooled_zero | 27.525 | 38.093 | 0.093 | 6 | 6 |
| 180 | 2 | matched_gru_pooled_zero | 27.542 | 38.156 | 0.093 | 6 | 6 |
| 180 | 2 | deep_gru_pooled_zero | 27.594 | 38.085 | 0.091 | 6 | 6 |
| 180 | 2 | deep_tcn_pooled_zero | 27.596 | 38.086 | 0.091 | 6 | 6 |
| 180 | 3 | deep_transformer_pooled_zero | 29.851 | 40.778 | 0.130 | 6 | 6 |
| 180 | 3 | matched_gru_finetune | 29.938 | 40.998 | 0.127 | 6 | 6 |
| 180 | 3 | deep_gru_pooled_zero | 29.964 | 40.863 | 0.127 | 6 | 6 |
| 180 | 3 | matched_gru_pooled_zero | 30.006 | 41.058 | 0.125 | 6 | 6 |
| 180 | 3 | deep_tcn_pooled_zero | 30.017 | 41.204 | 0.125 | 6 | 6 |

The lowest test error is descriptive after comparing many methods. It is not a new donor-selected deployment decision.

## Full second-order MAML and matched controls

Full MAML differentiates the query loss through support-gradient updates. FOMAML omits those Hessian terms; Reptile moves toward support-adapted weights. Their manual differentiable GRU backbone and native SGD adaptation are matched to ordinary transfer controls. Correctness checks include a genuine nonzero difference between first- and second-order gradients on a synthetic nonlinear task. These checks verify the implementation; target errors determine usefulness.

### Meta-learning versus ordinary matched transfer

Lower macro MAE occurred in 0/27 prespecified cells; 0 intervals lie entirely below zero. Negative differences favor the listed method. Intervals are unadjusted for multiple comparisons and condition on the selected six target cities.

| budget | horizon | method | reference | delta_mae | delta_mae_ci_low | delta_mae_ci_high | cities_improved | n_cities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | maml_gru_sgd | matched_gru_finetune | 2.626 | 1.598 | 3.117 | 0 | 6 |
| 30 | 1 | fomaml_gru_sgd | matched_gru_finetune | 2.613 | 1.710 | 3.128 | 0 | 6 |
| 30 | 1 | reptile_gru_sgd | matched_gru_finetune | 0.956 | 0.187 | 1.714 | 2 | 6 |
| 30 | 2 | maml_gru_sgd | matched_gru_finetune | 1.914 | 0.899 | 2.429 | 2 | 6 |
| 30 | 2 | fomaml_gru_sgd | matched_gru_finetune | 1.907 | 1.027 | 2.440 | 0 | 6 |
| 30 | 2 | reptile_gru_sgd | matched_gru_finetune | 2.930 | 1.990 | 4.019 | 0 | 6 |
| 30 | 3 | maml_gru_sgd | matched_gru_finetune | 1.963 | 0.913 | 2.645 | 2 | 6 |
| 30 | 3 | fomaml_gru_sgd | matched_gru_finetune | 2.021 | 1.081 | 2.662 | 1 | 6 |
| 30 | 3 | reptile_gru_sgd | matched_gru_finetune | 4.353 | 3.383 | 5.807 | 0 | 6 |
| 90 | 1 | maml_gru_sgd | matched_gru_finetune | 1.044 | 0.477 | 1.307 | 1 | 6 |
| 90 | 1 | fomaml_gru_sgd | matched_gru_finetune | 0.963 | 0.506 | 1.259 | 1 | 6 |
| 90 | 1 | reptile_gru_sgd | matched_gru_finetune | 0.765 | 0.028 | 1.485 | 2 | 6 |
| 90 | 2 | maml_gru_sgd | matched_gru_finetune | 0.360 | -0.217 | 0.638 | 3 | 6 |
| 90 | 2 | fomaml_gru_sgd | matched_gru_finetune | 0.372 | -0.093 | 0.685 | 3 | 6 |
| 90 | 2 | reptile_gru_sgd | matched_gru_finetune | 2.742 | 1.811 | 3.821 | 0 | 6 |
| 90 | 3 | maml_gru_sgd | matched_gru_finetune | 0.240 | -0.384 | 0.742 | 3 | 6 |
| 90 | 3 | fomaml_gru_sgd | matched_gru_finetune | 0.373 | -0.122 | 0.886 | 2 | 6 |
| 90 | 3 | reptile_gru_sgd | matched_gru_finetune | 4.066 | 3.092 | 5.527 | 0 | 6 |
| 180 | 1 | maml_gru_sgd | matched_gru_finetune | 1.146 | 0.463 | 1.474 | 2 | 6 |
| 180 | 1 | fomaml_gru_sgd | matched_gru_finetune | 0.854 | 0.386 | 1.147 | 1 | 6 |
| 180 | 1 | reptile_gru_sgd | matched_gru_finetune | 0.757 | 0.021 | 1.473 | 2 | 6 |
| 180 | 2 | maml_gru_sgd | matched_gru_finetune | 0.466 | -0.240 | 0.784 | 4 | 6 |
| 180 | 2 | fomaml_gru_sgd | matched_gru_finetune | 0.285 | -0.206 | 0.543 | 4 | 6 |
| 180 | 2 | reptile_gru_sgd | matched_gru_finetune | 2.774 | 1.825 | 3.866 | 0 | 6 |
| 180 | 3 | maml_gru_sgd | matched_gru_finetune | 0.445 | -0.308 | 0.942 | 3 | 6 |
| 180 | 3 | fomaml_gru_sgd | matched_gru_finetune | 0.325 | -0.240 | 0.819 | 2 | 6 |
| 180 | 3 | reptile_gru_sgd | matched_gru_finetune | 4.184 | 3.194 | 5.676 | 0 | 6 |

### Second-order versus first-order gradients

Lower macro MAE occurred in 3/9 prespecified cells; 1 intervals lie entirely below zero. Negative differences favor the listed method. Intervals are unadjusted for multiple comparisons and condition on the selected six target cities.

| budget | horizon | method | reference | delta_mae | delta_mae_ci_low | delta_mae_ci_high | cities_improved | n_cities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | maml_gru_sgd | fomaml_gru_sgd | 0.013 | -0.225 | 0.155 | 4 | 6 |
| 30 | 2 | maml_gru_sgd | fomaml_gru_sgd | 0.007 | -0.266 | 0.134 | 4 | 6 |
| 30 | 3 | maml_gru_sgd | fomaml_gru_sgd | -0.058 | -0.330 | 0.102 | 4 | 6 |
| 90 | 1 | maml_gru_sgd | fomaml_gru_sgd | 0.081 | -0.122 | 0.158 | 2 | 6 |
| 90 | 2 | maml_gru_sgd | fomaml_gru_sgd | -0.012 | -0.239 | 0.069 | 3 | 6 |
| 90 | 3 | maml_gru_sgd | fomaml_gru_sgd | -0.133 | -0.370 | -0.028 | 5 | 6 |
| 180 | 1 | maml_gru_sgd | fomaml_gru_sgd | 0.292 | -0.029 | 0.442 | 3 | 6 |
| 180 | 2 | maml_gru_sgd | fomaml_gru_sgd | 0.181 | -0.129 | 0.311 | 3 | 6 |
| 180 | 3 | maml_gru_sgd | fomaml_gru_sgd | 0.120 | -0.186 | 0.264 | 4 | 6 |

### MAML versus Reptile

Lower macro MAE occurred in 6/9 prespecified cells; 6 intervals lie entirely below zero. Negative differences favor the listed method. Intervals are unadjusted for multiple comparisons and condition on the selected six target cities.

| budget | horizon | method | reference | delta_mae | delta_mae_ci_low | delta_mae_ci_high | cities_improved | n_cities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | maml_gru_sgd | reptile_gru_sgd | 1.669 | 0.162 | 2.571 | 3 | 6 |
| 30 | 2 | maml_gru_sgd | reptile_gru_sgd | -1.017 | -2.552 | -0.097 | 4 | 6 |
| 30 | 3 | maml_gru_sgd | reptile_gru_sgd | -2.390 | -4.191 | -1.427 | 5 | 6 |
| 90 | 1 | maml_gru_sgd | reptile_gru_sgd | 0.279 | -0.838 | 1.051 | 3 | 6 |
| 90 | 2 | maml_gru_sgd | reptile_gru_sgd | -2.382 | -3.587 | -1.563 | 5 | 6 |
| 90 | 3 | maml_gru_sgd | reptile_gru_sgd | -3.826 | -5.178 | -3.084 | 6 | 6 |
| 180 | 1 | maml_gru_sgd | reptile_gru_sgd | 0.389 | -0.721 | 1.181 | 3 | 6 |
| 180 | 2 | maml_gru_sgd | reptile_gru_sgd | -2.308 | -3.521 | -1.547 | 5 | 6 |
| 180 | 3 | maml_gru_sgd | reptile_gru_sgd | -3.739 | -5.080 | -3.061 | 6 | 6 |

### Changing the final adaptation optimizer

Lower macro MAE occurred in 0/27 prespecified cells; 0 intervals lie entirely below zero. Negative differences favor the listed method. Intervals are unadjusted for multiple comparisons and condition on the selected six target cities.

| budget | horizon | method | reference | delta_mae | delta_mae_ci_low | delta_mae_ci_high | cities_improved | n_cities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | maml_gru_adam | maml_gru_sgd | 3.441 | 2.429 | 4.212 | 0 | 6 |
| 30 | 1 | fomaml_gru_adam | fomaml_gru_sgd | 2.974 | 2.050 | 3.653 | 1 | 6 |
| 30 | 1 | reptile_gru_adam | reptile_gru_sgd | 1.715 | 1.286 | 2.069 | 0 | 6 |
| 30 | 2 | maml_gru_adam | maml_gru_sgd | 3.465 | 2.285 | 4.443 | 0 | 6 |
| 30 | 2 | fomaml_gru_adam | fomaml_gru_sgd | 2.956 | 1.823 | 3.723 | 0 | 6 |
| 30 | 2 | reptile_gru_adam | reptile_gru_sgd | 1.489 | 0.967 | 1.955 | 0 | 6 |
| 30 | 3 | maml_gru_adam | maml_gru_sgd | 3.342 | 2.118 | 4.493 | 0 | 6 |
| 30 | 3 | fomaml_gru_adam | fomaml_gru_sgd | 2.871 | 1.756 | 3.795 | 0 | 6 |
| 30 | 3 | reptile_gru_adam | reptile_gru_sgd | 1.227 | 0.745 | 1.748 | 1 | 6 |
| 90 | 1 | maml_gru_adam | maml_gru_sgd | 2.041 | 1.064 | 2.625 | 1 | 6 |
| 90 | 1 | fomaml_gru_adam | fomaml_gru_sgd | 2.094 | 1.015 | 2.752 | 1 | 6 |
| 90 | 1 | reptile_gru_adam | reptile_gru_sgd | 0.660 | 0.272 | 0.924 | 1 | 6 |
| 90 | 2 | maml_gru_adam | maml_gru_sgd | 2.019 | 1.057 | 2.625 | 1 | 6 |
| 90 | 2 | fomaml_gru_adam | fomaml_gru_sgd | 2.139 | 1.076 | 2.769 | 1 | 6 |
| 90 | 2 | reptile_gru_adam | reptile_gru_sgd | 0.633 | 0.208 | 0.913 | 0 | 6 |
| 90 | 3 | maml_gru_adam | maml_gru_sgd | 2.202 | 1.225 | 2.904 | 1 | 6 |
| 90 | 3 | fomaml_gru_adam | fomaml_gru_sgd | 2.256 | 1.158 | 3.040 | 1 | 6 |
| 90 | 3 | reptile_gru_adam | reptile_gru_sgd | 0.494 | -0.071 | 0.852 | 2 | 6 |
| 180 | 1 | maml_gru_adam | maml_gru_sgd | 0.981 | 0.584 | 1.396 | 2 | 6 |
| 180 | 1 | fomaml_gru_adam | fomaml_gru_sgd | 1.121 | 0.615 | 1.544 | 2 | 6 |
| 180 | 1 | reptile_gru_adam | reptile_gru_sgd | 0.414 | 0.190 | 0.725 | 1 | 6 |
| 180 | 2 | maml_gru_adam | maml_gru_sgd | 1.145 | 0.725 | 1.612 | 2 | 6 |
| 180 | 2 | fomaml_gru_adam | fomaml_gru_sgd | 1.253 | 0.683 | 1.738 | 2 | 6 |
| 180 | 2 | reptile_gru_adam | reptile_gru_sgd | 0.162 | -0.061 | 0.520 | 4 | 6 |
| 180 | 3 | maml_gru_adam | maml_gru_sgd | 1.232 | 0.782 | 1.710 | 1 | 6 |
| 180 | 3 | fomaml_gru_adam | fomaml_gru_sgd | 1.380 | 0.741 | 1.889 | 1 | 6 |
| 180 | 3 | reptile_gru_adam | reptile_gru_sgd | 0.117 | -0.135 | 0.445 | 1 | 6 |

### Six versus the full donor pool

Lower macro MAE occurred in 9/27 prespecified cells; 3 intervals lie entirely below zero. Negative differences favor the listed method. Intervals are unadjusted for multiple comparisons and condition on the selected six target cities.

| budget | horizon | method | reference | delta_mae | delta_mae_ci_low | delta_mae_ci_high | cities_improved | n_cities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.549 | 0.226 | 0.743 | 0 | 6 |
| 30 | 1 | diversity6_finetune | matched_gru_finetune | 0.339 | 0.036 | 0.547 | 2 | 6 |
| 30 | 1 | diversity6_maml_sgd | maml_gru_sgd | -0.595 | -0.858 | -0.382 | 4 | 6 |
| 30 | 2 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.340 | -0.070 | 0.606 | 1 | 6 |
| 30 | 2 | diversity6_finetune | matched_gru_finetune | 0.208 | -0.180 | 0.482 | 1 | 6 |
| 30 | 2 | diversity6_maml_sgd | maml_gru_sgd | -0.697 | -0.960 | -0.454 | 4 | 6 |
| 30 | 3 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.476 | -0.106 | 0.938 | 1 | 6 |
| 30 | 3 | diversity6_finetune | matched_gru_finetune | 0.330 | -0.181 | 0.776 | 1 | 6 |
| 30 | 3 | diversity6_maml_sgd | maml_gru_sgd | -0.779 | -1.058 | -0.517 | 5 | 6 |
| 90 | 1 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.549 | 0.226 | 0.743 | 0 | 6 |
| 90 | 1 | diversity6_finetune | matched_gru_finetune | 0.377 | 0.083 | 0.571 | 0 | 6 |
| 90 | 1 | diversity6_maml_sgd | maml_gru_sgd | -0.079 | -0.303 | 0.083 | 4 | 6 |
| 90 | 2 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.340 | -0.070 | 0.606 | 1 | 6 |
| 90 | 2 | diversity6_finetune | matched_gru_finetune | 0.236 | -0.137 | 0.475 | 1 | 6 |
| 90 | 2 | diversity6_maml_sgd | maml_gru_sgd | -0.172 | -0.402 | 0.026 | 4 | 6 |
| 90 | 3 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.476 | -0.106 | 0.938 | 1 | 6 |
| 90 | 3 | diversity6_finetune | matched_gru_finetune | 0.332 | -0.195 | 0.743 | 1 | 6 |
| 90 | 3 | diversity6_maml_sgd | maml_gru_sgd | -0.114 | -0.346 | 0.100 | 4 | 6 |
| 180 | 1 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.549 | 0.226 | 0.743 | 0 | 6 |
| 180 | 1 | diversity6_finetune | matched_gru_finetune | 0.424 | 0.122 | 0.623 | 0 | 6 |
| 180 | 1 | diversity6_maml_sgd | maml_gru_sgd | -0.003 | -0.265 | 0.192 | 4 | 6 |
| 180 | 2 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.340 | -0.070 | 0.606 | 1 | 6 |
| 180 | 2 | diversity6_finetune | matched_gru_finetune | 0.268 | -0.126 | 0.520 | 1 | 6 |
| 180 | 2 | diversity6_maml_sgd | maml_gru_sgd | -0.061 | -0.292 | 0.172 | 4 | 6 |
| 180 | 3 | diversity6_pooled_zero | matched_gru_pooled_zero | 0.476 | -0.106 | 0.938 | 1 | 6 |
| 180 | 3 | diversity6_finetune | matched_gru_finetune | 0.382 | -0.138 | 0.817 | 1 | 6 |
| 180 | 3 | diversity6_maml_sgd | maml_gru_sgd | -0.028 | -0.278 | 0.233 | 4 | 6 |

The optimizer comparison includes donor-selected schedules, so its effect includes optimizer and schedule together. The six-city donor ablation also changes the amount of unique source data; matched update counts do not isolate task diversity alone. Neither comparison proves a physical pollution mechanism.

## What can be concluded

The registered hypothesis is supported only where cross-city methods improve on local training and a useful simple forecast at the stated horizon and fitting-label budget. Meta-learning adds value only where its matched comparisons show an advantage. Results may be mixed or negative. Model count and GPU time are not substitutes for these comparisons.

The source/identity/coverage limits, simulated fitting-label scarcity, recency/season confounding, frozen-2023 model aging, six-city convenience sampling, finite optimization budgets and unverified publication latency remain. Broad hyperparameter search can overfit donor validation and does not establish global optimality.

Further work: predeclare truly new target cities and fresh rolling-origin refits; match source volume in donor-diversity experiments; expand independent task diversity; compare longer histories with a common missing-input policy; validate station history and issue-time availability; test high-pollution calibration; and reserve final external confirmation before deployment.
