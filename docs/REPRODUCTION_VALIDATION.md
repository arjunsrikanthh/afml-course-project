# Public-package reproduction validation

Validated on 15 November 2026, Python 3.12.15, macOS. This records the portable package checks. The original full experiments remain the archived observed runs; validation does not claim a new full-training execution.

A separate checkout and an isolated virtual environment were created outside the original project. `pip install -r requirements.txt` succeeded in the new environment. The actual resolved packages are recorded in `evidence/public-package-environment.txt`. The full original execution inventory is separately preserved in `evidence/original-environment.txt`.

| Command/check | Observed result |
| --- | --- |
| `python reproduce.py check-inputs` | All 17 cached input hashes verified; 23 donor cities, six targets, 769,129 observations reconstructed |
| `python reproduce.py smoke` | Compact artificial run: all 23 methods, 2,484 predictions, 120 fitting records; expanded: all 33 methods, 990 predictions, 77 fitting records; deterministic checkpoint resume checks passed |
| Synthetic meta-gradient check | Full versus first-order MAML gradient difference L2 0.123727; full body gradient nonzero; native/manual GRU forward agreement within 7.46e-8; parameter updates present |
| `python reproduce.py prepare` | All 3,820 original XLSX files reparsed; observation/window/cohort/eligibility/coverage tables exactly match the frozen scientific tables |
| `python reproduce.py audit` | Reference: 9/9 PASS, expanded: 12/12 PASS, zero critical failures; no model engine imported by verifiers |
| `python reproduce.py synthesis` | Final synthesis executed successfully; shared example contexts, audited artifact hashes and independently recomputed saved MAE/RMSE/bias all verified |
| Executed notebook receipt checks | All four original notebook SHA256 values match their execution receipts; no historical output was regenerated or sanitized |
| Publication scan | No credential-pattern matches in repository text, textual notebook cells/outputs or archived project text; image base64 excluded from text matching |

All listed CLI checks also passed in a credential-disabled anonymous clone of public main (`6b34ee845b44be08b39e386d4d46cdf172f3b432`) using the new isolated environment. The entire 224,820,338-byte archive was downloaded anonymously from the public release and its SHA256 and ZIP CRC matched. Source preparation, both independent audits and the final synthesis then passed using only that clone and download. The portable paper figure script also reproduced the numerical evidence and results table. Machine-readable evidence is in `evidence/public-clean-clone-validation.json`; public API and paper-download checks are in `evidence/public-access-verification.json`.

Full reference/expanded retraining was not repeated during packaging. The scientific modules are byte-identical to the frozen modules used in the completed observed runs. Original backend receipts distinguish Kaggle EDA, Mac CPU reference, Apple MPS expanded study and local CPU synthesis. New hardware, libraries and parallel kernels may change fitted weights and resulting forecasts; the archived observed outputs are the reproducible reference record.

Independent artifact audits verify consistency, source hashes, chronology and saved metrics. They do not prove every declared optimizer update physically executed, establish scientific validity of each original measurement, validate real-time publication latency or certify external generalization.
