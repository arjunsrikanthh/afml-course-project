# Cross-City Transfer Learning for AQI Forecasting in Data-Scarce Indian Regions

A reproducible undergraduate study of daily overall AQI from India's Central Pollution Control Board (CPCB). We compare target-only training, pooled pretraining, target fine-tuning and meta-learning under limited target fitting labels. The experiment retains negative results and uses strong simple forecasts as controls.

**Team Toxic (23)** · Arjun Srikanth (PES1UG24AM920), Ankit Umadi (PES1UG24AM422), Jayanth K (PES1UH24AM123).

Start with the [executed research synthesis](notebooks/executed/AFML_04_Final_Research_Synthesis_Executed.ipynb). The [IEEE-format paper](paper/) and [presentation](slides/) explain the study. The [public frozen release](https://github.com/arjunsrikanthh/afml-course-project/releases/tag/study-v1.0.0) preserves the original source bytes, forecasts, fitting journals, saved weights and independent audits.

The compact reference produced 1,876,662 forecasts across 23 methods on Mac CPU. The expanded comparison produced 2,692,602 forecasts across 33 methods on Apple GPU (MPS), including full second-order MAML, FOMAML and Reptile. Notebook 01 ran on Kaggle; the executed model and synthesis notebooks in this repository are the completed local runs. A private Kaggle queue is not required to reproduce them.

The central result is conditional: pretraining helps relative to matching local neural models, while target fine-tuning can damage a strong pooled forecast. Meta-learning has no universal advantage over ordinary matched transfer. The strict original fine-tuning success criterion held in 0/99 evaluated method/budget/horizon cells; matched meta-learning did not improve on ordinary matched GRU fine-tuning in the 27 main comparison cells. See the synthesis for exact definitions, estimates, uncertainty and failures.

The [explanatory follow-up](followup/) replays the saved GRUs under matched initializations and constrained updates. It also tests nested donor panels with matched label volume.


## Run from a fresh clone

Use Python 3.12 (the original run's Python version) and a virtual environment. CPU works for smoke checks, audits and synthesis. Full expanded retraining takes hours and benefits from CUDA or Apple MPS.

```sh
git clone https://github.com/arjunsrikanthh/afml-course-project.git
cd afml-course-project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name 'Python 3'
python reproduce.py check-inputs
python reproduce.py smoke
python reproduce.py fetch
python reproduce.py prepare
python reproduce.py audit
python reproduce.py synthesis
```

`fetch` downloads the public 214 MB release archive, checks its fixed SHA256, validates ZIP members and extracts the original CPCB source snapshot. No Kaggle login, API key or private dataset is needed. `prepare` independently reparses all 3,820 preserved XLSX workbooks and compares five scientific tables to the committed cached inputs. `audit` recalculates the saved prediction metrics and checks source/cohort/chronology/budgets; it does not fit models. `synthesis` executes all final notebook computations from those forecasts. Outputs go to ignored `runs/` and `artifacts/` directories.

To fit the full frozen experiments yourself:

```sh
python reproduce.py train --study reference --device cpu
python reproduce.py train --study expanded --device auto
python reproduce.py audit --study reference --root runs/reference
python reproduce.py audit --study expanded --root runs/expanded
```

These commands use the committed, hash-checked inputs and exact frozen scientific modules. They preserve all seeds, label budgets and donor-only searches. Select `cpu`, `mps` or `cuda` explicitly if required. A completed run resumes from saved state; changing the config requires a new `--output` directory. Backend differences can change numerical results, so compare recomputed results with the preserved observed runs without claiming bitwise identity.

The numbered [runnable notebooks](notebooks/runnable/) provide the same portable entry points. The [executed notebooks](notebooks/executed/) retain historical source paths and actual cell outputs; their receipts in [evidence/execution](evidence/execution/) verify the unmodified notebook hashes. Use runnable notebooks or the CLI when executing the project on another machine.

## Data and experimental protocol

The authoritative data source is the [official CPCB daily AQI repository](https://airquality.cpcb.gov.in/ccr/#/repository/aqi). Our separately hosted [source snapshot](https://github.com/arjunsrikanthh/afml-course-project/releases/download/study-v1.0.0/CPCB_AQI_Source_Release_v1.zip) is a dated reproducibility archive, not a CPCB-published dataset or a stable public API. Source/reuse terms and access instructions are in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

The snapshot contains 3,820 original annual workbooks and 769,129 station-day readings from the declared 2017–2025 acquisition scope. A current 626-station roster is not a census of all historical monitoring. Seven unmatched IDs and four readings above 500 remain preserved and flagged.

The study uses 23 donor cities and six held-out target cities: Amaravati, Guwahati, Ahmedabad, Pithampur, Ludhiana and Udaipur. One representative station is used per selected city code; adjacent source stations within 25 km of any known target-city station are excluded. Eligibility requires at least 80% annual availability in each year from 2020 through 2025, so this is a retrospective availability-selected cohort.

Models use seven consecutive observed daily AQI values, issue-calendar features and a direct horizon indicator. All fitting ends in 2023; donor-only model selection uses 2023, 2024 is a secondary evaluation period, and 2025 is the main evaluation period. Target budgets are the latest 30, 90 or 180 eligible label dates by the end of 2023, with three horizon pairs per date and additional preceding input context. Seeds are 17, 43 and 71. Inputs and truth are never imputed.

Scarcity is simulated fitting-label scarcity with recent rolling observations, not naturally absent sensors. Daily one-to-three-day labels approximate the original 24–72-hour scope; operational publication latency was not verified. Results concern six station series and do not establish nationwide generalization or deployment performance. Bootstrap intervals condition on the selected cities and are unadjusted for multiple comparisons. See [data/frozen/protocol.json](data/frozen/protocol.json) and both preserved run protocols for the full contract.

## Repository map

| Location | Contents |
| --- | --- |
| `scripts/` | Exact frozen extraction, model, evaluation and independent verification modules |
| `configs/` | Original model settings and public-release size/SHA256 manifest |
| `data/frozen/` | Hash-checked derived observations, windows, cohort and source integrity tables |
| `notebooks/executed/` | Four actual executed notebooks, preserved byte-for-byte |
| `notebooks/runnable/` | Portable local source, training and synthesis entry points |
| `results/` | Saved metrics, plots, comparisons, original protocols and audit summaries |
| `evidence/` | Execution receipts, design freeze, derivative verification and actual environment inventory |
| `paper/`, `slides/` | Submission artifacts and their editable sources |
| `reproduce.py` | Public download, verification, preparation, training, audit and synthesis commands |

The [reproduction validation record](docs/REPRODUCTION_VALIDATION.md) distinguishes checks performed for this public package from the original full training. There is no claim of course submission, TA acceptance or published peer review.
