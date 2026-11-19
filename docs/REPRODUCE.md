# Reproduce the full AQI study

## Run the project notebook

Use Python 3.12 in a virtual environment.

```sh
git clone https://github.com/arjunsrikanthh/afml-course-project.git
cd afml-course-project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name 'Python 3'
```

Open [AFML_Demo.ipynb](../AFML_Demo.ipynb) in VS Code with its Jupyter extension, or in JupyterLab. Select the environment above and choose **Run All**. The example uses the committed inputs and needs only CPU.

## Verify the archived study

Run these commands from the repository root with the same environment active:

```sh
python reproduce.py check-inputs
python reproduce.py smoke
python reproduce.py fetch
python reproduce.py prepare
python reproduce.py audit
python reproduce.py synthesis
```

`fetch` downloads the public 214 MB release archive, checks its fixed SHA256, validates ZIP members and extracts the original CPCB source snapshot. No Kaggle login, API key or private dataset is needed. `prepare` independently reparses all 3,820 preserved XLSX workbooks and compares five scientific tables to the committed cached inputs. `audit` recalculates the saved prediction metrics and checks source/cohort/chronology/budgets; it does not fit models. `synthesis` executes all final notebook computations from those forecasts. Outputs go to ignored `runs/` and `artifacts/` directories.

## Retrain the full benchmarks

Full final-benchmark retraining takes hours and benefits from CUDA or Apple MPS. Run `fetch` first if you also want to audit the new results against the original source workbooks.

```sh
python reproduce.py train --study baseline --device cpu
python reproduce.py train --study final --device auto
python reproduce.py audit --study baseline --root runs/baseline
python reproduce.py audit --study final --root runs/final
```

These commands use the committed, hash-checked inputs and exact frozen scientific modules. They preserve all seeds, label budgets and donor-only searches. Select `cpu`, `mps` or `cuda` explicitly if required. A completed run resumes from saved state; changing the config requires a new `--output` directory. Backend differences can change numerical results, so compare recomputed results with the preserved observed runs without claiming bitwise identity.

The numbered [runnable notebooks](../notebooks/runnable/) provide the same portable entry points. The [executed notebooks](../notebooks/executed/) retain original paths and cell outputs. Their [execution receipts](../evidence/execution/) include hashes of the original notebooks.

## Data and experimental protocol

The authoritative data source is the [official CPCB daily AQI repository](https://airquality.cpcb.gov.in/ccr/#/repository/aqi). Our separately hosted [source snapshot](https://github.com/arjunsrikanthh/afml-course-project/releases/download/study-v1.0.0/CPCB_AQI_Source_Release_v1.zip) is a dated reproducibility archive, not a CPCB-published dataset or a stable public API. Source/reuse terms and access instructions are in [docs/DATA_SOURCES.md](DATA_SOURCES.md).

The snapshot contains 3,820 original annual workbooks and 769,129 station-day readings from the declared 2017–2025 acquisition scope. A current 626-station roster is not a census of all historical monitoring. Seven unmatched IDs and four readings above 500 remain preserved and flagged.

The study uses 23 donor cities and six held-out target cities: Amaravati, Guwahati, Ahmedabad, Pithampur, Ludhiana and Udaipur. One representative station is used per selected city code; adjacent source stations within 25 km of any known target-city station are excluded. Eligibility requires at least 80% annual availability in each year from 2020 through 2025, so this is a retrospective availability-selected cohort.

Models use seven consecutive observed daily AQI values, issue-calendar features and a direct horizon indicator. All fitting ends in 2023; donor-only model selection uses 2023, 2024 is a secondary evaluation period, and 2025 is the main evaluation period. Target budgets are the latest 30, 90 or 180 eligible label dates by the end of 2023, with three horizon pairs per date and additional preceding input context. Seeds are 17, 43 and 71. Inputs and truth are never imputed.

Scarcity is simulated fitting-label scarcity with recent rolling observations, not naturally absent sensors. Daily one-to-three-day labels approximate the original 24–72-hour scope; operational publication latency was not verified. Results concern six station series and do not establish nationwide generalization or deployment performance. Bootstrap intervals condition on the selected cities and are unadjusted for multiple comparisons. See [data/frozen/protocol.json](../data/frozen/protocol.json) and the preserved baseline and final run protocols for the full contract.

## Reproduction commands

| Command | What it does |
| --- | --- |
| `check-inputs` | Verifies the frozen input files and reconstructs the donor and target splits. |
| `smoke` | Runs small synthetic tests of the baseline and final pipelines. |
| `fetch` and `prepare` | Download the archived project and parse the original CPCB workbooks. |
| `audit` and `synthesis` | Check the saved forecasts and recompute comparisons from those forecasts. |
| `train --study baseline` or `train --study final` | Train a complete benchmark from the frozen inputs. |
| `python reproduce_followup.py --help` and `python reproduce_extension.py --help` | Show the commands for the follow-up experiments. |

See the [record of reproduction checks](REPRODUCTION_VALIDATION.md) and the [archived output index](../results/).
