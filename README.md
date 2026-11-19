# Cross-City Transfer Learning for AQI Forecasting in Data-Scarce Indian Regions

**Start here → [Open the project notebook](AFML_Demo.ipynb).**

It walks through the CPCB data, the city and year splits, the model code, a runnable AQI forecast, and the results of the full experiments. The notebook includes its outputs so you can also read it on GitHub.

| Deliverable | Open |
| --- | --- |
| IEEE report | [PDF](paper/AFML_IEEE_Paper.pdf) · [Editable LaTeX and figures](paper/AFML_IEEE_LaTeX_Source.zip) |
| Presentation | [PowerPoint](slides/AFML_Final_Presentation.pptx) · [PDF](slides/AFML_Final_Presentation.pdf) |
| Dataset | [Official CPCB repository](https://airquality.cpcb.gov.in/ccr/#/repository/aqi) · [Data provenance](docs/DATA_SOURCES.md) |
| Full experiments | [Run instructions](docs/REPRODUCE.md) · [Original executed notebooks](notebooks/executed/) · [Archived outputs](results/) |

## What's in the repository

| Location | Why it's here |
| --- | --- |
| `AFML_Demo.ipynb` | The entry point for the demo and for understanding the project. |
| `scripts/`, `reproduce.py` | The Python implementation and commands for the complete benchmark. |
| `configs/final.json` | The single configuration for the final 33-method benchmark. The earlier 23-method baseline settings are retained under `configs/archive/`. |
| `manifests/release-assets.json` | SHA256 and size checks for downloadable source/results archives; used by `fetch` and the follow-up replay tools. |
| `data/frozen/` | The checked, processed CPCB dataset, fixed cohort, forecast examples and data-quality evidence consumed by the models. |
| `results/` | Frozen baseline/final run protocols and combined comparison tables. Full forecasts, fitting logs and checkpoints are retained in the versioned release archive. |
| `notebooks/`, `evidence/` | Original executed notebooks, portable full-run notebooks, and records of where each run happened. |
| `followup/`, `reproduce_followup.py`, `reproduce_extension.py` | Follow-up experiments on learning rates, donor diversity and MAML objectives. |
| `paper/`, `slides/`, `docs/` | Submission files, sources for rebuilding the report, and instructions. |

**Team Toxic (23)**

| Team member | Contribution |
| --- | --- |
| Ankit Umadi (PES1UG24AM422) | MAML, FOMAML and Reptile implementation, meta-learning experiments, and objective and task controls. |
| Arjun Srikanth (PES1UG24AM920) | Research, data preparation, LSTM, MLP, TCN, Transformer, etc., and evaluation. |
| Jayanth K (PES1UG24AM123) | GRU implementation, pooled and local GRU training, fine-tuning, and GRU adaptation diagnostics. |
