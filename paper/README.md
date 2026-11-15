# IEEE conference paper

`AFML_IEEE_Paper.pdf` is the final IEEE conference-format paper. `AFML_IEEE_Paper.tex` is the editable LaTeX source. The original benchmark is published in [study-v1.0.0](https://github.com/arjunsrikanthh/afml-course-project/releases/tag/study-v1.0.0). Section VI reports the post-hoc tests on the same 2024 and 2025 evaluation windows, including the five-architecture learning-rate comparison and matched MAML objectives. [Extension code and evidence](https://github.com/arjunsrikanthh/afml-course-project/tree/extension-v1.0.0/followup/extension)

The class is the authentic IEEEtran 1.8b conference class, downloaded from CTAN on 15 November 2026. Source: https://ctan.org/pkg/ieeetran and https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls . Its copyright and license notices are retained in `IEEEtran.cls`.

Compile from this directory with `tectonic AFML_IEEE_Paper.tex`, or run `pdflatex AFML_IEEE_Paper.tex` twice. The vector figures and tables are included. To regenerate them from the bundled CSV files, install Python, pandas and Matplotlib and run the scripts below before recompiling.

- `python build_figures.py` regenerates the benchmark figures, table and numeric evidence. Use `--results-root` to read an alternative verified results directory.
- `python build_followup_figure.py` regenerates the deep-GRU adaptation plot from the included `figure_data/adaptation_contrasts.csv`.
- `python build_extension_tables.py` regenerates the five-architecture and MAML objective tables.

The paper reports the six-city experiment, distinguishes that cohort from the larger acquired archive, and preserves the negative fine-tuning and meta-learning results. It makes no publication, acceptance, national-generalization or deployment claim.

## Owning evidence

Paths below are relative to the AFML project root.

- Source inventory and cohort: `results/final_study_20261114/mac_research/afml_research/data/data_summary.json`, `protocol.json`, `cohort.csv` and `evaluation/eligibility_counts.csv`.
- Expanded metrics and paired intervals: `results/final_study_20261114/mac_research/afml_research/evaluation/macro_metrics.csv` and `paired_comparisons.csv`.
- Compact metrics: `results/final_study_20261114/mac_models/afml_final/evaluation/macro_metrics.csv`.
- Strict proposal criterion: `results/final_study_20261114/local_final_synthesis/original_proposal_success_conditions.csv` (45 compact and 54 expanded cells).
- Temporal sensitivity: `results/final_study_20261114/local_final_synthesis/paired_2024_2025_comparison.csv`.
- Exact implementation: `results/final_study_20261114/mac_research/afml_research_source/scripts/final_model_engine.py`, `research_model_engine.py` and `final_evaluation.py`.
- Audit: `results/final_study_20261114/independent_reference_audit/verification.json` and `independent_research_audit/verification.json`. These establish artifact consistency and chronology, with the audit limits explicitly stated in the paper.
- Follow-up: `results/explanatory_investigation_20261115/`, including `adaptation_contrasts.csv`, `meta_contrasts.csv`, `contrasts.csv`, `correction_decomposition.csv`, `donor_diversity/descriptive_mae_summary.csv`, and `independent_audit/`.
- Extension: the audited endpoint and interval CSVs under `results/cross_architecture_followup_20261115/` and the objective-schedule metrics and intervals under `results/explanatory_extension_20261115/maml_objective/`. The four CSVs used in the paper are also bundled in `figure_data/`.

`numeric_evidence.json` contains the metric records used for the paper's principal table and matched-meta discussion. Main comparisons use the 2025 test; 2024 is a separate evaluation year, never validation or tuning.

## Reference checks

Primary bibliographic sources were inspected on 15 November 2026:

- CPCB portal and official AQI report: https://airquality.cpcb.gov.in/ccr/ and https://airquality.cpcb.gov.in/ccr_docs/FINAL-REPORT_AQI_.pdf . The source snapshot is established by retained acquisition evidence; this paper build did not reacquire the archive or certify current file listings. Current repository route: https://airquality.cpcb.gov.in/ccr/#/repository/aqi .
- RegionTrans: https://ijcai.org/proceedings/2019/262 (crowd-flow demonstration, not an Indian AQI result).
- PM10 transfer: https://www.nature.com/articles/s41598-025-86550-6 (publisher-indexed metadata and full-text search excerpts; direct full-text page redirected to an inaccessible identity endpoint).
- GRU: https://aclanthology.org/D14-1179/ .
- LSTM: https://www.bioinf.jku.at/publications/older/2604.pdf (author-institution-hosted paper; MIT publisher URL returned 403).
- TCN: https://arxiv.org/abs/1803.01271 .
- Transformer: https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html .
- MAML: https://proceedings.mlr.press/v70/finn17a.html .
- FOMAML/Reptile: https://arxiv.org/abs/1803.02999 .
- Weight anchoring: https://proceedings.mlr.press/v80/li18a.html .

No new literature claim of novelty is made. In-text references use normal IEEE numbering.
