# IEEE-format research paper

[Compiled six-page paper](AFML_IEEE_Paper.pdf) · [Editable LaTeX archive](AFML_IEEE_LaTeX_Source.zip) · [LaTeX source](AFML_IEEE_Paper.tex).

Authors: Arjun Srikanth, Ankit Umadi and Jayanth K. The paper makes no claim of publication, acceptance, nationwide generalization or deployment readiness. No individual contributions are inferred.

The authentic IEEEtran 1.8b conference class was obtained from [CTAN](https://ctan.org/pkg/ieeetran) on 15 November 2026. Its copyright/license notices are retained in `IEEEtran.cls`. The supplied vector figures and results table allow compilation without installing the machine-learning environment:

```sh
cd paper
tectonic AFML_IEEE_Paper.tex
# Alternative with a conventional TeX installation:
pdflatex AFML_IEEE_Paper.tex
pdflatex AFML_IEEE_Paper.tex
```

To regenerate the vector figures, results table and numerical evidence from the repository's retained results:

```sh
# Run from the repository root, in the project Python environment.
python paper/build_figures.py --results-root results
```

This reads the metric and paired-comparison CSVs under `results/reference/` and `results/expanded/`; it neither refits models nor selects deployment winners. Main results use the 2025 test; 2024 remains a secondary evaluation year rather than a tuning split.

The source/cohort contract is in `data/frozen/`, the observed run protocols and independent audit receipts are in `results/reference/` and `results/expanded/`, and the strict proposal comparisons and sensitivity reports are in `results/synthesis/`. `numeric_evidence.json` records the values used for the principal table and matched-meta discussion. Exact scientific implementations are in `scripts/`; detailed original forecasts/fitting journals are in the public release archive.

`verification.json` records successful source/PDF checks. All six final pages were visually inspected, including balanced final columns. The final PDF SHA256 is `49c75015780630685b4d59cbd72f925a384c6d4f7c21569e214a29458fb4ac05`.
