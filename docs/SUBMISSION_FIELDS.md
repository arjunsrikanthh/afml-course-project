# AFML course project submission

**Cross-City Transfer Learning for AQI Forecasting in Data-Scarce Indian Regions**

**Team Toxic · Team 23**

## Public GitHub repository

[arjunsrikanthh/afml-course-project](https://github.com/arjunsrikanthh/afml-course-project) contains the data preparation, model training, evaluation, and reproduction code. Its README links the executed notebooks, frozen inputs, follow-up experiments, and audit evidence.

## Dataset

[CPCB Central Control Room AQI Repository](https://airquality.cpcb.gov.in/ccr/#/repository/aqi) is the official public source. Select **4 PM Daily AQI** to access the original annual station-level workbooks. [Dataset details and reproducibility snapshot](DATASET_LINKS.md).

## IEEE paper

[IEEE paper PDF](../paper/AFML_IEEE_Paper.pdf) is the IEEE conference-format paper. [Editable LaTeX source](../paper/AFML_IEEE_LaTeX_Source.zip) contains its editable source and figures.

## Presentation

[Editable PowerPoint](../slides/AFML_Final_Presentation.pptx) is the editable PowerPoint. [Presentation PDF](../slides/AFML_Final_Presentation.pdf) is its PDF preview.

## Team contributions

| Team member | SRN | Project work |
|---|---|---|
| Ankit Umadi | PES1UG24AM422 | MAML, FOMAML and Reptile implementation, meta-learning experiments, and objective and task controls. |
| Arjun Srikanth | PES1UG24AM920 | Literature review and experimental design; CPCB data acquisition and preparation; LSTM, MLP, TCN and Transformer experiments; Ridge regression, Random Forest and histogram gradient boosting comparisons; persistence, seven-day mean and previous-week baselines; evaluation and verification. |
| Jayanth K | PES1UG24AM123 | GRU implementation, pooled and local GRU training, fine-tuning, and GRU adaptation diagnostics. |
