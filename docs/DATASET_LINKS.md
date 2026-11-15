# Dataset links for the submission form

**Official public source used in the final study:** [CPCB Central Control Room AQI Repository](https://airquality.cpcb.gov.in/ccr/#/repository/aqi).

Select **4 PM Daily AQI**, then the state, city, station and year. The source exports annual station-level XLSX workbooks. These are overall AQI observations, not PM2.5 concentrations. The website may require its normal browser verification or CAPTCHA.

**Official explanation of the index:** [CPCB National Air Quality Index](https://cpcb.nic.in/National-Air-Quality-Index/).

The final study uses a frozen acquisition release containing 3,820 workbooks and 769,129 observed station-dates from the declared 2017–2025 scope. Modeling uses 2020–2025, 23 donor cities and six held-out target cities. Missing data and current-roster coverage prevent a claim of complete national historical coverage.

**Reproducibility mirror:** [Versioned project release](https://github.com/arjunsrikanthh/afml-course-project/releases/tag/study-v1.0.0). This project-hosted snapshot preserves source bytes and checksums. It is separate from the official publisher. No blanket license for CPCB data is asserted.

The final evaluation does not use the older Kaggle Air Quality Data in India dataset proposed in the original one-pager, or the separate pollutant/weather collection. Those exploratory sources should not replace the official source link above in the form.

The dataset requirement therefore has public datasets involved; a no-public-dataset justification is unnecessary. Provenance comes from the preserved acquisition receipts and workbook checks. Live portal requests during this packaging session timed out, so this document does not claim a fresh successful live download on 15 November.
