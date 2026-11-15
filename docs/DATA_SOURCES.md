# Official source and archived study inputs

The official CPCB portal is https://airquality.cpcb.gov.in/ccr/#/repository/aqi . Select the daily AQI repository, station and year, complete the portal's current access flow and export the data. Its current interface or available history may change. The study target is reported station-level overall daily AQI; it is not PM2.5 concentration, a city average or an hourly target.

The public project release hosts `CPCB_AQI_Source_Release_v1.zip`, SHA256 `014fa6ab88854a9649e4d903483d516be5b4b5ecf482ecd497e291cdf1a89811`. It preserves the bytes collected on 31 October–1 November and reconciled with the captured station listings on 14 November 2026. The 626-station captured current roster and 2017–2025 acquisition scope yielded 3,820 workbooks and 769,129 station-date readings. This is a versioned research snapshot, not a complete national monitoring history or an official CPCB release.

The archive contains all raw XLSX files, station metadata, file register, source URLs, SHA256 sums, captured listing evidence and acquisition/coverage scripts. `python reproduce.py fetch` verifies and extracts it from the public full-project archive. `python reproduce.py prepare` reparses the originals using the exact frozen extraction pipeline and checks the resulting core scientific tables against `data/frozen/`.

Both original workbook layouts are supported. Daily rows are distinct from categorical/summary rows. All raw values are retained. The normalized observation table preserves unmatched identities and range flags; the scientific cohort requires verified identities and stated availability. No clipping or input/truth imputation is used.

The committed `data/frozen/` tables are derivatives made by this project. `INPUT_MANIFEST.json` records original compressed bytes and, for gzip tables, the decompressed size and hash. These tables support training without repeated XLSX I/O; they do not replace the source register or identity limitations.

No license over CPCB data is granted by this repository, and no explicit CPCB redistribution license has been established by this study. Attribution, portal access and release availability do not imply an open-data license. Consult CPCB's applicable terms for any downstream redistribution or other intended use. Code, original research text and third-party source material have separate provenance; no blanket license is assigned here.

The recorded portal backend routes were observed during acquisition, not an asserted supported API contract. The release includes the acquisition scripts for transparency and offline audit. Fresh live collection is optional and will not necessarily recreate an identical historical snapshot. Do not overwrite the frozen source with new downloads.
