# Data sources and access

The three core-source archives distribute analysis summaries and figure inputs, without their complete raw datasets or trained model banks. Obtain those raw data from the providers and comply with their terms. The later Italian extension includes its publicly licensed source CSV with attribution and the provider's CC BY-NC 3.0 notice.

| Source | Official reference or access | Study role |
| --- | --- | --- |
| M5 | [M5 uncertainty competition](https://doi.org/10.1016/j.ijforecast.2021.10.009); [competition data](https://www.kaggle.com/competitions/m5-forecasting-uncertainty/data) | Three previously used development stores, 3,000 sampled item-store series |
| Online Retail II | [UCI repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii), DOI [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D) | 1,692 UK positive-sales product series; development source |
| Dominick’s | [Kilts Center data page](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) | 2,000 item-store series in four categories; reserved-period source with a documented startup amendment |

Dominick’s source data are provided for academic research. Acknowledge the Kilts Center for Marketing at the University of Chicago Booth School of Business. Missing records, invalid observations and product exits must not silently become zero sales. The observed-support scope is reported in the manuscript and Supplementary Data 4.

The `score_data/` files contain method-level, per-origin aggregate scoring records; they are not a complete SKU-level sales panel. The supplemental CSVs contain scores, contrasts, uncertainty summaries, support diagnostics and computation ledgers. Field inventories and row counts are in `DATA_DICTIONARY.json`.

No provider licence is replaced by a repository licence. Code status is described in `LICENSE_STATUS.txt`.


Italian grocery confirmation: Mancuso, Piccialli and Sudoso (2021), https://data.mendeley.com/datasets/s8dgbs3rng/1 ; doi:10.17632/s8dgbs3rng.1. The paper-associated record https://data.mendeley.com/datasets/njdkntcpc9/1 supplied byte-identical CSV data, not a second source. CC BY-NC 3.0 provider notice is preserved in the Italian archive. 118 pasta SKU series, one store, 2014–2018; 28 absent calendar days remain missing. This licence applies to provider data and does not grant a new licence to the entire repository.
