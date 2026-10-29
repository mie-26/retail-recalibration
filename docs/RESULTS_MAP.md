# Core reproduction coverage

| Paper evidence | Input after extraction | Executed verification |
| --- | --- | --- |
| Main Table 1 / design | `protocol/`, `cold_fit/reference/`, Italian cohort/support | Recorded design; not a new cohort selection |
| Main Table 2 / Figure 2 | `score_data/enhanced_daily.parquet`, `supplementary_data/Data1_*` | Core daily score aggregation, primary intervals and 12-row summary |
| Main Figure 3 | `figure_source_data/figure3_*`, `supplementary_data/Data2_*` | Plot from stored full diagnostic summaries; non-core daily diagnostics excluded |
| Main Figure 4 / Table 3 | Italian `selected/daily.parquet`, `analysis/` | Selected E2 group/series arms and Raw reaggregated; all archived time-block intervals repeated |
| Controlled E1/E2/E3 / S7–S9 | `controlled/` | Full controlled saved-score and reference-interval audit |
| Italian feedback and E3 / S10–S11 | Italian selected streams and `e3/` | Daily-score effects, intervals and identity check |
| Italian fixed-grid reversals | Italian `analysis/fixed_grid_scores.csv` | 144 interaction arithmetic checks; candidate daily arrays excluded |
| Remaining supplementary tables and figures | `supplementary_data/`, `robustness/`, `figure_source_data/` | Final tables retained; plots regenerated. Complete robust-grid/stratum reaggregation excluded |
| Main Figure 1 | `figure_assets/` | PPT-authored schematic and verified exports |

Four main figures and eleven supplementary figures are retained. Adverse estimates and limitations are not removed. All retained tables describe the same submission, not additional independent experiments.
