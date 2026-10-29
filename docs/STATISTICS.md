# Numerical and evidence conventions

- H and U denote days for M5 and Online Retail II, and weeks for Dominick's.
- Joint loss equally weights q90 and q95 after normalization by the fixed historical series scale. Scores, not predicted values, are averaged across optimization seeds.
- Relative loss change is `100 * (candidate / reference - 1)`. Absolute coverage is a fraction; coverage contrasts are percentage points. Strict coverage uses `Y < q`, weak coverage uses `Y <= q`; their gap records ties.
- Complete origin slices are resampled together using circular time blocks, preserving paired methods and same-origin cross-series dependence. The reference block is 2H, with 3,000 draws and random seed 20261021. Additional block lengths remain in the interval tables. Intervals are descriptive and conditional on these sources and observation masks, not independent enterprise inference.
- Simulation intervals use paired resampling of independent paths, not retail time blocks. Each of four specified worlds has 100 paths. Simulation misspecification may itself create calibration benefits.
- Nonobserved targets remain missing. Observed-count weighting is preserved; zero support is not zero demand. Strata and stores do not create independent enterprises.
- Selected calibration parameters use recorded legal historical warmup scores. All evaluation-grid results are reported; no new selection is performed by the reproduction command.
- Original core comparisons and later diagnostic controls retain their separate status. Dominick's startup amendment occurred after raw decoding and before the reported external fitting/scoring. The unmodified records are included in `protocol/`; a local hash or timestamp is not independent preregistration.
- Protocol code hashes refer to the implementation when the experiment ran. Portable extraction and output-path changes are documented in `protocol/publication_transformations.json`; the package manifest identifies delivered files. Prior workspaces need not be recreated for the score route.
- Fit-time sums, logical fitted-model counts, replay-kernel time, and simulated cost proxies are distinct quantities. The package neither promises original wall-clock times on another machine nor demonstrates realized operational profit.
