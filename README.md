# Retail recalibration — core reproduction

Core reproducibility materials for **Benefits and risks of online recalibration after retraining in retail quantile forecasting** (submission materials, 29 October 2026).

This is a deliberately smaller companion to the full submitted `Supplementary_Code.zip`. It preserves core daily scores, scientific implementations, final result tables (including adverse results), and all figure inputs. It is not the complete model bank or full candidate-grid audit. No manuscript, author profile, correspondence details or historical manuscript bundles are included.

## Run

Use Python 3.12. From this directory:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

No provider downloads, GPU, model fitting or network access are needed for the default score/figure route. Allow 1 GB of free disk space for restored inputs and outputs. Each step stops on failure; logs and the execution report are in `reproduced/`. Do not upload that generated directory.

## What is recalculated

- The original three-source core score summaries, original-unit loss, primary contrasts and 3,000-draw reference-block intervals. Training-budget and feedback-control summaries are retained for plotting, without reaggregating their daily inputs.
- The controlled E1/E2/E3 comparisons, their reference loss intervals and rule-by-sharing contrasts, using all original controlled E1/E2/E3 daily streams.
- Italian selected group/series E2 arms, Raw, feedback reuse/isolation and E3: per-quantile summaries, seed/year results, interactions, paired intervals at all archived block lengths (5,000 resamples), nonoverlap checks, and the E3/bridge identity check.
- Fixed-grid reversals from their **stored summary scores**, preserving all 144 combinations and 26 negative interactions. Candidate daily aggregation is not repeated here.
- Links between final result tables and figure inputs; retained support and simulation summaries.
- Four main and eleven supplementary figures in PNG/PDF/SVG. Figure 1 is a verified PowerPoint schematic export; the fourteen empirical figures are redrawn from saved numerical data.

## Scope limits

Full robustness-grid daily arrays, extended daily-stratum/original-unit arrays, non-core old-source daily diagnostics, and Italian candidate/E1/global/sorted daily arrays are omitted. **Their final result tables are retained**, including negative outcomes. Such retained summaries can be read and plotted, but this ZIP does not independently repeat their complete aggregation or bootstrap. No estimates were rounded or parameters changed to shrink the package; retained daily rows have exact original values. `CORE_ARCHIVES.json` identifies the two filtered tables and hashes every restored file.

The full numerical audit remains in the separately supplied `Supplementary_Code.zip`; this package does not require it for the commands above. Neither saved-score route reconstructs all SKU predictions or refits every original model. Uncertainty remains conditional on the fitted/selected paths. Stores/settings/seeds are not independent enterprises.

Small representative weight files and their inference check are described in `models/README.md`; they are not the complete rolling-retraining model bank.

## Finding results

`docs/RESULTS_MAP.md` maps main results and supplementary evidence. `SCOPE.json` lists exclusions. `docs/DISPLAY_MAP.json` preserves current display numbering. `DATA_DICTIONARY.json` defines the result fields; `docs/STATISTICS.md` explains analysis units and intervals; `docs/DATA_SOURCES.md` records data rights and access. `validation/compact_run.json` records the clean test of this distribution, including its limitations.

## Distribution and rights

Repository: https://github.com/mie-26/retail-recalibration. The single outer ZIP can be uploaded as a repository file; reviewers download and extract it locally before running. The ZIP does not unpack itself on GitHub. Local packaging has not created a remote release or pushed any changes.

The Italian source CSV retains its provider's CC BY-NC 3.0 notice. Full raw M5, Online Retail II and Dominick’s sales are not redistributed. See `LICENSE_STATUS.txt`; no unselected code licence is implied. SHA256 manifests protect file integrity; they are not scientific certification. Citation metadata remain in `CITATION.txt`.
