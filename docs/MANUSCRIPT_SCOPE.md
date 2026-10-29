# Final manuscript scope

This package supports the manuscript finalized on 29 October 2026. The latest changes clarify the interpretation of existing results; no numerical data or scientific computation was changed for those textual revisions.

- Results 4.4: identical short-/long-history R5 scores refer to M5 and Online Retail II. Dominick's/LightGBM/H1/U4 is an exception: TM versus TS joint scaled pinball changes by approximately −0.226823%. The saved scores are in `controlled/tables/E1_quantile_scores.csv` after restoration.
- Results 4.2: the quoted +0.272%, +1.109% and −0.603% delay examples use the calendar window. See `supplementary_data/Data2_feedback_effects.csv`.
- Results 4.5: the Q2−Q1 output-feedback comparison uses the original step coefficient 0.1. See `robustness/tables/paper_three_seed_mechanism.csv` and the retained robustness protocol.

The article has four main figures, three main tables, eleven supplementary figures and twenty-four supplementary tables. Main Figure 4 and Table 3 distinguish the selected rule-by-sharing interaction from loss improvements over Raw. The 26/144 fixed-grid reversals and the adverse Raw comparisons remain part of the evidence.

This compact package is derived from the manuscript submission's reproduction archive. It preserves the numerical values of retained data, with explicit input-subset and dispatch changes documented in SCOPE.json and CORE_ARCHIVES.json; it is not byte-identical to the full archive. No manuscript, author data or historical version bundle is distributed here.
