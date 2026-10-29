# Figure 1 editable mechanism schematic

`Figure_1.pptx` contains 181 editable native shapes, text boxes and connectors. The observation bars, quantile pair and residual marks are schematic, not measured values. Binary traces follow the exact timing example in `figure_source_data/figure1_design.json`; availability is not a coverage or performance guarantee.

Open the PPTX in PowerPoint to edit. The supplied PDF is a vector export at 7.2 by 5.85 inches with embedded fonts. PNG is rendered at 600 dpi, and SVG retains outlined text. `code/figure1_design.py` verifies the supplied export hashes before restoring the figure.

Optional native regeneration: install `@oai/artifact-tool` and run `node figure_assets/build_pptx.mjs /path/to/new/Figure_1.pptx`. The supplied shape specification recreates the slide. This authoring dependency is not needed for statistical reproduction. Visually inspect any edited export and intentionally update its hashes.

This revision uses a functional scientific schematic with no shadows, gradients or rendered illustration assets. Visual design references, inspected at their published figures, were Bodnar et al., Nature 2025, Figure 1 (https://doi.org/10.1038/s41586-025-09005-y); Lee et al., ICML 2025, Figure 1 (https://proceedings.mlr.press/v267/lee25ag.html); and Im and Kwon, ICLR 2026, Figure 1 (https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html). These inform hierarchy and temporal/data-flow presentation. No source artwork or figure elements are copied. An earlier conceptual exploration used image generation; no generated pixels appear in this final figure.
