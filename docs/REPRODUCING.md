# Execution scope

Run `python reproduce.py` after installing `requirements.txt`. The entry point verifies hashes, restores the four archives, runs feedback tests, core/control numerical checks, Italian subset checks, all plots and final consistency checks. It stops on any failed step.

This compact route intentionally has a smaller daily-input scope than the full supplied Supplementary_Code archive. See README.md and SCOPE.json for exactly which arrays are omitted. The saved result tables remain included; a successful compact run must not be reported as a full-grid, full-model-bank or raw-SKU reconstruction.

Reference blocks use the original 3,000 draws; Italian interactions and effects use their original 5,000 draws and all archived block lengths. This is re-execution of fixed analyses, not new experiments or reselection.
