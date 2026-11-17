# Energy heuristic optimization: reproduction release

This release provides GA, PSO, SA and TS implementations, representative-day input data, and the R4.4/R4.5 experimental protocols used to study static and SAC-guided repair operators.

## Contents and commands

| Directory | Purpose | Entry |
| --- | --- | --- |
| `main/` | GA, PSO, SA, TS with baseline, static and SAC operators | `python main/run_ga.py --no-show --device cpu` (also pso, sa, ts) |
| `reproducibility/r4-4/` | Legacy/revised SAC, seeds 42–46, contracts and ablations | See the included README; pass an explicit Python path to `code/reproduce.ps1` |
| `reproducibility/r4-5/R4_5_B120000/` | Paired common-budget campaign, 22 seeds, 120,000 EMS requests per formal run | `python run_all.py --output <new-results-directory>` from that directory |
| `reference_figures/` | PDF figures accompanying the study | View for reference alongside generated experiment outputs |

## Clean Windows reproduction

Use Python 3.10 on Windows x64. Run from this folder:

```powershell
python verify_release.py --work ../energy_validation_clean
```

The script creates an isolated venv, installs pinned CPU dependencies, checks the environment and short runs, then runs the main examples and full R4.4/R4.5 protocols. It stops at the first failure and retains logs/results. R4.4 uses the supplied Windows x64 DLL, with its C source and build script included. Use a new work directory for each independent experiment. The full campaign can be lengthy.

The wrapper uses NumPy 2.2.6, pandas 2.3.2, torch 2.7.1 CPU and SciPy 1.15.3. R4.4's archived environment was Python 3.13.3/SciPy 1.16.1. Historical requirements are preserved inside each package. The wrapper records the installed dependency versions for comparison across environments.

## Scientific interpretation

Main defaults use seed 10, one repetition and iteration budgets. For the 22-seed Appendix J campaign, use the R4.5 entry point. Main target times/counts are proportional estimates relative to each run's final objective. Main stdout selects the best feasible repetition; use all rows in results.json for statistics. Convergence means omit infeasible values, so they are conditional on feasibility.

R4.5 fixes tuning, pilot and formal seeds in experiment_config.json. The target is frozen from pilot results before formal runs. Preserve all outcomes, including failed/negative runs; do not retune on formal results. Its raw logs and generated statistics are the relevant outputs for common-budget claims.

## File provenance

Source files originate from the main-code collection and the review archive dated 2026-11-16. SOURCE_SHA256.json records the assembled files. Original source manifests remain inside the packages. Representative-day input CSV files are included with the corresponding implementations. Each new experiment writes its own results for comparison with the study's reference material.
