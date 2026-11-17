# R4.4 SAC implementation checks and reruns

This package reproduces the paired reruns of the retained and revised SAC implementations on the original five representative days. It includes the experiment driver, its inputs, the selected archived source and patched modules, and the recorded settings. The reported seeds are 42–46. The full experiment comprises 40 legacy and revised runs and 10 ablation runs, with 61,500 logged transitions. Iteration counts are matched within each algorithm. EMS request counts can differ between implementations.

## Environment

The archived environment was Windows x64 with Python 3.13.3, NumPy 2.2.6, pandas 2.3.2, PyTorch 2.7.1 CPU, Matplotlib 3.10.6, and SciPy 1.16.1. Install the CPU build of PyTorch first, then install the pinned requirements. The bundled EMS DLL is for Windows x64. Its C source and optional PowerShell build script are supplied.

```powershell
python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r code/requirements.txt
```

## Run

From this directory in PowerShell:

```powershell
New-Item -ItemType Directory -Path results -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
& .\code\reproduce.ps1 -Python (Get-Command python).Source
```

The driver runs the contracts, legacy and revised arms, the SA action-mapping ablation, the PSO transition-timing ablation, and aggregation. It writes stepwise logs and summaries under `results/` and figures under `figures/`. Run it on a fresh copy to keep the supplied package intact.

For the contract checks and a 12-step replay of the archived source loops:

```powershell
New-Item -ItemType Directory -Path results -Force | Out-Null
python -B code/test_contracts.py
python -B code/run_experiments.py --arms legacy --seeds 42 --smoke
python -B code/test_source_loops.py
```

After the full run, `python -B code/test_repeat.py` checks whether the revised seed-42 PSO trajectory is reproduced byte for byte. The `validation/archived_*.json` files report the saved validation outcomes. Newly run checks write separate files without the `archived_` prefix. `audit/settings.json` records settings and `audit/before_after.patch` records changes in the patched modules. `SHA256SUMS` lists the checksums of the supplied experiment files.

The archived main functions in `source_snapshot/` are used for the source-loop checks. `code/run_experiments.py` is the unified experiment entry. The evaluator retains the original representative-day objective. The chronology-based cost audit and common-EMS-budget comparisons use separate protocols.
