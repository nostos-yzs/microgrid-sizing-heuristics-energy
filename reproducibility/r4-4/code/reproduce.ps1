param([string]$Python='E:\python\python.exe')
$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $Here 'test_contracts.py');if($LASTEXITCODE){throw 'Contract checks failed'}
& $Python (Join-Path $Here 'run_experiments.py') --seeds 42 43 44 45 46;if($LASTEXITCODE){throw 'Full experiment failed'}
& $Python (Join-Path $Here 'run_experiments.py') --algorithms SA --arms mapping_only --seeds 42 43 44 45 46;if($LASTEXITCODE){throw 'SA ablation failed'}
& $Python (Join-Path $Here 'run_experiments.py') --algorithms PSO --arms transition_only --seeds 42 43 44 45 46;if($LASTEXITCODE){throw 'PSO timing ablation failed'}
& $Python (Join-Path $Here 'make_figures.py');if($LASTEXITCODE){throw 'Aggregation failed'}
& $Python (Join-Path $Here 'export_audit.py')
