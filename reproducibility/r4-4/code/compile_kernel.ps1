# The C legacy kernel is equivalent to the retained Python evaluator on the validation set.
# Build is optional on Windows if bundled ems_kernel.dll loads successfully.
param([string]$Gcc='E:\mingw64\bin\gcc.exe')
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
& $Gcc -O2 -shared (Join-Path $Here 'ems_kernel.c') -o (Join-Path $Here 'ems_kernel.dll') -lm
if($LASTEXITCODE){throw 'Kernel compilation failed'}
