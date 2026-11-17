@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "MPLBACKEND=Agg"
if defined R45_PYTHON (
    set "R45_EXE=%R45_PYTHON%"
) else (
    if exist "E:\python\python.exe" (
        set "R45_EXE=E:\python\python.exe"
    ) else (
        set "R45_EXE=python"
    )
)
"%R45_EXE%" "%~dp0run_all.py" %*
set "R45_EXIT=%ERRORLEVEL%"
if not "%R45_EXIT%"=="0" echo R4.5 stopped. Read the error above. Completed runs are preserved.
exit /b %R45_EXIT%
