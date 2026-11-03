@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title WeightSpace Lab - Local Web Dashboard
set "PY_CMD="

py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and sys.version_info.minor == 11 else 1)" >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.11"
if defined PY_CMD goto PYTHON_FOUND

py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and sys.version_info.minor == 12 else 1)" >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.12"
if defined PY_CMD goto PYTHON_FOUND

py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and sys.version_info.minor == 10 else 1)" >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.10"
if defined PY_CMD goto PYTHON_FOUND

python -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)" >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
if defined PY_CMD goto PYTHON_FOUND

echo [ERROR] Python 3.10, 3.11, or 3.12 is required.
echo Install Python from https://www.python.org/downloads/windows/
echo Enable the option to add Python to PATH during installation.
pause
exit /b 1

:PYTHON_FOUND
echo Using Python launcher: %PY_CMD%
%PY_CMD% --version

if exist ".venv\Scripts\python.exe" goto VENV_READY
echo [1/4] Creating virtual environment...
%PY_CMD% -m venv .venv
if errorlevel 1 goto ERROR

:VENV_READY
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" goto ERROR
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo [2/4] Updating pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto ERROR

"%VPY%" -c "import torch" >nul 2>&1
if not errorlevel 1 goto TORCH_READY
echo [3/4] Installing PyTorch CPU build...
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto ERROR

:TORCH_READY
echo [4/4] Installing web application dependencies...
"%VPY%" -m pip install --prefer-binary -r requirements.txt
if errorlevel 1 goto ERROR

echo.
echo Starting local web dashboard at http://127.0.0.1:8765
echo Keep this window open while using WeightSpace Lab.
start "" cmd /c "timeout /t 2 ^>nul ^& start http://127.0.0.1:8765"
"%VPY%" -m uvicorn webapp.app:app --host 127.0.0.1 --port 8765
if errorlevel 1 goto ERROR
goto END

:ERROR
echo.
echo [ERROR] Setup or launch failed.
echo Check Python version, internet access, and the error message above.
pause
exit /b 1

:END
endlocal
exit /b 0
