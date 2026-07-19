@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=D:\ProgramFiles\Python313\python.exe"
pushd "%ROOT%"

if not exist "%PYTHON%" (
    echo Python not found at:
    echo %PYTHON%
    echo Edit PYTHON in this .bat to point at your Python 3.13 install ^(the one with
    echo "pip install websockets" run against it^).
    pause
    popd
    exit /b 1
)

"%PYTHON%" main_rl.py
echo.
echo Run finished with exit code %ERRORLEVEL%.
pause
popd
