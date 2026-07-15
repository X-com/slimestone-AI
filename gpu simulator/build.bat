@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%build-msys2.ps1"
if errorlevel 1 (
    echo Build failed.
    pause
    popd
    exit /b 1
)

echo Build complete: %ROOT%build\gpu_simulator_stream.exe
pause
popd
