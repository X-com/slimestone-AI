@echo off
setlocal

set "ROOT=%~dp0"
set "EXE=%ROOT%build\gpu_kernel_stream.exe"
set "INPUT_DIR=%ROOT%..\flying machines\cycle-test"
set "COMPACT_DIR=%ROOT%cycle-test-compact"
set "GENETIC_ML=%ROOT%..\genetic algorithm"
pushd "%ROOT%"

if not exist "%EXE%" (
    call "%ROOT%build-cuda.bat"
    if errorlevel 1 (
        popd
        exit /b 1
    )
)

dir /b "%INPUT_DIR%\*.json" >nul 2>nul
if errorlevel 1 (
    echo No .json fixtures found in:
    echo %INPUT_DIR%
    pause
    popd
    exit /b 1
)

REM The CUDA kernel's file mode only reads the compact format (see
REM genetic_ml/compact_format.py) - convert the JSON cycle-test fixtures fresh on every run
REM rather than requiring a stale hand-maintained compact copy. Same fixtures
REM cpp simulator/run-cycle-test.bat verifies validCycle against - simple_valid_cycle should come
REM back "validCycle":true, simple_not_valid_cycle should come back "validCycle":false.
pushd "%GENETIC_ML%"
py convert_fixtures_to_compact.py "%INPUT_DIR%" "%COMPACT_DIR%"
set "CONVERT_ERR=%ERRORLEVEL%"
popd
if not "%CONVERT_ERR%"=="0" (
    popd
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$files = @(Get-ChildItem -Path '%COMPACT_DIR%\*.dat' | ForEach-Object { $_.FullName }); & '%EXE%' @files; exit $LASTEXITCODE"
echo.
echo Run finished with exit code %ERRORLEVEL%.
pause
popd
