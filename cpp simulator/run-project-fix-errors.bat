@echo off
setlocal

set "ROOT=%~dp0"
set "EXE=%ROOT%build\cpp_simulator_stream.exe"
set "INPUT_DIR=%ROOT%error-fix"
set "COMPACT_DIR=%ROOT%error-fix-compact"
set "GENETIC_ML=%ROOT%..\..\genetic-ml"
REM Each candidate gets its own file: outlog\cpp-update-trace-<id>.log, instead of every
REM candidate scrambling into one shared trace file.
set "TRACE_LOG=%ROOT%outlog\cpp-update-trace.log"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
pushd "%ROOT%"

if not exist "%EXE%" (
    call "%ROOT%build-cpp.bat"
    if errorlevel 1 (
        popd
        exit /b 1
    )
)

dir /b "%INPUT_DIR%\*.json" >nul 2>nul
if errorlevel 1 (
    echo No .json fixtures found in:
    echo %INPUT_DIR%
    echo Put flying-machine stream JSON files in the flying-json folder next to this .bat file.
    pause
    popd
    exit /b 1
)

REM cpp's file mode only reads the compact format now (see genetic_ml/compact_format.py) -
REM convert the JSON fixtures fresh on every run rather than requiring a stale hand-maintained
REM compact copy.
pushd "%GENETIC_ML%"
py convert_fixtures_to_compact.py "%INPUT_DIR%" "%COMPACT_DIR%"
set "CONVERT_ERR=%ERRORLEVEL%"
popd
if not "%CONVERT_ERR%"=="0" (
    popd
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$files = @(Get-ChildItem -Path '%COMPACT_DIR%\*.dat' | ForEach-Object { $_.FullName }); & '%EXE%' '--trace' '%TRACE_LOG%' @files; exit $LASTEXITCODE"
echo.
echo Run finished with exit code %ERRORLEVEL%.
echo Piston trace logs: %ROOT%outlog\cpp-update-trace-^<id^>.log (one per candidate)
pause
popd
