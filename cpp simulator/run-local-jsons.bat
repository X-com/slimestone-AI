@echo off
setlocal

set "ROOT=%~dp0"
set "EXE=%ROOT%build\cpp_simulator_stream.exe"
set "COMPACT_DIR=%ROOT%local-jsons-compact"
set "GENETIC_ML=%ROOT%..\..\genetic-ml"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
pushd "%ROOT%"

if not exist "%EXE%" (
    call "%ROOT%build-cpp.bat"
    if errorlevel 1 (
        popd
        exit /b 1
    )
)

dir /b "%ROOT%*.json" >nul 2>nul
if errorlevel 1 (
    echo No .json flying-machine inputs found in:
    echo %ROOT%
    pause
    popd
    exit /b 1
)

REM cpp's file mode only reads the compact format now (see genetic_ml/compact_format.py) -
REM convert the JSON fixtures fresh on every run rather than requiring a stale hand-maintained
REM compact copy.
pushd "%GENETIC_ML%"
py convert_fixtures_to_compact.py "%ROOT%" "%COMPACT_DIR%"
set "CONVERT_ERR=%ERRORLEVEL%"
popd
if not "%CONVERT_ERR%"=="0" (
    popd
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$files = Get-ChildItem -Path '%COMPACT_DIR%\*.dat' | ForEach-Object { $_.FullName }; & '%EXE%' @files; exit $LASTEXITCODE"
echo.
echo Run finished with exit code %ERRORLEVEL%.
pause
popd
