@echo off
setlocal

set "ROOT=%~dp0"
set "EXE=%ROOT%build\cpp_simulator_stream.exe"
set "FLYERS_FILE=%ROOT%..\..\genetic algorithm\data\compact-working\flyers.data"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
pushd "%ROOT%"

if not exist "%EXE%" (
    call "%ROOT%build-cpp.bat"
    if errorlevel 1 (
        popd
        exit /b 1
    )
)

if not exist "%FLYERS_FILE%" (
    echo No flyers.data found at:
    echo %FLYERS_FILE%
    echo Run main_ga.py with WORKING_STORAGE_FORMAT = "compact" first to generate it.
    pause
    popd
    exit /b 1
)

REM flyers.data is already compact format (see genetic_ml/compact_working_writer.py) - no
REM conversion step needed, just pass it straight to file mode as one input.
"%EXE%" "%FLYERS_FILE%"
echo.
echo Run finished with exit code %ERRORLEVEL%.
pause
popd
