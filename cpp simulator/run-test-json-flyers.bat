@echo off
setlocal

set "ROOT=%~dp0"
set "EXE=%ROOT%build\cpp_simulator_stream.exe"
set "INPUT_DIR=%ROOT%..\flying machines\json"
set "COMPACT_DIR=%ROOT%flying-json-compact"
set "GENETIC_ML=%ROOT%..\genetic algorithm"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
REM None of these fixtures have negative y, and a couple (e.g. the trenchers) are tall enough
REM that the default +64 y-offset would push their top past the simulator's y<256 bound and
REM silently truncate it - skip the offset entirely.
set "MCP1122_CPP_NO_Y_OFFSET=1"
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

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$files = @(Get-ChildItem -Path '%COMPACT_DIR%\*.dat' | Sort-Object Name); " ^
  "$lines = @(& '%EXE%' @($files.FullName) 2>$null); " ^
  "if ($lines.Count -ne $files.Count) { Write-Host \"warning: got $($lines.Count) result(s) for $($files.Count) fixture(s) - report below may be misaligned\" -ForegroundColor Yellow }; " ^
  "$fail = 0; " ^
  "for ($i = 0; $i -lt $files.Count -and $i -lt $lines.Count; $i++) { " ^
  "  $name = $files[$i].BaseName; " ^
  "  $r = $lines[$i] | ConvertFrom-Json; " ^
  "  $expectCycle = $name -notmatch 'doesnt_loop|not_valid'; " ^
  "  if (-not $r.ok) { $status = 'ERROR'; $fail++ } " ^
  "  elseif ([bool]$r.validCycle -eq $expectCycle) { $status = 'PASS' } " ^
  "  else { $status = 'FAIL'; $fail++ }; " ^
  "  '{0,-45} expect={1,-5} validCycle={2,-5} {3}' -f $name, $expectCycle, [bool]$r.validCycle, $status; " ^
  "}; " ^
  "if ($fail -gt 0) { Write-Host \"$fail of $($files.Count) fixture(s) did not match their name's expected cycle behavior\" -ForegroundColor Red } " ^
  "else { Write-Host \"all $($files.Count) fixture(s) matched their name's expected cycle behavior\" -ForegroundColor Green }; " ^
  "exit $fail"
echo.
echo Run finished with exit code %ERRORLEVEL%.
pause
popd
