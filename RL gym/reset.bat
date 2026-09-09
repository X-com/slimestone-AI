@echo off
setlocal

REM Delete EVERYTHING training has ever produced, so a new test starts from nothing.
REM
REM Why the whole `data` directory and not a curated list: training had been run across several
REM machines, and a partial wipe is the failure mode that matters here - a stale label file, a
REM cached graph or a k=2 label set from a different fixture is invisible, gets picked up
REM silently, and the run looks fresh while it is not. Everything under `data` is produced by
REM this project and can be rebuilt from the fixtures, so there is nothing here worth the risk
REM of keeping.
REM
REM   data\runs\      checkpoints (stage0\best.pt, live\), library, attempt log, metrics, replay
REM   data\labels\    exhaustive k=1 ground truth        ~1 min for one fixture
REM   data\graphs\    graph cache                        ~157ms per machine
REM   data\k2\        exhaustive k=2 ground truth        ~25 min - the expensive one, gone too
REM   anything else that has appeared in there since
REM
REM NOT touched: the fixtures. They live outside this folder and are inputs, never outputs.
REM
REM `data` is tracked in git, so if you want it back:  git checkout -- "RL gym/data"
REM
REM This ONLY resets. It does not start training - run train.bat when you want that.

set "GYM=%~dp0"
pushd "%GYM%"

echo.
if not exist "data" (
    echo   Already clean - there is no data folder.
    popd
    pause
    exit /b 0
)

echo   RESET - deletes the ENTIRE data folder ^(it does not start training^):
echo.
for /f "delims=" %%D in ('dir /b /a "data" 2^>nul') do echo       data\%%D
echo.
echo   All of it. Including data\k2, which takes ~25 min to rebuild - keeping it is
echo   how a "fresh" test ends up contaminated by the old one.
echo   Tracked in git: "git checkout -- ""RL gym/data""" restores everything.
echo.

set "CONFIRM="
set /p "CONFIRM=Type YES to delete it all: "
if /i not "%CONFIRM%"=="YES" (
    echo.
    echo Cancelled. Nothing was deleted.
    popd
    pause
    exit /b 1
)

echo.
echo   removing data
rmdir /s /q "data"

REM A locked file - most often a training run still holding a checkpoint, or the folder open in
REM Explorer - leaves part of the tree behind. Stopping loudly is the honest move: half a reset
REM is worse than none, because the next run looks fresh and is not.
if exist "data" (
    echo.
    echo   FAILED - something is still using it. What survived:
    for /f "delims=" %%D in ('dir /b /s /a-d "data" 2^>nul') do echo       %%D
    echo.
    echo   Close any running training ^(Ctrl-C in its window^) and any Explorer window
    echo   showing that folder, then run this again. DO NOT train until this is clean.
    popd
    pause
    exit /b 1
)

echo.
echo   Reset complete. Nothing from the old training remains.
echo   Run train.bat to start a new one - it rebuilds labels and graphs as it goes.
echo.
popd
pause
