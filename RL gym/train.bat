@echo off
setlocal enabledelayedexpansion

REM Double-click this. It brings up the training dashboard and runs the AlphaZero loop.
REM
REM This is the launcher and it lives inside RL gym, so the project stays self-contained: you
REM can copy this folder elsewhere and still start it. The only thing here that reaches
REM outside RL gym is the C++ simulator, which is the one dependency this project genuinely
REM has and cannot bring with it.
REM
REM Everything below is the pipeline from md/TRAINING.md, in the same order, and each stage is
REM SKIPPED if its output already exists - so the first run on a fresh clone does the slow work
REM once and every run after it starts training in seconds.
REM
REM   labels      exhaustive k=1 ground truth        ~20 min, once, ever
REM   stage 0     supervised on those labels         ~65 min, once, ever
REM   viewer      the flyer-web-visualizer, in its own window
REM   stage 1     the loop, streaming to that viewer  runs until you stop it
REM
REM Same conventions as the other launchers in this repo (see reinforcement learning/run-rl.bat).

set "GYM=%~dp0"
set "PYTHON=D:\ProgramFiles\Python313\python.exe"
set "SIMULATOR=%GYM%..\cpp simulator\build\cpp_simulator_stream.exe"
set "VISUALIZER=%GYM%..\flyer-web-visualizer"
REM The WebSocket the loop streams discoveries on, and the port the viewer's Connect box
REM defaults to. Change both together or the viewer will not find the run.
set "PORT=8765"

pushd "%GYM%"

if not exist "%PYTHON%" (
    echo.
    echo Python not found at:
    echo   %PYTHON%
    echo.
    echo Edit PYTHON in this .bat to point at your Python 3.11+ install - the one with
    echo numpy and torch installed against it.
    goto :fail
)

"%PYTHON%" -c "import numpy, torch, websockets" 2>nul
if errorlevel 1 (
    echo.
    echo numpy, torch and websockets are needed and at least one is missing for:
    echo   %PYTHON%
    echo.
    echo   "%PYTHON%" -m pip install numpy torch websockets
    goto :fail
)

if not exist "%SIMULATOR%" (
    echo.
    echo The C++ simulator is not built:
    echo   %SIMULATOR%
    echo.
    echo Build it first with "cpp simulator\build-cpp.bat" - nothing here can run without it,
    echo since the simulator is what decides whether a machine flies.
    goto :fail
)

REM --- stage: the exhaustive k=1 corpus ---------------------------------------------------
REM label_corpus caches per machine, so re-running it after an interrupted sweep resumes
REM rather than starting over.
if not exist "data\labels\simple_machine2.json" (
    echo.
    echo [1/4] Labelling the corpus. This is a one-off and takes about 20 minutes.
    echo       Interrupting is safe - it resumes from the machines already done.
    echo.
    "%PYTHON%" -m rlgym.labeller --all --out data\labels
    if errorlevel 1 goto :fail
) else (
    echo [1/4] Labels found - skipping the corpus sweep.
)

REM --- stage: supervised pre-training -----------------------------------------------------
if not exist "data\runs\stage0\best.pt" (
    echo.
    echo [2/4] Stage 0: supervised pre-training. One-off, about 65 minutes on CPU.
    echo       The number to watch is held-out AUC against the baseline printed at the top.
    echo.
    "%PYTHON%" -m rlgym.train --config configs\stage0.json --out data\runs\stage0
    if errorlevel 1 goto :fail
) else (
    echo [2/4] Stage 0 checkpoint found - skipping supervised pre-training.
)

REM --- stage: the viewer ------------------------------------------------------------------
REM Started in its own window, and NOT waited on: `npm run dev` never returns, so a `call` here
REM would hang before training ever began. Failing to start the viewer is deliberately not
REM fatal - training is the point, and the run is still fully recorded to disk without it.
if exist "%VISUALIZER%\package.json" (
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [3/4] npm not on PATH - skipping the viewer, training will still run.
        echo       Install Node.js to see discoveries in 3D: https://nodejs.org
    ) else (
        if not exist "%VISUALIZER%\node_modules" (
            echo [3/4] Installing the viewer's dependencies, one-off...
            pushd "%VISUALIZER%"
            call npm install
            popd
        )
        echo [3/4] Starting the viewer and opening it on the Live Training page.
        REM dev:live is `vite --open "/#/live"` - vite resolves its own port, so this still
        REM opens the right tab when 5173 is taken and it falls back to 5174. The router is
        REM hash-based, hence the #.
        start "flyer-web-visualizer" cmd /c "cd /d ""%VISUALIZER%"" && npm run dev:live"
    )
) else (
    echo [3/4] flyer-web-visualizer not found next to RL gym - skipping the viewer.
)

REM --- stage: the loop --------------------------------------------------------------------
echo.
echo [4/4] Stage 1: the training loop, streaming to ws://localhost:%PORT%
echo       Open the viewer's printed URL, go to Live Training, and press Connect - discovered
echo       machines appear in 3D as they are found. Ctrl-C stops the run; the library, metrics
echo       and checkpoint are saved at the end of every round.
echo.

"%PYTHON%" -m rlgym.serve --config configs\stage1.json --checkpoint data\runs\stage0\best.pt --out data\runs\live --port %PORT%

echo.
echo Finished with exit code %ERRORLEVEL%.
popd
pause
exit /b 0

:fail
echo.
echo Stopped. Nothing was changed.
popd
pause
exit /b 1
