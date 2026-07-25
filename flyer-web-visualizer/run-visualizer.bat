@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

if not exist "%ROOT%node_modules" (
    echo Installing dependencies - first run only...
    call npm install
    if errorlevel 1 (
        echo npm install failed.
        pause
        popd
        exit /b 1
    )
)

echo Starting the visualizer dev server - open the printed localhost URL in your browser.
echo Press Ctrl+C to stop.
call npm run dev

popd
