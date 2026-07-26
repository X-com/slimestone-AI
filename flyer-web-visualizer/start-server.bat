@echo off
setlocal

REM Starts the flyer-web-visualizer's Vite dev server. Run from anywhere - cds to this
REM script's own directory first (%~dp0), same convention as the other .bat launchers in
REM this repo (e.g. cpp simulator/run-test-json-flyers.bat).
set "ROOT=%~dp0"
pushd "%ROOT%"

if not exist "node_modules" (
    echo node_modules not found - running npm install first...
    call npm install
    if errorlevel 1 (
        popd
        exit /b 1
    )
)

call npm run dev
popd
