$ErrorActionPreference = "Stop"

$Gxx = "C:\msys64\ucrt64\bin\g++.exe"
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"

New-Item -ItemType Directory -Force -Path "build" | Out-Null

& $Gxx `
    -std=c++17 `
    -O3 `
    -Wall `
    -Wextra `
    -Wpedantic `
    -Isrc `
    src\main.cpp `
    src\json_stream.cpp `
    src\block_registry.cpp `
    src\piston.cpp `
    src\simulator.cpp `
    src\sim_event_log.cpp `
    src\trace.cpp `
    src\world.cpp `
    -o build\cpp_simulator_stream.exe
