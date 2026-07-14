$ErrorActionPreference = "Stop"

$Gxx = "C:\msys64\ucrt64\bin\g++.exe"
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"

New-Item -ItemType Directory -Force -Path "build-profile" | Out-Null

# -O2 so hot paths aren't heavily inlined away, -DMCP_PROFILE enables instrumentation
& $Gxx `
    -std=c++17 `
    -O2 `
    -DMCP_PROFILE `
    -Wall `
    -Wextra `
    -Wpedantic `
    -Isrc `
    src\main.cpp `
    src\json_stream.cpp `
    src\block_registry.cpp `
    src\piston.cpp `
    src\simulator.cpp `
    src\trace.cpp `
    src\world.cpp `
    src\profiler.cpp `
    -o build-profile\cpp_simulator_stream.exe
