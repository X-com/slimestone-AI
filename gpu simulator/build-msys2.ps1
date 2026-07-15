$ErrorActionPreference = "Stop"

$Gxx = "C:\msys64\ucrt64\bin\g++.exe"
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"

$CppExtractSrc = "..\cpp simulator\src"

New-Item -ItemType Directory -Force -Path "build" | Out-Null

& $Gxx `
    -std=c++17 `
    -O3 `
    -Wall `
    -Wextra `
    -Wpedantic `
    -Isrc\reference `
    -I"$CppExtractSrc" `
    src\reference\main.cpp `
    src\reference\piston.cpp `
    src\reference\simulator.cpp `
    src\reference\trace.cpp `
    src\reference\world.cpp `
    "$CppExtractSrc\json_stream.cpp" `
    "$CppExtractSrc\block_registry.cpp" `
    -o build\gpu_simulator_stream.exe
