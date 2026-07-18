#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_EXTRACT_SRC="$ROOT/../cpp simulator/src"
cd "$ROOT"

mkdir -p build

g++ \
    -std=c++17 \
    -O3 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -Isrc/reference \
    -I"$CPP_EXTRACT_SRC" \
    src/reference/main.cpp \
    src/reference/piston.cpp \
    src/reference/simulator.cpp \
    src/reference/trace.cpp \
    src/reference/world.cpp \
    "$CPP_EXTRACT_SRC/json_stream.cpp" \
    "$CPP_EXTRACT_SRC/block_registry.cpp" \
    -o build/gpu_simulator_stream

echo "Build complete: $ROOT/build/gpu_simulator_stream"
