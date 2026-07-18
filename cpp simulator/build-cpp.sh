#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p build

g++ \
    -std=c++17 \
    -O3 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -Isrc \
    src/main.cpp \
    src/json_stream.cpp \
    src/block_registry.cpp \
    src/piston.cpp \
    src/simulator.cpp \
    src/trace.cpp \
    src/world.cpp \
    -o build/cpp_simulator_stream

echo "Build complete: $ROOT/build/cpp_simulator_stream"
