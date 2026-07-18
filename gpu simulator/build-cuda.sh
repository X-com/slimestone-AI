#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p build

CPP_EXTRACT_SRC="../cpp simulator/src"

nvcc \
    -std=c++17 -O3 \
    -arch=sm_52 \
    --extended-lambda \
    -allow-unsupported-compiler \
    -Isrc/kernel \
    -I"$CPP_EXTRACT_SRC" \
    src/kernel/block_registry.cu \
    src/kernel/gpu_kernel.cu \
    src/host/main_gpu.cu \
    "$CPP_EXTRACT_SRC/json_stream.cpp" \
    -o build/gpu_kernel_stream

echo "Build succeeded: build/gpu_kernel_stream"
