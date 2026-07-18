#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$ROOT/build/cpp_simulator_stream"
INPUT_DIR="$ROOT/error-fix"
COMPACT_DIR="$ROOT/error-fix-compact"
GENETIC_ML="$ROOT/../genetic algorithm"
# Each candidate gets its own file: outlog/cpp-update-trace-<id>.log, instead of every
# candidate scrambling into one shared trace file.
TRACE_LOG="$ROOT/outlog/cpp-update-trace.log"
cd "$ROOT"

if [ ! -x "$EXE" ]; then
    "$ROOT/build-cpp.sh"
fi

shopt -s nullglob
json_files=("$INPUT_DIR"/*.json)
shopt -u nullglob
if [ ${#json_files[@]} -eq 0 ]; then
    echo "No .json fixtures found in:"
    echo "$INPUT_DIR"
    echo "Put flying-machine stream JSON files in the flying-json folder next to this script."
    exit 1
fi

# cpp's file mode only reads the compact format now (see genetic_ml/compact_format.py) -
# convert the JSON fixtures fresh on every run rather than requiring a stale hand-maintained
# compact copy.
(cd "$GENETIC_ML" && python3 convert_fixtures_to_compact.py "$INPUT_DIR" "$COMPACT_DIR")

shopt -s nullglob
dat_files=("$COMPACT_DIR"/*.dat)
shopt -u nullglob
set +e
"$EXE" --trace "$TRACE_LOG" "${dat_files[@]}"
exit_code=$?
set -e
echo
echo "Run finished with exit code $exit_code."
echo "Piston trace logs: $ROOT/outlog/cpp-update-trace-<id>.log (one per candidate)"
exit "$exit_code"
