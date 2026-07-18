#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$ROOT/build/cpp_simulator_stream"
FLYERS_FILE="$ROOT/../genetic algorithm/data/compact-working/flyers.data"
cd "$ROOT"

if [ ! -x "$EXE" ]; then
    "$ROOT/build-cpp.sh"
fi

if [ ! -f "$FLYERS_FILE" ]; then
    echo "No flyers.data found at:"
    echo "$FLYERS_FILE"
    echo 'Run main_ga.py with WORKING_STORAGE_FORMAT = "compact" first to generate it.'
    exit 1
fi

# flyers.data is already compact format (see genetic_ml/compact_working_writer.py) - no
# conversion step needed, just pass it straight to file mode as one input.
set +e
"$EXE" "$FLYERS_FILE"
exit_code=$?
set -e
echo
echo "Run finished with exit code $exit_code."
exit "$exit_code"
