#!/usr/bin/env bash
# Run torch.compile lessons with verbose Dynamo logging.
#
# Usage:
#   bash run_debug.sh            # runs all lessons with debug logs
#   bash run_debug.sh 02         # runs only lesson 02
#
# Log tokens:
#   +dynamo      — bytecode tracing, graph construction
#   +guards      — guard installation and evaluation
#   +recompiles  — recompilation events with reasons
#   +aot         — AOTAutograd joint/forward/backward graphs
#   +graph_breaks — location and reason for every graph break

set -euo pipefail

export TORCH_LOGS="+dynamo,+guards,+recompiles,+graph_breaks"
export TORCH_COMPILE_DEBUG=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_lesson() {
    local file="$1"
    echo ""
    echo "========================================"
    echo "Running: $(basename "$file")"
    echo "========================================"
    python "$file"
}

if [[ $# -eq 1 ]]; then
    # Run a single lesson by prefix, e.g. bash run_debug.sh 02
    file=$(ls "$SCRIPT_DIR/${1}"_*.py 2>/dev/null | head -1)
    if [[ -z "$file" ]]; then
        echo "No file matching '${1}_*.py' found." >&2
        exit 1
    fi
    run_lesson "$file"
else
    # Run all lessons in order
    for file in "$SCRIPT_DIR"/0*.py; do
        run_lesson "$file"
    done
fi

echo ""
echo "Done. Debug logs above show Dynamo tracing, guard checks, and recompilations."
