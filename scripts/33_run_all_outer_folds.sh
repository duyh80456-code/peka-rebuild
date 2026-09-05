#!/bin/bash
# Train and evaluate every leakage-free outer fold sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDS="${FOLDS:-5}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

for ((fold=0; fold<FOLDS; fold++)); do
  echo "==================================================="
  echo "Outer fold $fold/$((FOLDS - 1))"
  echo "==================================================="
  python "$SCRIPT_DIR/32_run_outer_fold.py" \
    --fold "$fold" \
    --folds "$FOLDS" \
    --run_id "$RUN_ID" \
    $EXTRA_FLAGS
done

echo "All $FOLDS leakage-free outer folds completed (run_id=$RUN_ID)."
