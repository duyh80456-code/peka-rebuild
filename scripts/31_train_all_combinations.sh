#!/bin/bash
# Sweep 4 PEFT methods × 2 encoders = 8 experiments.
# Run from the peka/ directory: bash scripts/31_train_all_combinations.sh
#
# Customize per-run flags via the env vars EXTRA_FLAGS, e.g.
#   EXTRA_FLAGS="--epochs 50 --batch_size 32 --with_logger wandb"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

ENCODERS=(H-optimus-0 UNI)
PEFTS=(bone lora adalora hra)

for encoder in "${ENCODERS[@]}"; do
  for peft in "${PEFTS[@]}"; do
    echo "==================================================="
    echo "Training: encoder=$encoder peft=$peft"
    echo "==================================================="
    python "$SCRIPT_DIR/30_train_phase2_kd.py" \
      --encoder "$encoder" \
      --peft "$peft" \
      $EXTRA_FLAGS
  done
done

echo "All 8 experiments done."
