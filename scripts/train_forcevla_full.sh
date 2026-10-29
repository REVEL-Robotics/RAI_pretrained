#!/usr/bin/env bash
# Train ForceVLA with full fine-tuning.
#
#   scripts/train_forcevla_full.sh
#   scripts/train_forcevla_full.sh --task flexiv_wipe_board_inputForce
#   scripts/train_forcevla_full.sh --task all
#
# Default task: flexiv_peel_cucumber_inputForce
# "all" trains on every task in the ForceVLA dataset.
#
# Norm stats are stored per task, or under forcevla_all. Compute them once for
# the selection you are about to train:
#   python scripts/compute_norm_stats.py --config-name forcevla_full --task TASK
set -euo pipefail

cd "$(dirname "$0")/.."

TASK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --task" >&2
        exit 1
      fi
      TASK="$2"
      shift 2
      ;;
    --task=*)
      TASK="${1#*=}"
      shift
      ;;
    -h|--help)
      sed -n '2,13p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: scripts/train_forcevla_full.sh [--task TASK]" >&2
      exit 1
      ;;
  esac
done

DATA_ARGS=()
if [[ -n "$TASK" ]]; then
  DATA_ARGS+=(--data.task="$TASK")
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py forcevla_full \
  "${DATA_ARGS[@]}" \
  --exp-name=my_experiment --overwrite --save_interval 2000 --keep_period 10000
