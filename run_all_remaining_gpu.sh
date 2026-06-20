#!/usr/bin/env bash
# Orchestrator: run all remaining experiment queues on GPU (tf215_gpu).
# Series 3 (9), 1j training ablation (9), 1j LOSO (5) = 23 runs.
# Idempotent: each run skipped if model_int8.tflite exists.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CONDA_ENV=tf215_gpu
unset CUDA_VISIBLE_DEVICES   # allow GPU

LOG_DIR="$SCRIPT_DIR/logs_gpu_run"
mkdir -p "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] ===== START remaining GPU runs (env=$CONDA_ENV) ====="

echo "[$(stamp)] >>> Series 3 (MobileNetV3Small width sweep)"
bash run_series3_ablation.sh           >"$LOG_DIR/series3.log" 2>&1
echo "[$(stamp)] <<< Series 3 done (exit $?)"

echo "[$(stamp)] >>> 1j training ablation (aug x split)"
bash run_training_ablation_12class.sh  >"$LOG_DIR/training_ablation.log" 2>&1
echo "[$(stamp)] <<< Training ablation done (exit $?)"

echo "[$(stamp)] >>> 1j LOSO CV (builds split CSVs first)"
bash run_loso_cv.sh                    >"$LOG_DIR/loso.log" 2>&1
echo "[$(stamp)] <<< LOSO done (exit $?)"

echo "[$(stamp)] ===== ALL REMAINING GPU RUNS COMPLETE ====="
