#!/usr/bin/env bash
# Re-run the two queues that crashed first time, on GPU:
#   Series 3 (9 runs) — fixed: keras.applications (was tf_keras NameError)
#   1j LOSO  (5 runs) — fixed: per-fold output dir suffix (was dir collision)
# Idempotent: each run skipped if model_int8.tflite exists.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CONDA_ENV=tf215_gpu
unset CUDA_VISIBLE_DEVICES

LOG_DIR="$SCRIPT_DIR/logs_gpu_run2"
mkdir -p "$LOG_DIR"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] ===== START series3 + loso (GPU) ====="

echo "[$(stamp)] >>> Series 3"
bash run_series3_ablation.sh   >"$LOG_DIR/series3.log" 2>&1
echo "[$(stamp)] <<< Series 3 (exit $?)"

echo "[$(stamp)] >>> 1j LOSO"
bash run_loso_cv.sh            >"$LOG_DIR/loso.log" 2>&1
echo "[$(stamp)] <<< LOSO (exit $?)"

echo "[$(stamp)] ===== DONE ====="
