#!/usr/bin/env bash
# EfficientNetB0 (1r) split ablation — 3 splits × 3 seeds = 9 runs.
#
# Compares train/val/test allocation on 12-class MyGardenBird dataset:
#   80:10:10  →  5760/720/720  (current standard)
#   75:10:15  →  5400/720/1080
#   70:15:15  →  5040/1080/1080
#
# All other hyperparameters fixed: n_mels=224, mixup=0.2, seeds 42/100/786.
# Results land in: results_mygardenbird_1_{darwin|linux}/
#
# Usage:
#   bash run_1r_splits_ablation.sh            # all 9 runs
#   bash run_1r_splits_ablation.sh 80:10:10   # only 3 seeds for that split
#   bash run_1r_splits_ablation.sh 75:10:15
#   bash run_1r_splits_ablation.sh 70:15:15

set -euo pipefail

FLAT_DIR="/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
META_DIR="/Volumes/Evo/MYGARDENBIRD/metadata16khz"
SEEDS=(42 100 786)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILTER="${1:-all}"

declare -A SPLIT_CSV
SPLIT_CSV["80:10:10"]="$META_DIR/splits_mip_80_10_10.csv"
SPLIT_CSV["75:10:15"]="$META_DIR/splits_mip_75_10_15.csv"
SPLIT_CSV["70:15:15"]="$META_DIR/splits_mip_70_15_15.csv"

SPLIT_KEYS=("80:10:10" "75:10:15" "70:15:15")

run_one() {
    local split_key="$1"
    local seed="$2"
    local csv="${SPLIT_CSV[$split_key]}"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  1r_efficientnetb0  split=$split_key  seed=$seed"
    echo "════════════════════════════════════════════════════"

    conda run -n tf215_gpu python3 "$SCRIPT_DIR/1r_efficientnetb0.py" \
        --splits_csv "$csv" \
        --flat_dir   "$FLAT_DIR" \
        --n_mels 224 \
        --mixup 0.2 \
        --random_seed "$seed"
}

for split_key in "${SPLIT_KEYS[@]}"; do
    if [[ "$FILTER" == "all" || "$FILTER" == "$split_key" ]]; then
        echo ""
        echo "▶▶▶  1r split=$split_key (3 seeds)"
        for seed in "${SEEDS[@]}"; do
            run_one "$split_key" "$seed"
        done
    fi
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  All runs complete."
echo "════════════════════════════════════════════════════"
