#!/usr/bin/env bash
# Series 2 TCN ablation on the 12-class MyGardenBird dataset.
#
# Identical hyperparameters for direct comparison:
#   n_mels=64, dropout=0.05, warmup=70, mixup=0.2, split 80:10:10
#   seeds: 42, 100, 786
#
# Results land in: results_mygardenbird_2_{darwin|linux}/
#
# Usage:
#   bash run_series2_ablation.sh          # all 24 runs (8 variants × 3 seeds)
#   bash run_series2_ablation.sh 2a       # only 2a (3 seeds)
#   bash run_series2_ablation.sh 2b       # only 2b (3 seeds)
#   bash run_series2_ablation.sh 2c       # only 2c (3 seeds)
#   bash run_series2_ablation.sh 2d       # only 2d (3 seeds)
#   bash run_series2_ablation.sh 2e       # only 2e (3 seeds)
#   bash run_series2_ablation.sh 2f       # only 2f (3 seeds)
#   bash run_series2_ablation.sh 2g       # only 2g (3 seeds)
#   bash run_series2_ablation.sh 2h       # only 2h (3 seeds)

set -euo pipefail

SPLITS_CSV="/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
FLAT_DIR="/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SEEDS=(42 100 786)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON_ARGS=(
    --splits_csv "$SPLITS_CSV"
    --flat_dir   "$FLAT_DIR"
    --n_mels 64
    --dropout 0.05
    --warmup_epochs 70
    --mixup 0.2
)

run_one() {
    local variant="$1"
    local seed="$2"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  2_tcn_ablation  variant=$variant  seed=$seed"
    echo "════════════════════════════════════════════════════"

    conda run -n tf215_gpu python3 "$SCRIPT_DIR/2_tcn_ablation.py" \
        --variant "$variant" \
        "${COMMON_ARGS[@]}" \
        --random_seed "$seed"
}

FILTER="${1:-all}"

if [[ "$FILTER" == "all" || "$FILTER" == "2a" ]]; then
    echo ""
    echo "▶▶▶  2a: Baseline (8L, 64ch, k=3)"
    for seed in "${SEEDS[@]}"; do
        run_one "2a" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2b" ]]; then
    echo ""
    echo "▶▶▶  2b: Shallow (4L, 64ch, k=3)"
    for seed in "${SEEDS[@]}"; do
        run_one "2b" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2c" ]]; then
    echo ""
    echo "▶▶▶  2c: Wide (8L, 128ch, k=3)"
    for seed in "${SEEDS[@]}"; do
        run_one "2c" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2d" ]]; then
    echo ""
    echo "▶▶▶  2d: Deep (10L, 64ch, k=3)"
    for seed in "${SEEDS[@]}"; do
        run_one "2d" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2e" ]]; then
    echo ""
    echo "▶▶▶  2e: Kernel-2 (RF≈256)"
    for seed in "${SEEDS[@]}"; do
        run_one "2e" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2f" ]]; then
    echo ""
    echo "▶▶▶  2f: Kernel-5 (RF≈1021)"
    for seed in "${SEEDS[@]}"; do
        run_one "2f" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2g" ]]; then
    echo ""
    echo "▶▶▶  2g: No-residual (8L, 64ch, k=3)"
    for seed in "${SEEDS[@]}"; do
        run_one "2g" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2h" ]]; then
    echo ""
    echo "▶▶▶  2h: Lightweight (6L, 32ch, k=3)"
    for seed in "${SEEDS[@]}"; do
        run_one "2h" "$seed"
    done
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  All Series 2 runs complete."
echo "  Results: results_mygardenbird_2_$(python3 -c 'import platform; print(platform.system().lower())')/"
echo "════════════════════════════════════════════════════"
