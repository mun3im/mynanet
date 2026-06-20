#!/usr/bin/env bash
# Series 3 MobileNetV3Small width-multiplier ablation on the 12-class MyGardenBird dataset.
# Training from scratch (no pretrained weights — audio domain).
#
# Identical hyperparameters for direct comparison:
#   n_mels=64, dropout=0.05, warmup=70, mixup=0.2, split 80:10:10
#   seeds: 42, 100, 786
#
# Variants: 3a=width×0.5, 3b=width×0.75, 3c=width×1.0
# Total: 3 widths × 3 seeds = 9 runs
#
# Results land in: results_mygardenbird_3_{darwin|linux}/
#
# Usage:
#   bash run_series3_ablation.sh          # all 9 runs (3 widths × 3 seeds)
#   bash run_series3_ablation.sh 3a       # only width=0.5 (3 seeds)
#   bash run_series3_ablation.sh 3b       # only width=0.75 (3 seeds)
#   bash run_series3_ablation.sh 3c       # only width=1.0 (3 seeds)

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

CONDA_ENV="${CONDA_ENV:-tf215_cpu}"
RESULTS_ROOT="$SCRIPT_DIR/results_mygardenbird_3_$(python3 -c 'import platform; print(platform.system().lower())')"

run_one() {
    local width="$1"
    local seed="$2"

    local result_dir="$RESULTS_ROOT/3_mobilenetv3s_w${width}_mels64_drop05_rand${seed}_warm70_mixup0.2_split80:10:10_$(python3 -c 'import platform; print(platform.system().lower())')"
    if [[ -f "$result_dir/model_int8.tflite" ]]; then
        echo "  ↷ SKIP width=$width seed=$seed (already complete)"
        return 0
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  3_mobilenet_ablation  width_mult=$width  seed=$seed"
    echo "════════════════════════════════════════════════════"

    conda run -n "$CONDA_ENV" python3 "$SCRIPT_DIR/3_mobilenet_ablation.py" \
        --width_mult "$width" \
        "${COMMON_ARGS[@]}" \
        --random_seed "$seed"
}

FILTER="${1:-all}"

if [[ "$FILTER" == "all" || "$FILTER" == "3a" ]]; then
    echo ""
    echo "▶▶▶  3a: MobileNetV3Small width×0.5"
    for seed in "${SEEDS[@]}"; do
        run_one "0.5" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "3b" ]]; then
    echo ""
    echo "▶▶▶  3b: MobileNetV3Small width×0.75"
    for seed in "${SEEDS[@]}"; do
        run_one "0.75" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "3c" ]]; then
    echo ""
    echo "▶▶▶  3c: MobileNetV3Small width×1.0"
    for seed in "${SEEDS[@]}"; do
        run_one "1.0" "$seed"
    done
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  All Series 3 runs complete."
echo "  Results: results_mygardenbird_3_$(python3 -c 'import platform; print(platform.system().lower())')/"
echo "════════════════════════════════════════════════════"
