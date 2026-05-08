#!/usr/bin/env bash
# Full ablation chain (1a–1g) on the 12-class MyGardenBird dataset.
#
# Identical hyperparameters for direct comparison:
#   n_mels=64, dropout=0.05, warmup=70, mixup=0.2, split 80:10:10
#   seeds: 42, 100, 786
#
# Results land in: results_mygardenbird_1_{darwin|linux}/
#
# Usage:
#   bash run_seabird12_ablation.sh          # all 21 runs (7 models × 3 seeds)
#   bash run_seabird12_ablation.sh 1a       # only 1a (3 seeds)
#   bash run_seabird12_ablation.sh 1b       # only 1b (3 seeds)
#   bash run_seabird12_ablation.sh 1c       # only 1c (3 seeds)
#   bash run_seabird12_ablation.sh 1d       # only 1d (3 seeds)
#   bash run_seabird12_ablation.sh 1e       # only 1e (3 seeds)
#   bash run_seabird12_ablation.sh 1f       # only 1f (3 seeds)
#   bash run_seabird12_ablation.sh 1g       # only 1g (3 seeds)

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
    local script="$1"
    local label="$2"
    local seed="$3"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $label  seed=$seed"
    echo "════════════════════════════════════════════════════"

    conda run -n tf215_gpu python3 "$SCRIPT_DIR/$script" \
        "${COMMON_ARGS[@]}" \
        --random_seed "$seed"
}

FILTER="${1:-all}"

if [[ "$FILTER" == "all" || "$FILTER" == "1a" ]]; then
    echo ""
    echo "▶▶▶  1a: Baseline 2D CNN"
    for seed in "${SEEDS[@]}"; do
        run_one "1a_baseline_2dcnn.py" "1a_baseline_2dcnn" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1b" ]]; then
    echo ""
    echo "▶▶▶  1b: Baseline DS-CNN (no SE, no residual)"
    for seed in "${SEEDS[@]}"; do
        run_one "1b_baseline_dscnn.py" "1b_baseline_dscnn" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1c" ]]; then
    echo ""
    echo "▶▶▶  1c: DS-CNN + SE only (no residual)"
    for seed in "${SEEDS[@]}"; do
        run_one "1c_dscnn_se.py" "1c_dscnn_se" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1d" ]]; then
    echo ""
    echo "▶▶▶  1d: DS-CNN + Residual only (no SE)"
    for seed in "${SEEDS[@]}"; do
        run_one "1d_dscnn_res.py" "1d_dscnn_res" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1e" ]]; then
    echo ""
    echo "▶▶▶  1e: DS-CNN + SE + Residual (MCU compatible)"
    for seed in "${SEEDS[@]}"; do
        run_one "1e_dscnn_se_res.py" "1e_dscnn_se_res" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1f" ]]; then
    echo ""
    echo "▶▶▶  1f: DS-CNN + SE + Residual + Wide, no Attention (MCU compatible)"
    for seed in "${SEEDS[@]}"; do
        run_one "1f_dscnn_se_res_wide.py" "1f_dscnn_se_res_wide" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1g" ]]; then
    echo ""
    echo "▶▶▶  1g: DS-CNN + SE + Residual + Attention"
    for seed in "${SEEDS[@]}"; do
        run_one "1g_dscnn_se_res_att.py" "1g_dscnn_se_res_att" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1h" ]]; then
    echo ""
    echo "▶▶▶  1h: DS-CNN + SE + Residual + Attention + Wide"
    for seed in "${SEEDS[@]}"; do
        run_one "1h_dscnn_se_res_att_wide.py" "1h_dscnn_se_res_att_wide" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1i" ]]; then
    echo ""
    echo "▶▶▶  1i: MBV2-style Inverted Residual + SE (MCU compatible)"
    for seed in "${SEEDS[@]}"; do
        run_one "1i_mbv2_se.py" "1i_mbv2_se" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "1j" ]]; then
    echo ""
    echo "▶▶▶  1j: MBV3-SE (5x5 dw blocks 3-4, hard-sigmoid SE, MCU compatible)"
    for seed in "${SEEDS[@]}"; do
        run_one "1j_mbv3_se.py" "1j_mbv3_se" "$seed"
    done
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  All runs complete."
echo "  Results: results_mygardenbird_1_$(python3 -c 'import platform; print(platform.system().lower())')/"
echo "════════════════════════════════════════════════════"
