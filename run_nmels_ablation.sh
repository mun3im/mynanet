#!/usr/bin/env bash
# n_mels ablation: MCU-compatible models × n_mels ∈ {48, 80} × 3 seeds
#
# Rationale: testing whether sparser/denser mel filterbanks improve ~93-95% plateau.
# 1a (baseline 2D CNN) and 1j (current best MCU) are the primary targets.
#
# Results land in: results_mygardenbird_1_{darwin|linux}/
# Output dirs encode n_mels (e.g. mels48_, mels80_) for easy comparison.
#
# Usage:
#   bash run_nmels_ablation.sh           # all runs
#   bash run_nmels_ablation.sh 48        # only n_mels=48
#   bash run_nmels_ablation.sh 80        # only n_mels=80
#   bash run_nmels_ablation.sh 1a        # only 1a (6 runs: 2 mel settings × 3 seeds)
#   bash run_nmels_ablation.sh 1b        # only 1b
#   bash run_nmels_ablation.sh 1c        # only 1c
#   bash run_nmels_ablation.sh 1d        # only 1d
#   bash run_nmels_ablation.sh 1e        # only 1e
#   bash run_nmels_ablation.sh 1i        # only 1i
#   bash run_nmels_ablation.sh 1j        # only 1j (6 runs: 2 mel settings × 3 seeds)

set -euo pipefail

SPLITS_CSV="/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
FLAT_DIR="/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SEEDS=(42 100 786)
NMELS_LIST=(48 80)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON_ARGS=(
    --splits_csv "$SPLITS_CSV"
    --flat_dir   "$FLAT_DIR"
    --dropout 0.05
    --warmup_epochs 70
    --mixup 0.2
)

run_one() {
    local script="$1"
    local label="$2"
    local seed="$3"
    local n_mels="$4"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $label  n_mels=$n_mels  seed=$seed"
    echo "════════════════════════════════════════════════════"

    conda run -n tf215_gpu python3 "$SCRIPT_DIR/$script" \
        "${COMMON_ARGS[@]}" \
        --n_mels "$n_mels" \
        --random_seed "$seed"
}

FILTER="${1:-all}"

for n_mels in "${NMELS_LIST[@]}"; do
    # Skip if a specific mel count was requested and doesn't match
    if [[ "$FILTER" =~ ^[0-9]+$ ]] && [[ "$FILTER" != "$n_mels" ]]; then
        continue
    fi

    if [[ "$FILTER" == "all" || "$FILTER" == "$n_mels" || "$FILTER" == "1a" ]]; then
        echo ""
        echo "▶▶▶  1a: Baseline 2D CNN  [n_mels=$n_mels]"
        for seed in "${SEEDS[@]}"; do
            run_one "1a_baseline_2dcnn.py" "1a_baseline_2dcnn" "$seed" "$n_mels"
        done
    fi

    if [[ "$FILTER" == "all" || "$FILTER" == "$n_mels" || "$FILTER" == "1b" ]]; then
        echo ""
        echo "▶▶▶  1b: Baseline DS-CNN  [n_mels=$n_mels]"
        for seed in "${SEEDS[@]}"; do
            run_one "1b_baseline_dscnn.py" "1b_baseline_dscnn" "$seed" "$n_mels"
        done
    fi

    if [[ "$FILTER" == "all" || "$FILTER" == "$n_mels" || "$FILTER" == "1c" ]]; then
        echo ""
        echo "▶▶▶  1c: DS-CNN + SE  [n_mels=$n_mels]"
        for seed in "${SEEDS[@]}"; do
            run_one "1c_dscnn_se.py" "1c_dscnn_se" "$seed" "$n_mels"
        done
    fi

    if [[ "$FILTER" == "all" || "$FILTER" == "$n_mels" || "$FILTER" == "1d" ]]; then
        echo ""
        echo "▶▶▶  1d: DS-CNN + Residual  [n_mels=$n_mels]"
        for seed in "${SEEDS[@]}"; do
            run_one "1d_dscnn_res.py" "1d_dscnn_res" "$seed" "$n_mels"
        done
    fi

    if [[ "$FILTER" == "all" || "$FILTER" == "$n_mels" || "$FILTER" == "1e" ]]; then
        echo ""
        echo "▶▶▶  1e: DS-CNN + SE + Residual (MCU target)  [n_mels=$n_mels]"
        for seed in "${SEEDS[@]}"; do
            run_one "1e_dscnn_se_res.py" "1e_dscnn_se_res" "$seed" "$n_mels"
        done
    fi

    if [[ "$FILTER" == "all" || "$FILTER" == "$n_mels" || "$FILTER" == "1i" ]]; then
        echo ""
        echo "▶▶▶  1i: MBV2 Inverted Residual + SE (MCU target)  [n_mels=$n_mels]"
        for seed in "${SEEDS[@]}"; do
            run_one "1i_mbv2_se.py" "1i_mbv2_se" "$seed" "$n_mels"
        done
    fi

    if [[ "$FILTER" == "all" || "$FILTER" == "$n_mels" || "$FILTER" == "1j" ]]; then
        echo ""
        echo "▶▶▶  1j: MBV3-SE 5×5 dw (current best MCU)  [n_mels=$n_mels]"
        for seed in "${SEEDS[@]}"; do
            run_one "1j_mbv3_se.py" "1j_mbv3_se" "$seed" "$n_mels"
        done
    fi
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  n_mels ablation complete."
echo "  Results: results_mygardenbird_1_$(conda run -n tf215_gpu python3 -c 'import platform; print(platform.system().lower())')/"
echo "════════════════════════════════════════════════════"
