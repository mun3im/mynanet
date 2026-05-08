#!/usr/bin/env bash
# TCN ablation chain on the 12-class MyGardenBird dataset.
#
# Core TCN variants for comparison:
#   2a (baseline), 2b (shallow), 2c (no-residual), 2d (lightweight),
#   2e (wide), 2f (deep), 2g (kernel2), 2h (kernel5),
#   2k (SE), 2l (pure 1D non-causal residual), 2m (raw audio TCN), 2q (optimal)
#
# Identical hyperparameters for direct comparison:
#   n_mels=64, dropout=0.3, warmup=50, mixup=0.2, split 80:10:10
#   seeds: 42, 100, 786
#   (2m uses raw audio input — n_mels not applicable)
#
# Results land in: results_mygardenbird_2_{darwin|linux}/
#
# Usage:
#   bash run_mygardenbird_tcn_ablation.sh           # all 36 runs (12 models × 3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2a        # only 2a (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2b        # only 2b (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2c        # only 2c (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2d        # only 2d (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2e        # only 2e (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2f        # only 2f (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2g        # only 2g (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2h        # only 2h (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2k        # only 2k (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2l        # only 2l (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2m        # only 2m (3 seeds)
#   bash run_mygardenbird_tcn_ablation.sh 2q        # only 2q (3 seeds)

set -euo pipefail

SPLITS_CSV="/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
FLAT_DIR="/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SEEDS=(42 100 786)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON_ARGS=(
    --splits_csv "$SPLITS_CSV"
    --flat_dir   "$FLAT_DIR"
    --dropout 0.3
    --warmup_epochs 50
    --mixup 0.2
)
# Note: --n_mels is only supported by 2a; 2b-2s use N_MELS=64 constant (hard-coded).

run_one() {
    local script="$1"
    local label="$2"
    local seed="$3"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $label  seed=$seed"
    echo "════════════════════════════════════════════════════"

    conda run -n metal python3 "$SCRIPT_DIR/$script" \
        "${COMMON_ARGS[@]}" \
        --random_seed "$seed"
}

FILTER="${1:-all}"

if [[ "$FILTER" == "all" || "$FILTER" == "2a" ]]; then
    echo ""
    echo "▶▶▶  2a: TCN Baseline"
    for seed in "${SEEDS[@]}"; do
        run_one "2a_tcn_baseline.py" "2a_tcn_baseline" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2b" ]]; then
    echo ""
    echo "▶▶▶  2b: TCN Shallow (5 dilation blocks)"
    for seed in "${SEEDS[@]}"; do
        run_one "2b_tcn_shallow.py" "2b_tcn_shallow" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2c" ]]; then
    echo ""
    echo "▶▶▶  2c: TCN No Residual"
    for seed in "${SEEDS[@]}"; do
        run_one "2c_tcn_no_residual.py" "2c_tcn_no_residual" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2d" ]]; then
    echo ""
    echo "▶▶▶  2d: TCN Lightweight"
    for seed in "${SEEDS[@]}"; do
        run_one "2d_tcn_lightweight.py" "2d_tcn_lightweight" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2e" ]]; then
    echo ""
    echo "▶▶▶  2e: TCN Wide"
    for seed in "${SEEDS[@]}"; do
        run_one "2e_tcn_wide.py" "2e_tcn_wide" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2f" ]]; then
    echo ""
    echo "▶▶▶  2f: TCN Deep"
    for seed in "${SEEDS[@]}"; do
        run_one "2f_tcn_deep.py" "2f_tcn_deep" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2g" ]]; then
    echo ""
    echo "▶▶▶  2g: TCN Kernel Size 2"
    for seed in "${SEEDS[@]}"; do
        run_one "2g_tcn_kernel2.py" "2g_tcn_kernel2" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2h" ]]; then
    echo ""
    echo "▶▶▶  2h: TCN Kernel Size 5"
    for seed in "${SEEDS[@]}"; do
        run_one "2h_tcn_kernel5.py" "2h_tcn_kernel5" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2k" ]]; then
    echo ""
    echo "▶▶▶  2k: TCN + SE Block"
    for seed in "${SEEDS[@]}"; do
        run_one "2k_tcn_se.py" "2k_tcn_se" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2l" ]]; then
    echo ""
    echo "▶▶▶  2l: Pure 1D TCN (non-causal, residual, LayerNorm)"
    for seed in "${SEEDS[@]}"; do
        run_one "2l_pure1d_tcn.py" "2l_pure1d_tcn" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2m" ]]; then
    echo ""
    echo "▶▶▶  2m: 1D TCN on raw audio (causal, 48000 samples, ported from 4i)"
    for seed in "${SEEDS[@]}"; do
        run_one "2m_tcn1d_raw.py" "2m_tcn1d_raw" "$seed"
    done
fi

if [[ "$FILTER" == "all" || "$FILTER" == "2q" ]]; then
    echo ""
    echo "▶▶▶  2q: TCN Optimal"
    for seed in "${SEEDS[@]}"; do
        run_one "2q_tcn_optimal.py" "2q_tcn_optimal" "$seed"
    done
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  All runs complete."
echo "  Results: results_mygardenbird_2_$(python3 -c 'import platform; print(platform.system().lower())')/"
echo "════════════════════════════════════════════════════"
