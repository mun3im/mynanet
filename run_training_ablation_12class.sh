#!/usr/bin/env bash
# ============================================================================
# Training-hyperparameter ablation on 12-class MyGardenBird (paper Table 3)
# ============================================================================
# Re-runs the augmentation × split-ratio ablation on the current 12-class
# dataset.  The numbers currently in tab:training_ablation are from the
# 10-class historical dataset and must be replaced.
#
# Model: 1j (MynaNet, MBV3-SE 5×5 dw) — the adopted architecture, so any
#        augmentation / split-ratio conclusion transfers directly to the
#        deployed model.
#
# Canonical arg set (identical across every cell in the table):
#   --splits_csv  $SPLIT_CSV          # varies per row
#   --flat_dir    $FLAT_DIR
#   --n_mels      64
#   --dropout     0.05
#   --warmup_epochs 70
#   --random_seed {42, 100, 786}
# Augmentation flag is mutually-exclusive:
#   --mixup       0.2         (mixup row)
#   --specaugment             (specaugment row)
#
# Cells in the ablation grid:
#   (mixup0.2, 80:10:10)   ← baseline
#   (mixup0.2, 75:10:15)
#   (mixup0.2, 70:15:15)
#   (specaugment, 80:10:10)
#
# Total: 4 cells × 3 seeds = 12 runs.
#
# Idempotency: each run is skipped if the expected result directory contains
# a training_report.txt.
#
# Usage:
#   bash run_training_ablation_12class.sh                 # all 12 runs
#   bash run_training_ablation_12class.sh mixup_80        # one cell × 3 seeds
#   bash run_training_ablation_12class.sh specaugment_80
# ============================================================================

set -euo pipefail

FLAT_DIR="/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
META_DIR="/Volumes/Evo/MYGARDENBIRD/metadata16khz"
SEEDS=(42 100 786)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="$SCRIPT_DIR/results_mygardenbird_1_$(python3 -c 'import platform; print(platform.system().lower())')"
CONDA_ENV="${CONDA_ENV:-tf215_gpu}"
SCRIPT_1J="$SCRIPT_DIR/develop/1j_mbv3_se.py"

# Canonical args shared by every cell.
COMMON_ARGS=(
    --flat_dir       "$FLAT_DIR"
    --n_mels         64
    --dropout        0.05
    --warmup_epochs  70
)

declare -A SPLIT_CSV
SPLIT_CSV["80:10:10"]="$META_DIR/splits_mip_80_10_10.csv"
SPLIT_CSV["75:10:15"]="$META_DIR/splits_mip_75_10_15.csv"
SPLIT_CSV["70:15:15"]="$META_DIR/splits_mip_70_15_15.csv"

run_cell() {
    local cell="$1"     # mixup_80 | mixup_75 | mixup_70 | specaugment_80
    local seed="$2"

    local aug_flag aug_dir_suffix split_key
    case "$cell" in
        mixup_80)      aug_flag=(--mixup 0.2); aug_dir_suffix="mixup0.2"; split_key="80:10:10" ;;
        mixup_75)      aug_flag=(--mixup 0.2); aug_dir_suffix="mixup0.2"; split_key="75:10:15" ;;
        mixup_70)      aug_flag=(--mixup 0.2); aug_dir_suffix="mixup0.2"; split_key="70:15:15" ;;
        specaugment_80) aug_flag=(--specaugment); aug_dir_suffix="specaugment"; split_key="80:10:10" ;;
        *) echo "Unknown cell: $cell" >&2; return 2 ;;
    esac

    local csv="${SPLIT_CSV[$split_key]}"
    local result_dir="$RESULTS_ROOT/1j_mbv3_se_mels64_drop05_rand${seed}_warm70_${aug_dir_suffix}_split${split_key}_$(python3 -c 'import platform; print(platform.system().lower())')"
    if [[ -f "$result_dir/model_int8.tflite" ]]; then
        echo "  ↷ SKIP $cell seed=$seed (already complete: $(basename "$result_dir"))"
        return 0
    fi

    if [[ ! -f "$SCRIPT_1J" ]]; then
        echo "  ✗ MISSING $SCRIPT_1J" >&2
        return 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  1j ablation cell=$cell  seed=$seed  split=$split_key  aug=$aug_dir_suffix"
    echo "════════════════════════════════════════════════════"

    conda run -n "$CONDA_ENV" python3 "$SCRIPT_1J" \
        --splits_csv  "$csv" \
        "${COMMON_ARGS[@]}" \
        "${aug_flag[@]}" \
        --random_seed "$seed"
}

# Cell filters from CLI (default = all 4).
CELLS=()
for a in "$@"; do
    case "$a" in
        mixup_80|mixup_75|mixup_70|specaugment_80) CELLS+=("$a") ;;
        *) echo "Unknown arg: $a" >&2; exit 2 ;;
    esac
done
if [[ ${#CELLS[@]} -eq 0 ]]; then
    CELLS=(mixup_80 mixup_75 mixup_70 specaugment_80)
fi

mkdir -p "$RESULTS_ROOT"

for cell in "${CELLS[@]}"; do
    echo ""
    echo "▶▶▶ cell=$cell (seeds: ${SEEDS[*]})"
    for seed in "${SEEDS[@]}"; do
        run_cell "$cell" "$seed" || true
    done
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  Training ablation pass complete."
echo "  Results: $RESULTS_ROOT"
echo "════════════════════════════════════════════════════"
