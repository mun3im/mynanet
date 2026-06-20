#!/usr/bin/env bash
# ============================================================================
# Peer-benchmark training on MyGardenBird (12-class, 80:10:10, INT8)
# ============================================================================
# Trains MCU-class / lightweight-KWS peers under the same protocol as 1j so
# the paper's tab:comparison can report apples-to-apples accuracy and
# footprint numbers instead of literature-reported peer numbers on disjoint
# tasks.
#
# Tier-A peers (drop-in fair comparisons):
#   5a_matchboxnet_3x2x64.py       — NeMo MatchboxNet (existing 4f/4g/4h)
#   5b_dscnn_l_helloedge.py        — Hello-Edge DS-CNN-L (Zhang 2017)
#   5c_bcresnet8.py                — Broadcasted-Residual ResNet-8 (Kim 2021)
#   5d_tcresnet14.py               — TC-ResNet-14-1.5 (Choi 2019)
#
# Tier-B (bioacoustic-specialist peers; require porting effort first):
#   5e_wrennet.py                  — semi-learnable front-end + small CNN
#   5f_tinychirp_cnntime.py        — TinyChirp CNN-Time
#
# The .py scripts are NOT auto-created by this driver; they are expected to
# live alongside this script and follow the same canonical CLI as 1j-4e:
#   --splits_csv   "$SPLITS_CSV"
#   --flat_dir     "$FLAT_DIR"
#   --dropout      0.05
#   --warmup_epochs 70
#   --mixup        0.2
#   --random_seed  {42, 100, 786}
#
# Idempotency: each run is skipped if the expected results subdirectory
# contains a training_report.txt.
#
# Usage:
#   bash run_peer_benchmarks.sh                # all Tier-A peers × 3 seeds
#   bash run_peer_benchmarks.sh 5b             # only DS-CNN-L
#   bash run_peer_benchmarks.sh 5a 5c          # MatchboxNet + BC-ResNet
#   bash run_peer_benchmarks.sh --tier-b       # include Tier-B (if scripts present)
# ============================================================================

set -euo pipefail

SPLITS_CSV="/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
FLAT_DIR="/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
# SEEDS can be overridden via env var, e.g. SEEDS="42" bash run_peer_benchmarks.sh
if [[ -n "${SEEDS:-}" ]]; then
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
else
    SEEDS=(42 100 786)
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Per-peer .py scripts live in this directory (mynanet-benchmarking/).  The
# Tier-A .py scripts are forks of mobilenet-inspired/4a_squeezenet_v11.py
# with the architecture function replaced by one from peer_architectures.py.
RESULTS_ROOT="$SCRIPT_DIR/results_mygardenbird_5_$(python3 -c 'import platform; print(platform.system().lower())')"
CONDA_ENV="${CONDA_ENV:-tf215_gpu}"

# Canonical args (identical to 1j/4a–4e protocol).
COMMON_ARGS=(
    --splits_csv     "$SPLITS_CSV"
    --flat_dir       "$FLAT_DIR"
    --dropout        0.05
    --warmup_epochs  70
    --mixup          0.2
)

# bash 3.2 (macOS default) doesn't support associative arrays, so we use
# case statements instead.  This script works on both macOS bash 3.2 and
# modern Linux bash 4+.
script_of() {
    case "$1" in
        5a) echo "5a_matchboxnet_3x2x64.py" ;;
        5b) echo "5b_dscnn_l_helloedge.py" ;;
        5c) echo "5c_bcresnet8.py" ;;
        5d) echo "5d_tcresnet14.py" ;;
        5e) echo "5e_wrennet.py" ;;
        5f) echo "5f_tinychirp_cnntime.py" ;;
    esac
}

prefix_of() {
    case "$1" in
        5a) echo "5a_matchboxnet_3x2x64_drop05" ;;
        5b) echo "5b_dscnn_l_helloedge_drop05" ;;
        5c) echo "5c_bcresnet8_drop05" ;;
        5d) echo "5d_tcresnet14_drop05" ;;
        5e) echo "5e_wrennet_drop05" ;;
        5f) echo "5f_tinychirp_cnntime_drop05" ;;
    esac
}

TIER_A=(5a 5b 5c 5d)
TIER_B=(5e 5f)

run_one() {
    local variant="$1"
    local seed="$2"
    local script
    local prefix
    script=$(script_of "$variant")
    prefix=$(prefix_of "$variant")

    local result_dir="$RESULTS_ROOT/${prefix}_rand${seed}_warm70_mixup0.2_split80:10:10"
    if [[ -f "$result_dir/training_report.txt" ]]; then
        echo "  ↷ SKIP $variant seed=$seed (already exists)"
        return 0
    fi

    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        echo "  ✗ MISSING $SCRIPT_DIR/$script — skipping $variant (write the .py before running)" >&2
        return 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Peer $variant  seed=$seed  ($(basename "$script"))"
    echo "════════════════════════════════════════════════════"

    conda run -n "$CONDA_ENV" python3 "$SCRIPT_DIR/$script" \
        "${COMMON_ARGS[@]}" \
        --random_seed  "$seed"
}

# Parse args.
FILTERS=()
INCLUDE_TIER_B=0
for a in "$@"; do
    case "$a" in
        --tier-b) INCLUDE_TIER_B=1 ;;
        5a|5b|5c|5d|5e|5f) FILTERS+=("$a") ;;
        *) echo "Unknown arg: $a" >&2; exit 2 ;;
    esac
done
if [[ ${#FILTERS[@]} -eq 0 ]]; then
    FILTERS=("${TIER_A[@]}")
    if [[ "$INCLUDE_TIER_B" -eq 1 ]]; then
        FILTERS+=("${TIER_B[@]}")
    fi
fi

mkdir -p "$RESULTS_ROOT"

for variant in "${FILTERS[@]}"; do
    echo ""
    echo "▶▶▶ ${variant} (seeds: ${SEEDS[*]})"
    for seed in "${SEEDS[@]}"; do
        run_one "$variant" "$seed" || true
    done
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  Peer benchmark pass complete."
echo "  Results: $RESULTS_ROOT"
echo "════════════════════════════════════════════════════"
