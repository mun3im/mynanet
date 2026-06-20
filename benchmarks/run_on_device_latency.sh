#!/usr/bin/env bash
# ============================================================================
# On-device latency measurement (Cortex-M7 / Portenta H7)
# ============================================================================
# Addresses Reviewer-2 concern #4: §7 latency / SRAM / energy are analytical
# estimates "carried over from an earlier MAC-count profile of the DS-CNN
# predecessor".  The reviewer (correctly) won't accept this for a paper whose
# thesis is edge deployment.
#
# This script is a DOCUMENTED PLACEHOLDER for the ARGUS Smart Biologger
# flash-and-time workflow.  It cannot be run until the firmware is wired up.
# Once the firmware is ready, the steps below should be uncommented and the
# absolute paths to the ARGUS toolchain filled in.
#
# Required toolchain (none of which is auto-installable):
#   * Arduino CLI ≥ 0.35.0
#   * Portenta H7 board package (mbed_portenta) ≥ 4.x
#   * tflite-micro CMSIS-NN port (vendored in the ARGUS firmware repo)
#   * stm32CubeProgrammer or arduino-cli upload backend
#   * USB-serial port to read measurement output (e.g. /dev/tty.usbmodem*)
#
# Outputs expected from the device under test:
#   * mean ± std per-inference latency over N=100 inferences
#   * peak SRAM usage (from a fault-handler-style arena watermark)
#   * flash usage (.tflite size + tflite-micro runtime + app code)
#   * current draw (μA) at idle and during inference (requires DMM, separate)
#
# Outputs written here:
#   * h7_latency_run<ts>.csv
#   * h7_deployment_profile_seed42.md
# ============================================================================

set -euo pipefail

# === CONFIGURE THESE BEFORE RUNNING =========================================
ARGUS_FIRMWARE_DIR="${ARGUS_FIRMWARE_DIR:-/Users/mun3im/Dropbox/ARGUS_Deployment_Manual/firmware}"
PORT="${PORT:-/dev/tty.usbmodem*}"
BAUD="${BAUD:-115200}"
BOARD_FQBN="${BOARD_FQBN:-arduino:mbed_portenta:envie_m7}"
MODEL_TFLITE="${MODEL_TFLITE:-/Users/mun3im/Dropbox/Conda/mynanet/results_mygardenbird_1_linux/1j_mbv3_se_mels64_drop05_rand42_warm70_mixup0.2_split80:10:10_linux/model_int8.tflite}"
N_INFERENCES="${N_INFERENCES:-100}"
# ============================================================================

echo "════════════════════════════════════════════════════"
echo "  H7 on-device latency for 1j seed=42"
echo "════════════════════════════════════════════════════"

if [[ ! -f "$MODEL_TFLITE" ]]; then
    echo "✗ MISSING model: $MODEL_TFLITE" >&2; exit 1
fi
echo "✓ Model: $MODEL_TFLITE ($(stat -f%z "$MODEL_TFLITE" 2>/dev/null || stat -c%s "$MODEL_TFLITE") bytes)"

if [[ ! -d "$ARGUS_FIRMWARE_DIR" ]]; then
    cat <<EOF >&2

✗ ARGUS firmware directory not found at:
    $ARGUS_FIRMWARE_DIR

This script is a PLACEHOLDER for the on-device measurement workflow.
The required steps once the firmware is in place are:

  1. Convert the .tflite to a C array (xxd -i model_int8.tflite > model_data.cc)
     and drop it into the firmware project's tensors/ directory.
  2. Build:
       arduino-cli compile -b $BOARD_FQBN "$ARGUS_FIRMWARE_DIR"
  3. Flash:
       arduino-cli upload  -b $BOARD_FQBN -p \$PORT "$ARGUS_FIRMWARE_DIR"
  4. Read N=$N_INFERENCES latency samples over serial:
       arduino-cli monitor -p \$PORT -c baudrate=$BAUD \\
           | awk '/^LAT/' > h7_latency_run\$(date +%s).csv
  5. Parse → h7_deployment_profile_seed42.md with mean ± std latency,
     peak SRAM (from the firmware's arena-watermark print), and flash usage.

Until then: \$7 of the paper should state explicitly that latency is unmeasured.
EOF
    exit 2
fi

# Once firmware is in place, the actual commands go here (currently commented).
# arduino-cli compile -b "$BOARD_FQBN" "$ARGUS_FIRMWARE_DIR"
# arduino-cli upload  -b "$BOARD_FQBN" -p "$PORT" "$ARGUS_FIRMWARE_DIR"
# arduino-cli monitor -p "$PORT" -c baudrate=$BAUD | awk '/^LAT/' \
#     > h7_latency_run$(date +%s).csv

echo "(placeholder — fill in firmware path and uncomment build/flash/monitor)"
