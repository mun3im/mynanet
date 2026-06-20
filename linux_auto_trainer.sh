#!/bin/bash
# ============================================================================
# AUTO-TRAINER FOR LINUX LAB MACHINE (Behind NAT)
# ============================================================================
# This script runs on Linux machine via cron every minute
# It checks for a trigger file from Mac (synced via Dropbox)
# When trigger detected, starts training automatically
# ============================================================================

# Paths (UPDATE THESE FOR YOUR LINUX MACHINE)
MYNANET_DIR="$HOME/Dropbox/Conda/mynanet"
TRIGGER_FILE="$MYNANET_DIR/START_TRAINING_NOW.trigger"
TRAINING_SCRIPT="$MYNANET_DIR/run_linux_authoritative.sh"
STATUS_FILE="$MYNANET_DIR/LINUX_STATUS.txt"

# Navigate to working directory
cd "$MYNANET_DIR" || exit 1

# Update status (so Mac knows Linux agent is alive)
echo "Linux agent alive: $(date)" > "$STATUS_FILE.tmp"
echo "Hostname: $(hostname)" >> "$STATUS_FILE.tmp"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')" >> "$STATUS_FILE.tmp"
echo "Checking for trigger: $TRIGGER_FILE" >> "$STATUS_FILE.tmp"
mv "$STATUS_FILE.tmp" "$STATUS_FILE"

# Check if trigger file exists
if [ -f "$TRIGGER_FILE" ]; then
    echo "========================================" >> "$STATUS_FILE"
    echo "TRIGGER DETECTED! Starting training..." >> "$STATUS_FILE"
    echo "Time: $(date)" >> "$STATUS_FILE"
    echo "========================================" >> "$STATUS_FILE"

    # Remove trigger file immediately
    rm "$TRIGGER_FILE"

    # Update paths in training script (Linux-specific paths)
    # IMPORTANT: Update these paths for your Linux environment
    SPLITS_CSV="$HOME/Dropbox/Conda/SEAbird/seabird_validation/splits_csv/seabird_splits_80_10_10_seed42.csv"
    FLAT_DIR="/Volumes/Evo/seabird16khz_flat"  # UPDATE THIS PATH

    # Check if data paths exist
    if [ ! -f "$SPLITS_CSV" ]; then
        echo "ERROR: Splits CSV not found: $SPLITS_CSV" >> "$STATUS_FILE"
        echo "Please update SPLITS_CSV path in linux_auto_trainer.sh" >> "$STATUS_FILE"
        exit 1
    fi

    if [ ! -d "$FLAT_DIR" ]; then
        echo "ERROR: Flat directory not found: $FLAT_DIR" >> "$STATUS_FILE"
        echo "Please update FLAT_DIR path in linux_auto_trainer.sh" >> "$STATUS_FILE"
        exit 1
    fi

    # Update the training script with correct paths
    sed -i "s|SPLITS_CSV=.*|SPLITS_CSV=\"$SPLITS_CSV\"|" "$TRAINING_SCRIPT"
    sed -i "s|FLAT_DIR=.*|FLAT_DIR=\"$FLAT_DIR\"|" "$TRAINING_SCRIPT"

    echo "Updated training script with paths:" >> "$STATUS_FILE"
    echo "  SPLITS_CSV=$SPLITS_CSV" >> "$STATUS_FILE"
    echo "  FLAT_DIR=$FLAT_DIR" >> "$STATUS_FILE"

    # Activate conda environment
    echo "Activating conda environment..." >> "$STATUS_FILE"
    source "$HOME/miniconda3/etc/profile.d/conda.sh" || source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate tf215_gpu 2>&1 | tee -a "$STATUS_FILE"

    # Verify GPU
    echo "Checking GPU..." >> "$STATUS_FILE"
    python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))" 2>&1 | tee -a "$STATUS_FILE"

    # Make script executable
    chmod +x "$TRAINING_SCRIPT"

    # Start training in background
    echo "Launching training script..." >> "$STATUS_FILE"
    nohup "$TRAINING_SCRIPT" > authoritative_master.log 2>&1 &
    TRAINING_PID=$!

    # Save PID
    echo $TRAINING_PID > training_pid.txt

    echo "========================================" >> "$STATUS_FILE"
    echo "✓ TRAINING STARTED!" >> "$STATUS_FILE"
    echo "PID: $TRAINING_PID" >> "$STATUS_FILE"
    echo "Started: $(date)" >> "$STATUS_FILE"
    echo "Monitor: tail -f $MYNANET_DIR/authoritative_master.log" >> "$STATUS_FILE"
    echo "========================================" >> "$STATUS_FILE"

    # Create a confirmation file for Mac
    echo "Training started successfully!" > TRAINING_CONFIRMED.txt
    echo "PID: $TRAINING_PID" >> TRAINING_CONFIRMED.txt
    echo "Started: $(date)" >> TRAINING_CONFIRMED.txt
    echo "Estimated completion: $(date -d '+18 hours' 2>/dev/null || date -v+18H)" >> TRAINING_CONFIRMED.txt

    exit 0
fi

# No trigger found - just update status and exit
exit 0
