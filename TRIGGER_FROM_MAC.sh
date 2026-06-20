#!/bin/bash
# ============================================================================
# TRIGGER LINUX TRAINING FROM MAC
# ============================================================================
# Run this script on Mac to trigger training on Linux lab machine
# Works through Dropbox sync (no SSH needed)
# ============================================================================

MYNANET_DIR="$HOME/Dropbox/Conda/mynanet"
TRIGGER_FILE="$MYNANET_DIR/START_TRAINING_NOW.trigger"
STATUS_FILE="$MYNANET_DIR/LINUX_STATUS.txt"
CONFIRM_FILE="$MYNANET_DIR/TRAINING_CONFIRMED.txt"

cd "$MYNANET_DIR" || exit 1

echo "============================================================================"
echo "TRIGGERING LINUX TRAINING FROM MAC"
echo "============================================================================"
echo ""

# Check if Linux agent is alive
if [ -f "$STATUS_FILE" ]; then
    echo "Linux Agent Status:"
    cat "$STATUS_FILE"
    echo ""

    # Check how old the status is
    STATUS_AGE=$(( $(date +%s) - $(stat -f %m "$STATUS_FILE" 2>/dev/null || stat -c %Y "$STATUS_FILE") ))
    if [ $STATUS_AGE -gt 120 ]; then
        echo "⚠️  WARNING: Status file is $STATUS_AGE seconds old"
        echo "Linux agent might not be running. Status should update every minute."
        echo ""
    else
        echo "✓ Linux agent is alive (status updated $STATUS_AGE seconds ago)"
        echo ""
    fi
else
    echo "⚠️  WARNING: No status file found from Linux agent"
    echo "Make sure linux_auto_trainer.sh is running on Linux (via cron)"
    echo ""
fi

# Remove old confirmation file if exists
rm -f "$CONFIRM_FILE"

# Create trigger file
echo "Creating trigger file for Linux..."
echo "Triggered from Mac at $(date)" > "$TRIGGER_FILE"
echo "User: $(whoami)" >> "$TRIGGER_FILE"
echo "Mac hostname: $(hostname)" >> "$TRIGGER_FILE"

echo "✓ Trigger file created: $TRIGGER_FILE"
echo ""
echo "============================================================================"
echo "WAITING FOR LINUX TO RESPOND..."
echo "============================================================================"
echo "The Linux machine will:"
echo "  1. Detect trigger file (checks every minute)"
echo "  2. Update training script paths"
echo "  3. Start training in background"
echo "  4. Create TRAINING_CONFIRMED.txt"
echo ""
echo "Waiting up to 2 minutes for confirmation..."
echo ""

# Wait for confirmation (max 2 minutes)
for i in {1..24}; do
    if [ -f "$CONFIRM_FILE" ]; then
        echo ""
        echo "============================================================================"
        echo "✓ TRAINING STARTED ON LINUX!"
        echo "============================================================================"
        cat "$CONFIRM_FILE"
        echo ""
        echo "Monitor progress:"
        echo "  - Check: cat ~/Dropbox/Conda/mynanet/LINUX_STATUS.txt"
        echo "  - Logs: tail -f ~/Dropbox/Conda/mynanet/authoritative_master.log"
        echo "  - Individual: tail -f ~/Dropbox/Conda/mynanet/v1_seed42_linux.log"
        echo ""
        echo "Training will take ~18 hours (6 models × 3 hours each)"
        echo "============================================================================"
        exit 0
    fi

    # Show progress
    echo -n "."
    sleep 5
done

echo ""
echo ""
echo "============================================================================"
echo "⏳ NO CONFIRMATION YET"
echo "============================================================================"
echo "This is normal if:"
echo "  - Linux agent hasn't run yet (runs every minute)"
echo "  - Dropbox is still syncing"
echo ""
echo "To verify:"
echo "  1. Wait another minute and check: cat ~/Dropbox/Conda/mynanet/TRAINING_CONFIRMED.txt"
echo "  2. Check Linux status: cat ~/Dropbox/Conda/mynanet/LINUX_STATUS.txt"
echo "  3. Check logs: tail ~/Dropbox/Conda/mynanet/authoritative_master.log"
echo ""
echo "If training doesn't start within 5 minutes:"
echo "  - Linux agent might not be set up (see LINUX_SETUP_INSTRUCTIONS.md)"
echo "  - Dropbox might not be syncing"
echo "  - Check Linux machine is on and connected"
echo "============================================================================"
