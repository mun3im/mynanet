#!/bin/bash
# Quick completion checker

echo "Checking completion status..."
echo ""

# Count completed scripts
completed=0
failed=0
total=14

if [ -f "ablation_summary.txt" ]; then
    completed=$(grep -c "✓ SUCCESS:" ablation_summary.txt 2>/dev/null || echo "0")
    failed=$(grep -c "✗ FAILED:" ablation_summary.txt 2>/dev/null || echo "0")
fi

echo "Progress: $((completed + failed))/$total completed"
echo "  Success: $completed"
echo "  Failed: $failed"
echo "  Remaining: $((total - completed - failed))"
echo ""

# Check if still running
if pgrep -f "run_all_ablations" > /dev/null; then
    echo "Status: RUNNING"

    # Show current script
    current_log=$(ls -t ablation_logs/*.log 2>/dev/null | head -1)
    if [ -n "$current_log" ]; then
        echo "Current: $(basename $current_log .log)"

        # Show last line of current log
        last_line=$(tail -1 "$current_log" 2>/dev/null)
        echo "Latest: $last_line"
    fi
else
    echo "Status: COMPLETED or NOT RUNNING"
fi
