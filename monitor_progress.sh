#!/bin/bash
# Monitor the progress of ablation experiments

echo "======================================================================="
echo "Ablation Experiment Progress Monitor"
echo "Started monitoring at: $(date)"
echo "======================================================================="
echo ""

# Check if summary file exists
if [ -f "ablation_summary.txt" ]; then
    echo "Current Summary:"
    cat ablation_summary.txt
    echo ""
fi

# Check running processes
echo "Running Python processes:"
pgrep -f "python3.*\.py" | wc -l
echo ""

# Check latest log activity
echo "Latest log activity (last 30 lines):"
if [ -d "ablation_logs" ]; then
    # Find the most recently modified log
    latest_log=$(ls -t ablation_logs/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "From: $latest_log"
        echo "---"
        tail -30 "$latest_log"
    else
        echo "No logs found yet"
    fi
else
    echo "Log directory not created yet"
fi

echo ""
echo "======================================================================="
echo "Monitor completed at: $(date)"
echo "======================================================================="
