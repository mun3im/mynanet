#!/bin/bash
# Update all model files to use fixed train/val/test directories

echo "=========================================================================="
echo "Updating all models for fixed train/val/test directory structure"
echo "Dataset: /Volumes/Evo/seabird16k with train/, val/, test/ subdirectories"
echo "Split ratio: 75:10:15"
echo "=========================================================================="
echo ""

# List of all model files
models=(
    "1_baseline_2dcnn.py"
    "5d_depthwise_cnn.py"
    "2b_mobilenetv3_64x300.py"
    "2c_mobilenetv3_pretrained_224x224.py"
    "3a_transformer_encoder.py"
    "4a_tcn_baseline.py"
    "4b_tcn_shallow.py"
    "4c_tcn_wide.py"
    "4d_tcn_deep.py"
    "4e_tcn_kernel2.py"
    "4f_tcn_kernel5.py"
    "4g_tcn_no_residual.py"
    "4h_tcn_lightweight.py"
    "4j_tcn_optimized.py"
)

for model in "${models[@]}"; do
    if [ ! -f "$model" ]; then
        echo "⚠️  Skipping $model (not found)"
        continue
    fi

    echo "Updating $model..."

    # 1. Update docstring
    sed -i 's/Data Split: 90 test \/ 60 val \/ 450 train per class (600 total)/Data Split: Fixed 75:10:15 (train\/val\/test) from dataset directories/g' "$model"

    # 2. Comment out or remove TEST_SIZE_PER_CLASS and VAL_SIZE_PER_CLASS
    sed -i 's/^TEST_SIZE_PER_CLASS = /# DEPRECATED: Now using fixed directories - TEST_SIZE_PER_CLASS = /g' "$model"
    sed -i 's/^VAL_SIZE_PER_CLASS = /# DEPRECATED: Now using fixed directories - VAL_SIZE_PER_CLASS = /g' "$model"

    # 3. Update dataset directory comment
    sed -i 's|DEFAULT_DATASET_DIR = "/Volumes/Evo/seabird16k"|DEFAULT_DATASET_DIR = "/Volumes/Evo/seabird16k"  # Fixed train/val/test subdirectories|g' "$model"

    echo "  ✓ Basic updates applied to $model"
done

echo ""
echo "=========================================================================="
echo "✅ Phase 1 complete: Basic updates applied to all files"
echo "=========================================================================="
echo ""
echo "⚠️  IMPORTANT: Manual update still required for data loading logic"
echo ""
echo "Next step: Apply the data loading template to replace per-class splitting"
echo "with directory-based loading. This requires careful code replacement."
echo ""
echo "Press Enter to continue with automated data loading fix, or Ctrl+C to stop..."
read

# Create backup before major changes
echo "Creating backup before data loading changes..."
mkdir -p backup_before_split_fix
for model in "${models[@]}"; do
    if [ -f "$model" ]; then
        cp "$model" "backup_before_split_fix/"
    fi
done
echo "✓ Backup saved to backup_before_split_fix/"
echo ""

echo "The data loading logic needs custom replacement for each model."
echo "Recommendation: Use the provided Python template to manually update the"
echo "load_dataset section (lines ~1000-1150) in each file."
echo ""
echo "=========================================================================="
echo "SUMMARY: Files updated for fixed directory structure"
echo "=========================================================================="
echo ""
echo "Updated models:"
for model in "${models[@]}"; do
    if [ -f "$model" ]; then
        echo "  ✓ $model"
    fi
done
echo ""
echo "Manual step required:"
echo "  Replace the data loading section (around line 1000-1150) with logic"
echo "  that loads from train/, val/, test/ subdirectories instead of splitting."
echo ""
echo "See DATA_LOADING_TEMPLATE.py for the replacement code."
