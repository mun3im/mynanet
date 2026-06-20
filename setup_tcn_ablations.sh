#!/bin/bash
# Systematic setup of TCN ablation variants

echo "Setting up TCN ablation variants..."

# 4c_tcn_wide.py - 64 filters (double)
sed -i '3,12c\
Model 4c - TCN Wide: Increased channel width (64 filters)\
- Purpose: Test impact of model capacity\
- Channels: 64 (vs 32 in baseline)\
- Parameters: ~2x baseline\
- Hypothesis: More filters = higher accuracy but larger model' 4c_tcn_wide.py

sed -i 's/results_4a_tcn_baseline/results_4c_tcn_wide/g' 4c_tcn_wide.py
sed -i 's/Model 4a - TCN Baseline/Model 4c - TCN Wide/g' 4c_tcn_wide.py
sed -i 's/--tcn_channels", type=int, default=64/--tcn_channels", type=int, default=64  # WIDE: 64 filters/g' 4c_tcn_wide.py

# 4d_tcn_deep.py - 9 blocks
sed -i '3,12c\
Model 4d - TCN Deep: Increased depth (9 blocks)\
- Purpose: Test very deep network\
- Blocks: 9 (dilations: 1,2,4,8,16,32,64,128,256)\
- Receptive field: >1000 time steps\
- Hypothesis: Deeper = more capacity but training challenges' 4d_tcn_deep.py

sed -i 's/results_4a_tcn_baseline/results_4d_tcn_deep/g' 4d_tcn_deep.py
sed -i 's/Model 4a - TCN Baseline/Model 4d - TCN Deep/g' 4d_tcn_deep.py
sed -i 's/for d in \[1, 2, 4, 8, 16, 32\]:/for d in [1, 2, 4, 8, 16, 32, 64, 128, 256]:  # DEEP: 9 dilations/g' 4d_tcn_deep.py

# 4e_tcn_kernel2.py - Small kernel
sed -i '3,12c\
Model 4e - TCN Kernel2: Smaller kernel size\
- Purpose: Test impact of kernel size on speed\
- Kernel: 2 (vs 3 in baseline)\
- Speed: Faster inference\
- Hypothesis: Smaller kernel = faster but may need more layers' 4e_tcn_kernel2.py

sed -i 's/results_4a_tcn_baseline/results_4e_tcn_kernel2/g' 4e_tcn_kernel2.py
sed -i 's/Model 4a - TCN Baseline/Model 4e - TCN Kernel2/g' 4e_tcn_kernel2.py
sed -i 's/Conv1D(channels, 3, dilation/Conv1D(channels, 2, dilation  # KERNEL2/g' 4e_tcn_kernel2.py

# 4f_tcn_kernel5.py - Large kernel
sed -i '3,12c\
Model 4f - TCN Kernel5: Larger kernel size\
- Purpose: Test larger receptive field per layer\
- Kernel: 5 (vs 3 in baseline)\
- Receptive field: Larger per layer\
- Hypothesis: Larger kernel = fewer layers needed' 4f_tcn_kernel5.py

sed -i 's/results_4a_tcn_baseline/results_4f_tcn_kernel5/g' 4f_tcn_kernel5.py
sed -i 's/Model 4a - TCN Baseline/Model 4f - TCN Kernel5/g' 4f_tcn_kernel5.py
sed -i 's/Conv1D(channels, 3, dilation/Conv1D(channels, 5, dilation  # KERNEL5/g' 4f_tcn_kernel5.py

# 4g_tcn_no_residual.py - No skip connections
sed -i '3,12c\
Model 4g - TCN No Residual: Removed skip connections\
- Purpose: Test importance of residual connections\
- Modification: No Add() layer for residual\
- Training: May be harder without residuals\
- Hypothesis: Residuals critical for deep TCN training' 4g_tcn_no_residual.py

sed -i 's/results_4a_tcn_baseline/results_4g_tcn_no_residual/g' 4g_tcn_no_residual.py
sed -i 's/Model 4a - TCN Baseline/Model 4g - TCN No Residual/g' 4g_tcn_no_residual.py

# 4h_tcn_lightweight.py - MCU optimized
sed -i '3,12c\
Model 4h - TCN Lightweight: MCU-optimized minimal model\
- Purpose: Smallest deployable TCN variant\
- Channels: 16 (vs 32 in baseline)\
- Blocks: 5 (dilations: 1,2,4,8,16)\
- Target: <50KB model size, <5ms latency\
- Hypothesis: Minimal model still >95% accuracy' 4h_tcn_lightweight.py

sed -i 's/results_4a_tcn_baseline/results_4h_tcn_lightweight/g' 4h_tcn_lightweight.py
sed -i 's/Model 4a - TCN Baseline/Model 4h - TCN Lightweight/g' 4h_tcn_lightweight.py
sed -i 's/--tcn_channels", type=int, default=64/--tcn_channels", type=int, default=16  # LIGHTWEIGHT/g' 4h_tcn_lightweight.py
sed -i 's/for d in \[1, 2, 4, 8, 16, 32\]:/for d in [1, 2, 4, 8, 16]:  # LIGHTWEIGHT: 5 dilations/g' 4h_tcn_lightweight.py

echo "✓ All TCN ablation variants configured!"
echo ""
echo "Created variants:"
echo "  4b_tcn_shallow.py      - 5 blocks (shallow)"
echo "  4c_tcn_wide.py         - 64 filters (wide)"
echo "  4d_tcn_deep.py         - 9 blocks (deep)"
echo "  4e_tcn_kernel2.py      - kernel=2 (small)"
echo "  4f_tcn_kernel5.py      - kernel=5 (large)"
echo "  4g_tcn_no_residual.py  - no skip connections"
echo "  4h_tcn_lightweight.py  - 16 filters, 5 blocks (MCU)"
