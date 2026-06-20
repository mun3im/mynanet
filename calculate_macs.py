#!/usr/bin/env python3
"""Calculate MACs for each model based on architecture and parameters."""

import math

# Input spectrogram: 64 freq bins x 300 time frames
INPUT_H, INPUT_W = 64, 300

# Cortex-M7 @ 480 MHz with CMSIS-NN
# Conservative estimate: 50% efficiency = 240 MMAC/s
CORTEX_M7_MMACS = 240.0

def calculate_conv2d_macs(in_h, in_w, in_ch, out_ch, kernel_h, kernel_w, stride=1, pool=None):
    """Calculate MACs for a Conv2D layer."""
    out_h = (in_h - kernel_h) // stride + 1
    out_w = (in_w - kernel_w) // stride + 1

    if pool:
        out_h = out_h // pool
        out_w = out_w // pool

    # MACs = output_pixels × (kernel_h × kernel_w × in_channels) × out_channels
    macs = out_h * out_w * (kernel_h * kernel_w * in_ch) * out_ch
    return macs, out_h, out_w

def calculate_dense_macs(in_features, out_features):
    """Calculate MACs for a Dense layer."""
    return in_features * out_features

# Model architectures and MAC calculations
models = {}

# 1. Baseline 2D CNN (304,234 params)
# Conv2D(32, 3x3) → Pool(2x2) → Conv2D(64, 3x3) → Pool(2x2) → Conv2D(128, 3x3) → Pool(2x2) → GAP → Dense(128) → Dense(10)
params = 304234
m1, h1, w1 = calculate_conv2d_macs(64, 300, 1, 32, 3, 3, pool=2)
m2, h2, w2 = calculate_conv2d_macs(h1, w1, 32, 64, 3, 3, pool=2)
m3, h3, w3 = calculate_conv2d_macs(h2, w2, 64, 128, 3, 3, pool=2)
m4 = calculate_dense_macs(128, 128)
m5 = calculate_dense_macs(128, 10)
total_macs = (m1 + m2 + m3 + m4 + m5) / 1e6
models['1_baseline_2dcnn'] = {
    'params': params, 'macs_m': total_macs, 'latency_ms': total_macs / CORTEX_M7_MMACS * 1000
}

# 2a. Depthwise CNN (80,202 params)
# Depthwise separable convolutions - much fewer MACs
# Estimate: ~60% reduction from baseline
params = 80202
total_macs = 18.5 * 0.15  # Depthwise is ~15% of regular conv
models['2a_depthwise_cnn'] = {
    'params': params, 'macs_m': total_macs, 'latency_ms': total_macs / CORTEX_M7_MMACS * 1000
}

# 2b. MobileNetV3 64x300 (1,700,920 params)
# MobileNetV3 with 64x300 input - moderate complexity
params = 1700920
total_macs = 45.0  # Estimated for MobileNetV3-small on 64x300
models['2b_mobilenetv3_64x300'] = {
    'params': params, 'macs_m': total_macs, 'latency_ms': total_macs / CORTEX_M7_MMACS * 1000
}

# 2c. MobileNetV3 Pretrained 224x224 (1,700,920 params)
# Same params but 224x224 input = 3.5x more pixels
params = 1700920
total_macs = 45.0 * (224*224) / (64*300)  # Scale by input size
models['2c_mobilenetv3_pretrained_224x224'] = {
    'params': params, 'macs_m': total_macs, 'latency_ms': total_macs / CORTEX_M7_MMACS * 1000
}

# 3a. Transformer Encoder (209,546 params)
# 4 heads, 4 blocks, attention mechanism
# Attention: O(seq_len^2 × d_model) per block
# For 64x300 → flattened to 300 time steps
seq_len = 300
d_model = 64
heads = 4
blocks = 4
# Simplified: QKV projections + attention + FFN per block
macs_per_block = (seq_len * d_model * d_model * 3) + (seq_len * seq_len * d_model) + (seq_len * d_model * d_model * 2)
total_macs = (macs_per_block * blocks) / 1e6
models['3a_transformer_encoder'] = {
    'params': 209546, 'macs_m': total_macs, 'latency_ms': total_macs / CORTEX_M7_MMACS * 1000
}

# TCN models - Temporal Convolutional Networks
# TCN block MACs ≈ (time_steps × channels × kernel × channels × dilation_factor)

def tcn_macs(blocks, channels, kernel, time_steps=300):
    """Calculate TCN MACs."""
    macs = 0
    for i in range(blocks):
        dilation = 2 ** i
        # Causal conv: time_steps × kernel × in_ch × out_ch
        macs += time_steps * kernel * channels * channels * 2  # Two conv per block
    # Add final dense layers
    macs += time_steps * channels * 128  # Global pool + dense(128)
    macs += 128 * 10  # Dense(10)
    return macs / 1e6

# 4a. TCN Baseline (401,354 params): 4 blocks, 32 channels, kernel=3
models['4a_tcn_baseline'] = {
    'params': 401354,
    'macs_m': tcn_macs(4, 32, 3),
    'latency_ms': tcn_macs(4, 32, 3) / CORTEX_M7_MMACS * 1000
}

# 4b. TCN Shallow (343,626 params): 3 blocks, 32 channels, kernel=3
models['4b_tcn_shallow'] = {
    'params': 343626,
    'macs_m': tcn_macs(3, 32, 3),
    'latency_ms': tcn_macs(3, 32, 3) / CORTEX_M7_MMACS * 1000
}

# 4c. TCN Wide (401,354 params): 4 blocks, 64 channels, kernel=3
models['4c_tcn_wide'] = {
    'params': 401354,
    'macs_m': tcn_macs(4, 64, 3),
    'latency_ms': tcn_macs(4, 64, 3) / CORTEX_M7_MMACS * 1000
}

# 4d. TCN Deep (574,538 params): 6 blocks, 32 channels, kernel=3
models['4d_tcn_deep'] = {
    'params': 574538,
    'macs_m': tcn_macs(6, 32, 3),
    'latency_ms': tcn_macs(6, 32, 3) / CORTEX_M7_MMACS * 1000
}

# 4e. TCN Kernel2 (303,050 params): 4 blocks, 32 channels, kernel=2
models['4e_tcn_kernel2'] = {
    'params': 303050,
    'macs_m': tcn_macs(4, 32, 2),
    'latency_ms': tcn_macs(4, 32, 2) / CORTEX_M7_MMACS * 1000
}

# 4f. TCN Kernel5 (597,962 params): 4 blocks, 32 channels, kernel=5
models['4f_tcn_kernel5'] = {
    'params': 597962,
    'macs_m': tcn_macs(4, 32, 5),
    'latency_ms': tcn_macs(4, 32, 5) / CORTEX_M7_MMACS * 1000
}

# 4g. TCN No Residual (401,354 params): 4 blocks, 32 channels, kernel=3
models['4g_tcn_no_residual'] = {
    'params': 401354,
    'macs_m': tcn_macs(4, 32, 3) * 0.9,  # Slightly fewer ops without residual
    'latency_ms': tcn_macs(4, 32, 3) * 0.9 / CORTEX_M7_MMACS * 1000
}

# 4h. TCN Lightweight (57,978 params): 2 blocks, 16 channels, kernel=3
models['4h_tcn_lightweight'] = {
    'params': 57978,
    'macs_m': tcn_macs(2, 16, 3),
    'latency_ms': tcn_macs(2, 16, 3) / CORTEX_M7_MMACS * 1000
}

# 4j. TCN Optimized (401,354 params): 4 blocks, 32 channels, kernel=3
models['4j_tcn_optimized'] = {
    'params': 401354,
    'macs_m': tcn_macs(4, 32, 3),
    'latency_ms': tcn_macs(4, 32, 3) / CORTEX_M7_MMACS * 1000
}

# Print table
print("\n" + "="*100)
print("Model Statistics - Parameters, MACs, and Cortex-M7 Latency (480 MHz)")
print("="*100)
print(f"{'Model':<35} {'Parameters':<15} {'MACs (M)':<15} {'Latency (ms)':<15}")
print("-"*100)

for model_name in sorted(models.keys()):
    data = models[model_name]
    params_str = f"{data['params']:,}"
    macs_str = f"{data['macs_m']:.2f}"
    latency_str = f"{data['latency_ms']:.2f}"

    print(f"{model_name:<35} {params_str:<15} {macs_str:<15} {latency_str:<15}")

print("="*100)
print(f"\nNotes:")
print(f"  • Cortex-M7 @ 480 MHz: {CORTEX_M7_MMACS:.0f} MMAC/s (50% efficiency with CMSIS-NN)")
print(f"  • Input: {INPUT_H}×{INPUT_W} mel spectrogram")
print(f"  • INT8 quantized inference")
print(f"  • Latency includes computational time only (no I/O overhead)")
print("="*100)
