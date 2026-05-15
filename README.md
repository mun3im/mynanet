# MynaNet

Lightweight CNN for bird call classification targeting deployment on **Arduino Portenta H7** (Cortex-M7, 512 KB flash).

## Adopted Model: MynaNet = `1j` (MBV3-SE)

**`1j_mbv3_se.py`** is the production MynaNet model.

- Architecture: MobileNetV3-style inverted residual blocks with 5×5 depthwise convolutions and hard-sigmoid Squeeze-Excitation
- **94.91% INT8** mean accuracy (3 seeds, Linux/CUDA authoritative)
- **267 KB INT8** — well within the 512 KB H7 flash limit
- MCU-deployable: all ops supported by TFLite Micro on Portenta H7

## Dataset

**mygardenbird16khz** — 12 garden bird species, 16 kHz  
Fixed 80:10:10 train/val/test split (CSV-based, no leakage)

## Series 1 Ablation Scripts

All scripts share the same protocol: `n_mels=64`, `dropout=0.05`, `warmup=70`, `mixup=0.2`, seeds 42/100/786.

| Script | Model | INT8 KB | INT8 % (Linux, mean) | MCU |
|--------|-------|---------|----------------------|-----|
| `1a_baseline_2dcnn.py` | Baseline 2D CNN | 1629.9 | 93.33 | ✗ (3× over limit) |
| `1b_baseline_dscnn.py` | DS-CNN | 289.7 | 93.06 | ✓ |
| `1c_dscnn_se.py` | DS-CNN + SE | 376.9 | 93.61 | ✓ |
| `1d_dscnn_res.py` | DS-CNN + Residual | 294.8 | 93.10 | ✓ |
| `1e_dscnn_se_res.py` | DS-CNN + SE + Residual | 377.2 | 94.12 | ✓ |
| `1f_dscnn_se_res_wide.py` | DS-CNN + SE + Res + Wide | 455.3 | 93.61 | ✓ |
| `1g_dscnn_se_res_att.py` | DS-CNN + SE + Res + Attention | 371.8 | 94.21 | ✗ (BATCH_MATMUL) |
| `1h_dscnn_se_res_att_wide.py` | DS-CNN + SE + Res + Att + Wide | 529.3 | 93.89 | ✗ (over limit + BATCH_MATMUL) |
| `1i_mbv2_se.py` | MBV2 inverted residual + SE | 258.7 | 94.03 | ✓ |
| **`1j_mbv3_se.py`** ★ | **MynaNet — MBV3-SE (5×5 dw + hard-sigmoid SE)** | **267.1** | **94.91** | **✓** |
| `1k_mbv3_se_v2.py` | MBV3-SE v2 (stacked res, hard-swish, 5th block) | 650.0 | 93.75 | ✗ (over limit) |
| `1m_nanodepthwise.py` | NanoDepthwise-CNN (<50 KB) | ~50 | ~87 | ✓ |
| `1p_mbv3_se_hs.py` | MBV3-SE + Hard-Swish (blks 3–4) | 269.3 | 94.68 | ✓ |
| `1q_efficientnet_se.py` | EfficientNet-SE (se_ratio=0.25 on output ch) | 219.9 | 94.35 | ✓ |
| `1r_efficientnetb0.py` | EfficientNetB0 pretrained (accuracy ceiling) | 5126.6 | 91.58† | ✗ (10× over limit) |

> † 1r FP32 = 95.42% (dataset ceiling); INT8 drops 3.8 pp due to swish quantization  
> BATCH_MATMUL (used by Keras MultiHeadAttention) is not in the TFLite Micro op set on Portenta H7

## Training

```bash
# Train MynaNet (1j) — 3 seeds
bash run_seabird12_ablation.sh 1j

# Train a specific model
bash run_seabird12_ablation.sh 1e

# Train all Series 1 models
bash run_seabird12_ablation.sh
```

Results are written to `results_mygardenbird_1_{darwin|linux}/`.

## Key Constraint

`BATCH_MATMUL` (used by Keras `MultiHeadAttention`) is **not** in the TFLite Micro op set on Portenta H7. Models 1g and 1h are excluded from MCU deployment despite their accuracy.

## License

MIT
