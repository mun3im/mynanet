# MynaNet — Ablation Scripts

Series 1 architecture ablation for the 12-class mygardenbird dataset.
All scripts share the same training protocol and can be run individually or via `run_seabird12_ablation.sh`.

## Protocol

- Dataset: mygardenbird16khz (12 classes, 16 kHz)
- Split: 80:10:10 train/val/test (CSV-based, no leakage)
- n_mels: 64, dropout: 0.05, warmup: 70 epochs, mixup: 0.2
- Seeds: 42, 100, 786
- Authoritative results: Linux/CUDA (RTX GPU)

## Scripts

| Script | Model | Description | INT8 KB | INT8 % (Linux mean) | MCU |
|--------|-------|-------------|---------|----------------------|-----|
| `1a_baseline_2dcnn.py` | 1a | Baseline 2D CNN | 1629.9 | 93.33 | ✗ (3× over limit) |
| `1b_baseline_dscnn.py` | 1b | DS-CNN | 289.7 | 93.06 | ✓ |
| `1c_dscnn_se.py` | 1c | DS-CNN + SE | 376.9 | 93.61 | ✓ |
| `1d_dscnn_res.py` | 1d | DS-CNN + Residual | 294.8 | 93.10 | ✓ |
| `1e_dscnn_se_res.py` | 1e | DS-CNN + SE + Residual | 377.2 | 94.12 | ✓ |
| `1f_dscnn_se_res_wide.py` | 1f | DS-CNN + SE + Res + Wide channels | 455.3 | 93.61 | ✓ |
| `1g_dscnn_se_res_att.py` | 1g | DS-CNN + SE + Res + Attention | 371.8 | 94.21 | ✗ (BATCH_MATMUL) |
| `1h_dscnn_se_res_att_wide.py` | 1h | DS-CNN + SE + Res + Att + Wide | 529.3 | 93.89 | ✗ (over limit + BATCH_MATMUL) |
| `1i_inverted_residual.py` | 1i | InvRes + SE (MobileNetV2-style) | 258.7 | 94.03 | ✓ |
| `1j_hardsigmoid_5x5dw.py` | 1j | InvRes + HardSE + 5×5 DW **(MynaNet)** | 267.1 | **94.91** | ✓ |
| `1k_stacked_residual_hswish.py` | 1k | 1j + stacked residuals, 5th block | 650.0 | 93.75 | ✗ (over limit) |
| `1l_hardswish_ablation.py` | 1l | 1j + hard-swish in blocks 3–4 | 269.3 | 94.68 | ✓ |
| `1m_se_stochdepth.py` | 1m | InvRes + output-channel SE (se_ratio=0.25) | 219.9 | 94.35 | ✓ |
| `1n_learnable_freq_warp.py` | 1n | WrenNet-inspired learnable frequency warp (1j blocks + 1o epilogue) | 225.0 | 93.28 | ✓ |

> MCU ✓ = TFLite Micro compatible on Portenta H7 (≤512 KB flash, no BATCH_MATMUL)

## Running

```bash
# Run a single model across all 3 seeds
bash run_seabird12_ablation.sh 1j \
  --flat_dir /path/to/mygardenbird16khz \
  --splits_csv /path/to/metadata16khz/splits_mip_80_10_10.csv

# Run all models
bash run_seabird12_ablation.sh \
  --flat_dir /path/to/mygardenbird16khz \
  --splits_csv /path/to/metadata16khz/splits_mip_80_10_10.csv
```

Results are written to `results_mygardenbird_1_{darwin|linux}/`.

## Key findings

- **Best MCU model: 1j** (MynaNet) — 94.91% INT8, 267 KB, fully H7-deployable
- **Smallest MCU: 1m** — 94.35% INT8, 220 KB (smallest by flash); **fastest MCU: 1i** — 94.03% INT8, 259 KB (23 min training)
- **Best DS-CNN: 1e** — 94.12% INT8, 377 KB (most consistent DS-CNN family)
- **Attention ceiling (1g)**: +0.09 pp over 1e, not H7-deployable (BATCH_MATMUL)
- **Width (1f)**: no gain without attention
- **Hard-swish (1l)**: −0.23 pp vs 1j; ReLU6-based hard-sigmoid in 1j quantizes better
- **EfficientNetB0 ceiling** (`benchmarks/effnetb0_pretrained_ceiling.py`, no longer under a 1-series ID): highest FP32 (95.42%) but INT8 collapses 3.8 pp — 267 KB 1j beats 5 MB pretrained model post-quantization
- **1n (WrenNet learnable freq warp)**: 93.28 ± 0.75% INT8 (5 seeds: 7/42/100/786/2020), 225.0 KB, H7-deployable — below 1j/1o/1l/1e, reassigned from the abandoned "1p" slot (2026-07-28)
