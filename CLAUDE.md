
# MynaNet — Project Context for Claude

## Project
Bird sound classifier targeting deployment on **Arduino Portenta H7** (Cortex-M7).
The deployed model must use only ops supported by TFLite Micro on Portenta H7.

## Key Constraint
`BATCH_MATMUL` (used by Keras `MultiHeadAttention`) is **not** in the TFLite Micro op set
on Portenta H7. Only 1a–1e (no attention) produce compatible `.tflite` files.

## Dataset
- **mygardenbird16khz** — 12-class garden bird dataset, 16 kHz (expanded from 10; added Pied Fantail, Yellow-vented Bulbul)
- Fixed 80:10:10 train/val/test split (CSV-based, no leakage)
- Flat dir: `/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz`
- Splits CSV: `/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv`
- Also available: `splits_mip_75_10_15.csv`, `splits_mip_70_15_15.csv`

---
# ARCHIVED 10-CLASS ABLATION
---
## Model Ablation — 10-class dataset (mels=64, drop=0.05, warmup=70, mixup=0.2, split 80:10:10, Linux/CUDA)

| Model | Seed | FP32 % | INT8 % | Drop | MCU |
|-------|------|--------|--------|------|-----|
| 1c | 42 | 94.17 | 93.83 | -0.34 | ✓ |
| 1c | 100 | 94.17 | 95.00 | +0.83 | ✓ |
| 1c | 786 | 93.00 | 93.17 | +0.17 | ✓ |
| 1d | 42 | 94.67 | 94.67 | +0.00 | ✗ |
| 1d | 100 | 95.50 | 95.50 | +0.00 | ✗ |
| 1d | 786 | 96.33 | 96.17 | -0.16 | ✗ |
| 1e | 42 | 96.00 | 95.67 | -0.33 | ✗ |
| 1e | 100 | 96.83 | 97.00 | +0.17 | ✗ |
| 1e | 786 | 94.00 | 93.83 | -0.17 | ✗ |

---
# SERIES 1 -- 12-CLASS ABLATION — AUTHORITATIVE (Linux/CUDA, RTX GPU)
---
## Series 1 — DS-CNN family (12-class mygardenbird)

Platform: Linux x86_64 · Split: 80:10:10 · mels=64 · dropout=0.05 · warmup=70 · mixup=0.2

### Per-seed results

| Model | Description | INT8 KB | Seed | Linux FP32 % | Linux INT8 % | Linux Runtime | macOS FP32 % | macOS INT8 % | macOS Runtime | MCU |
|-------|-------------|---------|------|-------------|-------------|--------------|-------------|-------------|--------------|-----|
| 1a | Baseline 2D CNN | 1629.9 | 42 | 93.75 | 93.61 | 28m | 93.75 | 93.61 | 48m | ✗† |
| 1a | | | 100 | 92.92 | 93.19 | 27m | 92.08 | 92.08 | 22m | |
| 1a | | | 786 | 93.33 | 93.19 | 28m | 94.72 | 94.17 | 58m | |
| 1b | DS-CNN | 289.7 | 42 | 93.33 | 93.75 | 28m | 93.61 | 93.75 | 41m | ✓ |
| 1b | | | 100 | 93.33 | 93.06 | 29m | 91.53 | 91.39 | 23m | |
| 1b | | | 786 | 92.22 | 92.36 | 29m | 92.36 | 91.94 | 41m | |
| 1c | DS+SE | 376.9 | 42 | 94.72 | 94.44 | 39m | 93.47 | 93.61 | 1h 06m | ✓ |
| 1c | | | 100 | 93.89 | 93.33 | 39m | 94.17 | 93.89 | 1h 05m | |
| 1c | | | 786 | 92.64 | 93.06 | 21m | 94.17 | 94.03 | 1h 05m | |
| 1d | DS+Res | 294.8 | 42 | 93.06 | 92.50 | 31m | 92.50 | 92.36 | 55m | ✓ |
| 1d | | | 100 | 92.78 | 92.78 | 28m | 92.36 | 92.50 | 53m | |
| 1d | | | 786 | 93.89 | 94.03 | 37m | 91.67 | 91.94 | 31m | |
| 1e | DS+SE+Res | 377.2 | 42 | 94.86 | 94.44 | 40m | 90.69 | 91.39 | 1h 07m | ✓ |
| 1e | | | 100 | 93.75 | 93.75 | 40m | 93.19 | 92.92 | 58m | |
| 1e | | | 786 | 93.75 | 94.17 | 40m | 92.92 | 92.92 | 1h 07m | |
| 1f | DS+SE+Res+Wide (no Att) | 455.3 | 42 | 94.03 | 94.03 | 1h 20m | 93.61 | 93.47 | 2h 17m | ✓ |
| 1f | | | 100 | 94.03 | 93.89 | 1h 20m | 92.92 | 93.19 | 2h 24m | |
| 1f | | | 786 | 93.33 | 92.92 | 1h 07m | 91.67 | 91.53 | 1h 16m | |
| 1g | DS+SE+Res+Att | 371.8 | 42 | 94.31 | 93.89 | 31m | 94.58 | 94.44 | 39m | ✗ |
| 1g | | | 100 | 93.89 | 94.03 | 25m | 94.72 | 94.86 | 1h 10m | |
| 1g | | | 786 | 94.58 | 94.72 | 23m | 94.72 | 94.86 | 38m | |
| 1h | DS+SE+Res+Att+Wide | 529.3 | 42 | 94.17 | 93.75 | 1h 21m | 94.86 | 95.00 | 2h 15m | ✗‡ |
| 1h | | | 100 | 94.72 | 94.58 | 1h 17m | 95.28 | 95.00 | 1h 46m | |
| 1h | | | 786 | 93.19 | 93.33 | 46m | 95.83 | 95.42 | 2h 19m | |
| 1i | MBV2 Inv.Res+SE | 258.7 | 42 | 93.89 | 94.44 | 24m | 94.31 | 94.58 | 43m | ✓ |
| 1i | | | 100 | 94.31 | 94.17 | 24m | 94.44 | 93.75 | 42m | |
| 1i | | | 786 | 93.47 | 93.47 | 20m | 94.72 | 94.44 | 42m | |

### Summary (n=3 seeds)

| Model | Description             | INT8 KB | Linux FP32 ±sd | Linux INT8 ±sd   | Linux Runtime | macOS FP32 ±sd | macOS INT8 ±sd   | macOS Runtime | Δ prev MCU | MCU |
| ----- | ----------------------- | ------- | -------------- | ---------------- | ------------- | -------------- | ---------------- | ------------- | ---------- | --- |
| 1a    | Baseline 2D CNN         | 1629.9  | 93.33 ± 0.34   | 93.33 ± 0.20     | 28m           | 93.52 ± 1.34   | 93.29 ± 1.08     | 42m           | —          | ✗†  |
| 1b    | DS-CNN                  | 289.7   | 92.96 ± 0.52   | 93.06 ± 0.57     | 29m           | 92.50 ± 1.05   | 92.36 ± 1.23     | 35m           | baseline   | ✓   |
| 1c    | DS+SE                   | 376.9   | 93.75 ± 0.85   | 93.61 ± 0.60     | 33m           | 93.94 ± 0.40   | 93.84 ± 0.21     | 1h 05m        | +0.55      | ✓   |
| 1d    | DS+Res                  | 294.8   | 93.24 ± 0.47   | 93.10 ± 0.67     | 32m           | 92.18 ± 0.44   | 92.27 ± 0.29     | 46m           | −0.51      | ✓   |
| 1e    | DS+SE+Res               | 377.2   | 94.12 ± 0.52   | **94.12** ± 0.28 | 40m           | 92.27 ± 1.37   | 92.41 ± 0.88     | 1h 04m        | +1.02      | ✓   |
| 1f    | DS+SE+Res+Wide (no Att) | 455.3   | 93.80 ± 0.33   | 93.61 ± 0.49     | 1h 16m        | 92.73 ± 0.98   | 92.73 ± 1.05     | 1h 59m        | −0.51      | ✓   |
| 1g    | DS+SE+Res+Att           | 371.8   | 94.26 ± 0.28   | 94.21 ± 0.36     | 26m           | 94.67 ± 0.08   | 94.72 ± 0.24     | 49m           | —          | ✗   |
| 1h    | DS+SE+Res+Att+Wide      | 529.3   | 94.03 ± 0.63   | 93.89 ± 0.52     | 1h 08m        | 95.32 ± 0.49   | 95.14 ± 0.24     | 2h 07m        | —          | ✗‡  |
| 1i    | MBV2 Inv.Res+SE         | 258.7   | 93.89 ± 0.34   | 94.03 ± 0.41     | 23m           | 94.49 ± 0.21   | **94.26** ± 0.44 | 42m           | +0.42      | ✓   |
| 1j    | MBV3-SE (5×5 dw)        | ~270    | pending        | pending          | —             | —              | —                | —             | —          | ✓   |

> † 1a: 1629.9 KB — 3× over H7 512 KB flash limit
> ‡ 1h: 529.3 KB — exceeds H7 512 KB limit; BATCH_MATMUL also unsupported
> MCU ✓ = TFLite Micro compatible (Portenta H7); ✗ = BATCH_MATMUL unsupported or exceeds flash
> Δ prev MCU = INT8 delta vs previous MCU-compatible model (1b→1c→1d→1e→1f→1i chain)
> Last updated: 2026-05-07 (all 1a–1i complete)

### Key findings (Linux/CUDA authoritative)

- **Best MCU model: 1i** (MBV2 Inv.Res+SE) — 94.03% INT8, 258.7 KB, 23m runtime
- **Best MCU overall: 1e** (DS+SE+Res) — 94.12% INT8, 377.2 KB — marginally higher mean, more consistent
- **1f confirms width adds nothing** without attention — 93.61% INT8, same as 1c
- **Attention ceiling (1g)**: 94.21% INT8 — only +0.6 pp over 1e, not H7 deployable
- **1h underperforms 1g**: wider attention model doesn't help on this task
- **Mel sweep (macOS, COMPLETE)**: n_mels=80,96 for 1c+1i — n_mels=64 remains best for both; higher mels hurt (see mel sweep section)

---
## MEL SWEEP — Series 1 top MCU models (macOS Apple M4 Pro)
---
## Mel sweep: 1c + 1i × n_mels ∈ {80, 96} × 3 seeds

Platform: macOS (darwin) · Split: 80:10:10 · dropout=0.05 · warmup=70 · mixup=0.2
Baseline (n_mels=64) from Linux authoritative run shown for reference.

### Per-seed results

| Model | n_mels | INT8 KB | Seed | FP32 % | INT8 % | Runtime |
|-------|--------|---------|------|--------|--------|---------|
| 1c | 64† | 376.9 | 42 | 94.72 | 94.44 | 39m |
| 1c | 64† | | 100 | 93.89 | 93.33 | 39m |
| 1c | 64† | | 786 | 92.64 | 93.06 | 21m |
| 1c | 80 | 376.9 | 42 | 93.75 | 93.19 | 1h25m |
| 1c | 80 | | 100 | 92.08 | 91.94 | 1h23m |
| 1c | 80 | | 786 | 92.64 | 93.33 | 1h22m |
| 1c | 96 | 376.9 | 42 | 92.64 | 92.78 | 1h59m |
| 1c | 96 | | 100 | 92.78 | 93.19 | 1h52m |
| 1c | 96 | | 786 | 92.50 | 92.64 | 1h42m |
| 1i | 80 | 258.7 | 42 | 94.17 | 93.75 | 53m |
| 1i | 80 | | 100 | 92.22 | 92.92 | 55m |
| 1i | 80 | | 786 | 94.58 | 94.86 | 1h0m |
| 1i | 96 | 258.7 | 42 | 93.61 | 93.61 | 1h2m |
| 1i | 96 | | 100 | 89.58 | 89.86 | 29m |
| 1i | 96 | | 786 | 92.50 | 92.92 | 1h12m |

### Summary (n=3 seeds)

| Model | n_mels | INT8 KB | FP32 mean±sd | INT8 mean±sd | Runtime mean |
|-------|--------|---------|--------------|--------------|--------------|
| 1c | 64† | 376.9 | 93.75 ± 0.85 | 93.61 ± 0.60 | 33m |
| 1c | 80 | 376.9 | 92.82 ± 0.85 | 92.82 ± 0.77 | 1h23m |
| 1c | 96 | 376.9 | 92.64 ± 0.14 | 92.87 ± 0.29 | 1h51m |
| 1i | 64† | 258.7 | 93.89 ± 0.34 | 94.03 ± 0.41 | 23m |
| 1i | 80 | 258.7 | 93.66 ± 1.26 | 93.84 ± 0.97 | 56m |
| 1i | 96 | 258.7 | 91.90 ± 2.08 | 92.13 ± 2.00 | 54m |

> † Baseline n_mels=64 from Linux/CUDA authoritative run (not macOS)
> macOS runtimes longer (~3–4×) due to Metal vs CUDA
> Last updated: 2026-05-07 (COMPLETE — all 12 runs done)

---
# MEL SWEEP — Series 1 MCU models (Linux/CUDA, RTX GPU)
---
## Mel sweep: 1e/1i × n_mels ∈ {48, 80} × 3 seeds; 1b/1c/1d/1e × {80, 96} (broader)

Platform: Linux x86_64 · Split: 80:10:10 · dropout=0.05 · warmup=70 · mixup=0.2
Baseline (n_mels=64) from Linux authoritative run shown for reference.
Testing whether denser/sparser mel filterbanks improve the accuracy plateau at ~93-94%.
Updated incrementally as each model+mels 3-seed set completes.

### Per-seed results

| Model | n_mels | Seed | FP32 % | INT8 % | Runtime |
|-------|--------|------|--------|--------|---------|
| 1b | 64† | 42 | 93.33 | 93.75 | 28m |
| 1b | 64† | 100 | 93.33 | 93.06 | 29m |
| 1b | 64† | 786 | 92.22 | 92.36 | 29m |
| 1b | 80 | 42 | 92.08 | 92.08 | 39m |
| 1b | 80 | 100 | 91.53 | 91.25 | 38m |
| 1b | 80 | 786 | 91.53 | 91.81 | 24m |
| 1b | 96 | — | pending | pending | — |
| 1c | 64† | 42 | 94.72 | 94.44 | 39m |
| 1c | 64† | 100 | 93.89 | 93.33 | 39m |
| 1c | 64† | 786 | 92.64 | 93.06 | 21m |
| 1c | 80 | 42 | 93.47 | 93.33 | 52m |
| 1c | 80 | 100 | running | running | — |
| 1c | 80 | 786 | pending | pending | — |
| 1c | 96 | — | pending | pending | — |
| 1d | 64† | 42 | 93.06 | 92.50 | 31m |
| 1d | 64† | 100 | 92.78 | 92.78 | 28m |
| 1d | 64† | 786 | 93.89 | 94.03 | 37m |
| 1d | 80 | — | pending | pending | — |
| 1d | 96 | — | pending | pending | — |
| 1e | 48 | 42 | 93.89 | 94.17 | 31m |
| 1e | 48 | 100 | 94.17 | 94.03 | 31m |
| 1e | 48 | 786 | 93.33 | 93.06 | 29m |
| 1e | 64† | 42 | 94.86 | 94.44 | 40m |
| 1e | 64† | 100 | 93.75 | 93.75 | 40m |
| 1e | 64† | 786 | 93.75 | 94.17 | 40m |
| 1e | 80 | 42 | 93.19 | 93.19 | 1h04m |
| 1e | 80 | 100 | running | running | — |
| 1e | 80 | 786 | pending | pending | — |
| 1e | 96 | — | pending | pending | — |
| 1i | 48 | — | pending | pending | — |
| 1i | 64† | 42 | 93.89 | 94.44 | 24m |
| 1i | 64† | 100 | 94.31 | 94.17 | 24m |
| 1i | 64† | 786 | 93.47 | 93.47 | 20m |
| 1i | 80 | — | pending | pending | — |

### Summary (n=3 seeds, Linux/CUDA)

| Model | n_mels | FP32 mean±sd | INT8 mean±sd | Runtime |
|-------|--------|--------------|--------------|---------|
| 1b | 64† | 92.96 ± 0.52 | 93.06 ± 0.57 | 29m |
| 1b | 80 | 91.71 ± 0.32 | 91.71 ± 0.42 | 34m |
| 1b | 96 | pending | pending | — |
| 1c | 64† | 93.75 ± 0.85 | 93.61 ± 0.60 | 33m |
| 1c | 80 | partial (1/3) | partial (1/3) | — |
| 1c | 96 | pending | pending | — |
| 1d | 64† | 93.24 ± 0.47 | 93.10 ± 0.67 | 32m |
| 1d | 80 | pending | pending | — |
| 1d | 96 | pending | pending | — |
| 1e | 48 | 93.80 ± 0.43 | 93.75 ± 0.60 | 30m |
| 1e | 64† | 94.12 ± 0.52 | 94.12 ± 0.28 | 40m |
| 1e | 80 | partial (1/3) | partial (1/3) | — |
| 1e | 96 | pending | pending | — |
| 1i | 48 | pending | pending | — |
| 1i | 64† | 93.89 ± 0.34 | 94.03 ± 0.41 | 23m |
| 1i | 80 | pending | pending | — |

> † Baseline n_mels=64 from Linux authoritative Series 1 run
> All models MCU-deployable (TFLite Micro compatible, Portenta H7)
> Last updated: 2026-05-08 (1b×80, 1e×48 complete; 1c×80+1e×80 seed42 done; seed100 running)


---
# SERIES 2 — TCN variants (12-class mygardenbird, macOS Apple M4 Pro)
---
## Series 2 — TCN ablation chain

Platform: macOS (darwin) · Split: 80:10:10 · n_mels=64 · dropout=0.3 · warmup=50 · mixup=0.2
⚠ WARNING: Large FP32→INT8 drop observed — likely Metal quantization issue (same as Series 3). FP32 results more reliable.

### Per-seed results

| Model | Description     | INT8 KB | Seed | macOS FP32 % | macOS INT8 % | macOS Runtime | MCU        |
| ----- | --------------- | ------- | ---- | ------------ | ------------ | ------------- | ---------- |
| 2a    | TCN Baseline    | 526.8   | 42   | 60.83        | 45.56        | 38m           | ✗ (>512KB) |
| 2a    |                 |         | 100  | 67.36        | 39.17        | 35m           |            |
| 2a    |                 |         | 786  | 60.42        | 38.89        | 36m           |            |
| 2b    | TCN Shallow     | 452.6   | 42   | 55.42        | 28.33        | 29m           | ✓ (flash)  |
| 2b    |                 |         | 100  | 63.47        | 14.03        | 26m           |            |
| 2b    |                 |         | 786  | 66.67        | 33.06        | 28m           |            |
| 2c    | TCN No Residual | 524.4   | 42   | 74.31        | 55.83        | 36m           | ✗ (>512KB) |
| 2c    |                 |         | 100  | 78.06        | 45.00        | 36m           |            |
| 2c    |                 |         | 786  | 71.25        | 41.25        | 23m           |            |
| 2d    | TCN Lightweight | 133.1   | 42   | 40.69        | 32.92        | 15m           | ✓ (flash)  |
| 2d    |                 |         | 100  | 35.14        | 34.58        | 12m           |            |
| 2d    |                 |         | 786  | 41.53        | 26.94        | 12m           |            |
| 2e    | TCN Wide        | 526.8   | 42   | 38.06        | 19.44        | 36m           | ✗ (>512KB) |
| 2e    |                 |         | 100  | 64.03        | 30.00        | 35m           |            |
| 2e    |                 |         | 786  | 46.67        | 25.56        | 35m           |            |
| 2f    | TCN Deep        | 765.4   | 42   | 56.81        | 46.25        | 1h13m         | ✗ (>512KB) |
| 2f    |                 |         | 100  | 50.97        | 31.25        | 1h12m         |            |
| 2f    |                 |         | 786  | 46.94        | 22.50        | 59m           |            |
| 2g    | TCN Kernel=2    | 430.8   | 42   | 55.42        | 9.86         | 32m           | ✓ (flash)  |
| 2g    |                 |         | 100  | 54.03        | 16.39        | 32m           |            |
| 2g    |                 |         | 786  | 62.92        | 28.06        | 28m           |            |
| 2h    | TCN Kernel=5    | 718.8   | 42   | 65.00        | 20.14        | 36m           | ✗ (>512KB) |
| 2h    |                 |         | 100  | 67.50        | 24.72        | 36m           |            |
| 2h    |                 |         | 786  | 62.50        | 9.17         | 36m           |            |
| 2k    | TCN + SE        | 1480.1  | 42   | 43.19        | 13.47        | 54m           | ✗ (>512KB) |
| 2k    |                 |         | 100  | —            | —            | incomplete    |            |

### Summary (n=3 seeds unless noted)

| Model | Description     | INT8 KB | macOS FP32 ±sd   | macOS INT8 ±sd | macOS Runtime | MCU        |
| ----- | --------------- | ------- | ---------------- | -------------- | ------------- | ---------- |
| 2a    | TCN Baseline    | 526.8   | 62.87 ± 3.89     | 41.21 ± 3.77†  | 36m           | ✗ (>512KB) |
| 2b    | TCN Shallow     | 452.6   | 61.85 ± 5.80     | 25.14 ± 9.91†  | 28m           | ✓ (flash)  |
| 2c    | TCN No Residual | 524.4   | **74.54** ± 3.41 | 47.36 ± 7.57†  | 32m           | ✗ (>512KB) |
| 2d    | TCN Lightweight | 133.1   | 39.12 ± 3.47     | 31.48 ± 4.02†  | 13m           | ✓ (flash)  |
| 2e    | TCN Wide        | 526.8   | 49.59 ± 13.23    | 25.00 ± 5.30†  | 35m           | ✗ (>512KB) |
| 2f    | TCN Deep        | 765.4   | 51.57 ± 4.96     | 33.33 ± 12.01† | 1h08m         | ✗ (>512KB) |
| 2g    | TCN Kernel=2    | 430.8   | 57.46 ± 4.78     | 18.10 ± 9.22†  | 31m           | ✓ (flash)  |
| 2h    | TCN Kernel=5    | 718.8   | 65.00 ± 2.50     | 18.01 ± 7.99†  | 36m           | ✗ (>512KB) |
| 2k    | TCN + SE        | 1480.1  | (1/3 seeds)      | (1/3 seeds)    | —             | ✗ (>512KB) |
| 2c    | TCN No Residual (Linux/CUDA) | 524.4 | pending | pending | — | ✗ (>512KB) |
| 2l    | Pure 1D TCN (non-causal+res) | ~200  | pending | pending | — | ✓ (flash)  |

> † INT8 severely degraded on macOS Metal — FP32 is the reliable metric here
> MCU ✓ (flash) = fits H7 512 KB flash; TFLite Micro op compatibility for TCN not verified
> Best macOS FP32: 2c (no residual) = 74.54% — far below Series 1 MCU models (~93–94%)
> 2c Linux/CUDA: pending — running to verify CUDA ceiling (macOS Metal INT8 unreliable)
> 2l: pure 1D TCN (non-causal padding=same, residual, LayerNorm) — hypothesis: full context > causal
> Last updated: 2026-05-08 (2a–2h complete; 2k partial; 2c Linux + 2l pending after current jobs)


---
# SERIES 3 — MobileNet family (12-class mygardenbird, macOS Apple M4 Pro)
---
## Series 3 — MobileNetV3Small variants

Platform: macOS (darwin) · Split: 80:10:10 · dropout=0.2 · warmup=50 · mixup=0.2

### Per-seed results

| Model | Description | INT8 KB | Seed | macOS FP32 % | macOS INT8 % | macOS Runtime | MCU |
|-------|-------------|---------|------|-------------|-------------|--------------|-----|
| 3a | MBV3Small pretrained 224×224 | 1504.2 | 42 | 80.28 | 26.53† | 19m | ✗ (>512KB) |
| 3a | | | 100 | 70.97 | 24.44† | 11m | |
| 3a | | | 786 | 82.36 | 23.33† | 19m | |
| 3b | MBV3Small 64×300 native | 1997.9 | 42 | 84.17 | 84.31 | 29m | ✗ (>512KB) |
| 3b | | | 100 | 73.75 | 73.61 | 23m | |
| 3b | | | 786 | 85.97 | 85.97 | 27m | |
| 3c | MBV3Small 48×300 native | 1997.9 | 42 | 77.78 | 79.31 | 22m | ✗ (>512KB) |
| 3c | | | 100 | 78.33 | 78.33 | 23m | |
| 3c | | | 786 | 79.31 | 78.89 | 23m | |
| 3d | MBV3Small 80×300 native | 1997.9 | 42 | 84.58 | 84.31 | 35m | ✗ (>512KB) |
| 3d | | | 100 | 78.89 | 78.61 | 28m | |
| 3d | | | 786 | 80.69 | 80.97 | 34m | |
| 3e | MBV3Small width×0.75 64×300 | 1270.2 | 42 | 75.83 | 76.39 | 20m | ✗ (>512KB) |
| 3e | | | 100 | 78.33 | 78.06 | 27m | |
| 3e | | | 786 | 78.75 | 78.06 | 27m | |
| 3f | MBV3Small width×0.75 48×300 | 1270.2 | 42 | 76.11 | 75.42 | 22m | ✗ (>512KB) |
| 3f | | | 100 | 81.53 | 82.22 | 20m | |
| 3f | | | 786 | 75.28 | 75.14 | 22m | |

### Summary (n=3 seeds)

| Model | Description | INT8 KB | macOS FP32 ±sd | macOS INT8 ±sd | macOS Runtime | MCU |
|-------|-------------|---------|----------------|----------------|--------------|-----|
| 3a | MBV3Small pretrained 224×224 | 1504.2 | 77.87 ± 4.97 | 24.77† (broken) | 16m | ✗ (>512KB) |
| 3b | MBV3Small 64×300 native | 1997.9 | 81.30 ± 5.43 | 81.30 ± 5.26 | 26m | ✗ (>512KB) |
| 3c | MBV3Small 48×300 native | 1997.9 | 78.47 ± 0.62 | 78.84 ± 0.41 | 22m | ✗ (>512KB) |
| 3d | MBV3Small 80×300 native | 1997.9 | 81.39 ± 2.37 | 81.30 ± 2.35 | 32m | ✗ (>512KB) |
| 3e | MBV3Small width×0.75 64×300 | 1270.2 | 77.64 ± 1.28 | 77.50 ± 0.78 | 25m | ✗ (>512KB) |
| 3f | MBV3Small width×0.75 48×300 | 1270.2 | 77.64 ± 2.81 | 77.59 ± 3.25 | 21m | ✗ (>512KB) |

> † 3a INT8 collapsed: preprocess_input quantization mismatch in TFLite conversion. FP32 results valid. Rerun needed for correct INT8.
> 3a FP32 77.87% significantly below Stage9's 93% — likely finetune hyperparameter issue (lr=1e-5 too slow on Metal/macOS).
> 3b–3f: native MBV3Small from scratch. All far exceed H7 512 KB limit. Accuracy benchmark only.
> Best native resolution: 3d (80×300) = 81.39% FP32. Best overall: 3b/3d neck-and-neck. Narrower input (48mel) hurts.
> Width×0.75 (3e/3f) saves ~36% size (1270 vs 1998 KB) at cost of ~3–4 pp accuracy.
> Last updated: 2026-05-08 (3b–3f complete; 3a rerun needed for correct INT8)


---
# Architecture Summary
---

| Model | Key difference                  | INT8 KB | INT8 % (Linux) | MCU           |
| ----- | ------------------------------- | ------- | -------------- | ------------- |
| 1a    | Baseline 2D CNN                 | 1629.9  | 93.33          | ✗†            |
| 1b    | DS-CNN (no SE, no residual)     | 289.7   | 93.06          | ✓             |
| 1c    | + SE block                      | 376.9   | 93.61          | ✓             |
| 1d    | + Residual only                 | 294.8   | 93.10          | ✓             |
| 1e    | + SE + Residual                 | 377.2   | **94.12**      | ✓             |
| 1f    | + wider channels, no attention  | 455.3   | 93.61          | ✓             |
| 1g    | + Multi-Head Self-Attention     | 371.8   | 94.21          | ✗             |
| 1h    | 1g + wider channels             | 529.3   | 93.89          | ✗‡            |
| 1i    | MBV2 inverted residual + SE     | 258.7   | **94.03**      | ✓             |
| 1j    | MBV3-SE (5×5 dw + hard-sigmoid) | ~270    | pending        | ✓             |
| 3b–3d | MBV3Small native (64–80 mel)    | ~1998   | ~81            | ✗ (4× over)   |
| 3e–3f | MBV3Small width×0.75            | ~1270   | ~77–78         | ✗ (2.5× over) |

> Best H7-deployable: **1e** (94.12% INT8, 377 KB) or **1i** (94.03% INT8, 259 KB — smaller)
> 1f: width alone adds nothing without attention · 1g/1h: attention ceiling ~94.2%, not H7-deployable
> No MBV3Small variant fits H7 512 KB — smallest is width×0.75 at ~1270 KB
> 2-series: TCN variants (pending) · 3-series: MobileNet family · 4-series: archival

---
# SCRIPTS
---

## Scripts — Series 1 (DS-CNN ablation, 12-class mygardenbird)
- `1a_baseline_2dcnn.py` — trains 1a
- `1b_baseline_dscnn.py` — trains 1b
- `1c_dscnn_se.py` — trains 1c (DS+SE, no residual)
- `1d_dscnn_res.py` — trains 1d (DS+Res, no SE)
- `1e_dscnn_se_res.py` — trains 1e (MCU-compatible target)
- `1f_dscnn_se_res_wide.py` — trains 1f (wide + no attention, MCU compatible)
- `1g_dscnn_se_res_att.py` — trains 1g (+ attention)
- `1h_dscnn_se_res_att_wide.py` — trains 1h (attention + wide)
- `1i_mbv2_se.py` — trains 1i (MBV2 inverted residual + SE, MCU compatible)
- `1j_mbv3_se.py` — trains 1j (MBV3-SE: 5×5 dw in blocks 3-4 + hard-sigmoid SE, MCU compatible)
- `run_seabird12_ablation.sh` — runs all 27 experiments (9 models × 3 seeds)
- `run_mels_1c_1i.sh` — mel sweep: 1c+1i × n_mels ∈ {80,96} × 3 seeds (12 runs)
- `run_nmels_ablation.sh` — broader mel sweep: MCU models 1b–1e × {80,96} × 3 seeds
- `update_ablation_results.py` — parses results and updates this file

## Scripts — Series 2 (TCN variants, pending rerun on 12-class mygardenbird)
Old dataset: 10-class seabird. Scripts ported to 12-class mygardenbird paths; results pending.
- `2a_tcn_baseline.py` — TCN baseline (was 4a)
- `2b_tcn_shallow.py` — shallow TCN (was 4b)
- `2c_tcn_no_residual.py` — TCN no residual (was 4g)
- `2d_tcn_lightweight.py` — lightweight TCN (was 4h)
- `2e_tcn_wide.py` — wide TCN (was 4c)
- `2f_tcn_deep.py` — deep TCN (was 4d)
- `2g_tcn_kernel2.py` — kernel size 2 (was 4e)
- `2h_tcn_kernel5.py` — kernel size 5 (was 4f)
- `2i_tcn_specaugment.py` — + SpecAugment (was 4m)
- `2j_tcn_combined.py` — combined augmentation (was 4n)
- `2k_tcn_se.py` — + SE block (was 4o)
- `2l_pure1d_tcn.py` — **pure 1D TCN**: non-causal padding=same, residual, LayerNorm, 8 dilated blocks [1,2,4,8,16,32,1,2]
- `2m_tcn_optimized.py` — optimized variant (was 4j)
- `2n_tcn_mild_augmentation.py` — mild augmentation (was 4r)
- `2o_tcn_distillation.py` — knowledge distillation (was 4q)
- `2p_tcn_advanced_distillation.py` — advanced distillation (was 4s)
- `2q_tcn_optimal.py` — optimal config (was 4t)
- `2r_tcn_ultra.py` — ultra variant (was 4l)
- `2s_tcn_optimized_v2.py` — optimized v2 (was 4k)

## Scripts — Series 3 (MobileNet family, ordered by complexity, pending rerun on 12-class mygardenbird)
Scripts ported to 12-class mygardenbird paths; results pending. Runner: `run_mygardenbird_mobilenet_ablation.sh`
- `3a_mobilenetv3_pretrained_224x224.py` — MBV3Small pretrained ImageNet 224×224 (was 2c/3g; minimal mods, transfer learning)
- `3b_mobilenetv3_64x300.py` — MBV3Small 64-mel native input from scratch (was 2b/3a)
- `3c_mobilenetv3_48x300.py` — MBV3Small 48-mel native input (was 2d/3b)
- `3d_mobilenetv3_80x300.py` — MBV3Small 80-mel native input (was 2e/3c)
- `3e_mobilenetv3_width075_64x300.py` — MBV3Small width×0.75 64-mel (was 2f/3d)
- `3f_mobilenetv3_width075_48x300.py` — MBV3Small width×0.75 48-mel (was 2f1/3e)
- `3g_mobilenetv3_optimized.py` — MBV3 optimized + CSV splits (was 2f_optimized_csv/3f)
- `3h_mobilenetv1_narrow_64x300.py` — MBV1 narrow width 64-mel (was 7a)
- `3i_mobilenetv1_regularized_64x300.py` — MBV1 narrow + strong regularization (was 7b)
- `3j_mobilenetv1_narrow_regularized_v2.py` — MBV1 narrow regularized v2 (was 7c)
- `3k_mobilenetv1_width035.py` — MBV1 width×0.35 (was 10)
- `3l_mobilenetv2_narrow_64x300.py` — MBV2 narrow width 64-mel (was 8a)

## Scripts — Series 4 (Archival — other architectures, pending rerun on 12-class mygardenbird)
Miscellaneous architectures evaluated during exploration. Scripts ported to 12-class mygardenbird paths; results pending.
- `4a_squeezenet_v11.py` — SqueezeNet v1.1 (was 5a)
- `4b_shufflenetv2_2x.py` — ShuffleNetV2 width×2.0 (was 5b)
- `4c_depthwise_cnn.py` — Depthwise Separable CNN baseline (was 5d)
- `4d_ultralight_dscnn_wide.py` — Ultralight DS-CNN wide (was 6c)
- `4e_matchboxnet_64x300.py` — MatchboxNet-3×2×64 (was 9a)
- `4f_matchboxnet_regularized_64x300.py` — MatchboxNet + strong regularization (was 9b)
- `4g_matchboxnet_wider_64x300.py` — MatchboxNet-3×2×128 wider (was 9c)
- `4h_transformer_encoder.py` — Pure Transformer Encoder baseline (was 3a)
- `4i_tcn1d_dual_stage.py` — 1D TCN dual-stage classifier (was 11)
- `_retired_3a_transformer_encoder.py` — retired original (superseded by 4h)
