
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
## TABLE 1: Model Ablation — 10-class dataset (mels=64, drop=0.05, warmup=70, mixup=0.2, split 80:10:10, Linux/CUDA)

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

### TABLE 2: Per-seed results for Series 1

| Model | Description        | INT8 KB | Seed | Linux FP32 % | Linux INT8 % | Linux Runtime | macOS FP32 % | macOS INT8 % | macOS Runtime | MCU  |
| ----- | ------------------ | ------- | ---- | ------------ | ------------ | ------------- | ------------ | ------------ | ------------- | ---- |
| 1a    | Baseline 2D CNN    | 1629.9  | 42   | 93.75        | 93.61        | 28m           | 93.75        | 93.61        | 48m           | ✗†   |
| 1a    |                    |         | 100  | 92.92        | 93.19        | 27m           | 92.08        | 92.08        | 22m           |      |
| 1a    |                    |         | 786  | 93.33        | 93.19        | 28m           | 94.72        | 94.17        | 58m           |      |
| 1b    | DS-CNN             | 289.7   | 42   | 93.33        | 93.75        | 28m           | 93.61        | 93.75        | 41m           | ✓    |
| 1b    |                    |         | 100  | 93.33        | 93.06        | 29m           | 91.53        | 91.39        | 23m           |      |
| 1b    |                    |         | 786  | 92.22        | 92.36        | 29m           | 92.36        | 91.94        | 41m           |      |
| 1c    | DS+SE              | 376.9   | 42   | 94.72        | 94.44        | 39m           | 93.47        | 93.61        | 1h 06m        | ✓    |
| 1c    |                    |         | 100  | 93.89        | 93.33        | 39m           | 94.17        | 93.89        | 1h 05m        |      |
| 1c    |                    |         | 786  | 92.64        | 93.06        | 21m           | 94.17        | 94.03        | 1h 05m        |      |
| 1d    | DS+Res             | 294.8   | 42   | 93.06        | 92.50        | 31m           | 92.50        | 92.36        | 55m           | ✓    |
| 1d    |                    |         | 100  | 92.78        | 92.78        | 28m           | 92.36        | 92.50        | 53m           |      |
| 1d    |                    |         | 786  | 93.89        | 94.03        | 37m           | 91.67        | 91.94        | 31m           |      |
| 1e    | DS+SE+Res          | 377.2   | 42   | 94.86        | 94.44        | 40m           | 90.69        | 91.39        | 1h 07m        | ✓    |
| 1e    |                    |         | 100  | 93.75        | 93.75        | 40m           | 93.19        | 92.92        | 58m           |      |
| 1e    |                    |         | 786  | 93.75        | 94.17        | 40m           | 92.92        | 92.92        | 1h 07m        |      |
| 1f    | DS+SE+Res+Wide     | 455.3   | 42   | 94.03        | 94.03        | 1h 20m        | 93.61        | 93.47        | 2h 17m        | ✓    |
| 1f    |                    |         | 100  | 94.03        | 93.89        | 1h 20m        | 92.92        | 93.19        | 2h 24m        |      |
| 1f    |                    |         | 786  | 93.33        | 92.92        | 1h 07m        | 91.67        | 91.53        | 1h 16m        |      |
| 1g    | DS+SE+Res+Att      | 371.8   | 42   | 94.31        | 93.89        | 31m           | 94.58        | 94.44        | 39m           | ✗    |
| 1g    |                    |         | 100  | 93.89        | 94.03        | 25m           | 94.72        | 94.86        | 1h 10m        |      |
| 1g    |                    |         | 786  | 94.58        | 94.72        | 23m           | 94.72        | 94.86        | 38m           |      |
| 1h    | DS+SE+Res+Att+Wide | 529.3   | 42   | 94.17        | 93.75        | 1h 21m        | 94.86        | 95.00        | 2h 15m        | ✗‡   |
| 1h    |                    |         | 100  | 94.72        | 94.58        | 1h 17m        | 95.28        | 95.00        | 1h 46m        |      |
| 1h    |                    |         | 786  | 93.19        | 93.33        | 46m           | 95.83        | 95.42        | 2h 19m        |      |
| 1i    | MBV2 Inv.Res+SE    | 258.7   | 42   | 93.89        | 94.44        | 24m           | 94.31        | 94.58        | 43m           | ✓    |
| 1i    |                    |         | 100  | 94.31        | 94.17        | 24m           | 94.44        | 93.75        | 42m           |      |
| 1i    |                    |         | 786  | 93.47        | 93.47        | 20m           | 94.72        | 94.44        | 42m           |      |
| 1j    | MBV3-SE (5×5 dw)   | 267.1   | 42   | 94.03        | 94.31        | 52m           | 93.89        | 94.17        | 1h 20m        | ✓    |
| 1j    |                    |         | 100  | 95.14        | **95.42**    | 53m           | 94.72        | **95.14**    | 1h 15m        |      |
| 1j    |                    |         | 786  | 94.58        | 95.00        | 53m           | 93.89        | 94.17        | 34m           |      |
| 1k    | MBV3-SE v2         | 650.0‡‡ | 42   | 93.06        | 92.92        | 20m           |              |              |               | ✗‡‡  |
| 1k    |                    |         | 100  | 93.61        | 93.75        | 26m           | 95.56        | 95.42        | 36m           |      |
| 1k    |                    |         | 786  | 94.58        | 94.58        | 25m           |              |              |               |      |
| 1p    | MBV3-SE+HardSwish  | 269.3   | 42   | 95.42        | 95.42        | 27m           | 94.72        | 94.86        |               | ✓    |
| 1p    |                    |         | 100  | 94.86        | 94.72        | 21m           | 94.72        | 95.00        |               |      |
| 1p    |                    |         | 786  | 93.75        | 93.89        | 14m           | 93.33        | 93.47        |               |      |
| 1q    | EfficientNet       | 219.9   | 42   | 94.17        | 94.17        | 21m           |              |              |               |      |
| 1q    |                    |         | 100  | 93.75        | 93.75        | 15m           |              |              |               |      |
| 1q    |                    |         | 786  | 95.00        | 95.14        | 25m           |              |              |               |      |
| 1r    | EfficientNetB0 pt  | 5126.6  | 42   | 94.58        | 90.28†††     | 54m           |              |              |               | ✗†††|
| 1r    |                    |         | 100  | 96.11        | 92.92†††     | 1h 12m        |              |              |               |      |
| 1r    |                    |         | 786  | 95.56        | 91.53†††     | 58m           |              |              |               |      |


### TABLE 3: Summary (n=3 seeds) for Series 1

| Model | Description        | INT8 KB | Linux FP32 ±sd   | Linux INT8 ±sd   | Linux Runtime | macOS FP32 ±sd | macOS INT8 ±sd   | macOS Runtime | Δ prev MCU | MCU  |
| ----- | ------------------ | ------- | ---------------- | ---------------- | ------------- | -------------- | ---------------- | ------------- | ---------- | ---- |
| 1a    | Baseline 2D CNN    | 1629.9  | 93.33 ± 0.34     | 93.33 ± 0.20     | 28m           | 93.52 ± 1.34   | 93.29 ± 1.08     | 42m           | baseline   | ✗†   |
| 1b*   | DS-CNN             | 289.7   | 92.96 ± 0.52     | 93.06 ± 0.57     | 29m           | 92.50 ± 1.05   | 92.36 ± 1.23     | 35m           | −0.27      | ✓    |
| 1c    | DS+SE              | 376.9   | 93.75 ± 0.85     | 93.61 ± 0.60     | 33m           | 93.94 ± 0.40   | 93.84 ± 0.21     | 1h 05m        | +0.55      | ✓    |
| 1d    | DS+Res             | 294.8   | 93.24 ± 0.47     | 93.10 ± 0.67     | 32m           | 92.18 ± 0.44   | 92.27 ± 0.29     | 46m           | −0.51      | ✓    |
| 1e*   | DS+SE+Res          | 377.2   | 94.12 ± 0.52     | **94.12** ± 0.28 | 40m           | 92.27 ± 1.37   | 92.41 ± 0.88     | 1h 04m        | +1.02      | ✓    |
| 1f    | DS+SE+Res+Wide     | 455.3   | 93.80 ± 0.33     | 93.61 ± 0.49     | 1h 16m        | 92.73 ± 0.98   | 92.73 ± 1.05     | 1h 59m        | −0.51      | ✓    |
| 1g    | DS+SE+Res+Att      | 371.8   | 94.26 ± 0.28     | 94.21 ± 0.36     | 26m           | 94.67 ± 0.08   | 94.72 ± 0.24     | 49m           | —          | ✗    |
| 1h    | DS+SE+Res+Att+Wide | 529.3   | 94.03 ± 0.63     | 93.89 ± 0.52     | 1h 08m        | 95.32 ± 0.49   | 95.14 ± 0.24     | 2h 07m        | —          | ✗‡   |
| 1i    | MBV2 Inv.Res+SE    | 258.7   | 93.89 ± 0.34     | 94.03 ± 0.41     | 23m           | 94.49 ± 0.21   | **94.26** ± 0.44 | 42m           | +0.42      | ✓    |
| 1j*   | MBV3-SE (5×5 dw)   | 267.1   | 94.58 ± 0.55     | **94.91** ± 0.56 | 52m           | 94.17 ± 0.48   | **94.49** ± 0.57 | 1h 03m        | +0.88      | ✓    |
| 1k    | MBV3-SE v2         | 650.0‡‡ | 93.75 ± 0.63     | 93.75 ± 0.69     | 24m           | 94.82 ± 0.43   | **94.77** ± 0.50 | 50m           | —          | ✗‡‡  |
| 1p    | MBV3-SE+HardSwish  | 269.3   | 94.68 ± 0.69     | **94.68** ± 0.62 | 21m           |                |                  |               | −0.23      | ✓    |
| 1q    | EfficientNet       | 219.9   | 94.31 ± 0.64     | 94.35 ± 0.71     | 21m           |                |                  |               |            |      |
| 1r    | EfficientNetB0 pt  | 5126.6  | **95.42** ± 0.63 | 91.58††† ± 1.08  | 1h 01m        |                |                  |               | —          | ✗††† |


> † 1a: 1629.9 KB — 3× over H7 512 KB flash limit
> ‡ 1h: 529.3 KB — exceeds H7 512 KB limit; BATCH_MATMUL also unsupported
> ‡‡ 1k: 488 KB params but TFLite INT8 file = 650 KB (TFLite buffer overhead) — exceeds H7 512 KB flash; needs size reduction
> ††† 1r: EfficientNetB0 pretrained (ImageNet) — 5126 KB INT8 (~10× H7 limit); NOT MCU-deployable. Stage9 pipeline: mel→dB→p2/p98→efficientnet.preprocess_input. FP32 **95.42%** = dataset ceiling (+0.5 pp above 1j FP32). INT8 91.58% — **3.3 pp below 1j INT8 (94.91%)** — swish activations lose ~3.8 pp to quantization. Our 267KB custom model beats 5MB pretrained network post-quantization.
> MCU ✓ = TFLite Micro compatible (Portenta H7); ✗ = BATCH_MATMUL unsupported or exceeds flash
> Δ = INT8 delta vs 1a baseline (1a→1b→1c→1d→1e→1f→1i→1j chain); 1p Δ vs 1j
> Last updated: 2026-05-13 (1r Stage9 pipeline: 3 seeds Linux done; FP32 95.42% ceiling confirmed)

### Key findings (Linux/CUDA authoritative)

- **Best MCU model: 1j** (MBV3-SE 5×5 dw) — **94.91% INT8**, 267.1 KB, 52m — still best
- **1p (MBV3-SE+HardSwish)**: 94.68% INT8 Linux (3 seeds) — −0.23 pp vs 1j; hard-swish alone didn't improve; macOS pending
- **Best compact MCU: 1i** (MBV2 Inv.Res+SE) — 94.03% INT8, 258.7 KB, 23m — smallest + fastest MCU model
- **Best DS-CNN MCU: 1e** (DS+SE+Res) — 94.12% INT8, 377.2 KB — most consistent DS-CNN
- **1k (MBV3-SE v2)**: warm50/no-mixup 93.75% INT8; specaugment/warm70 **94.44% INT8** — better but TFLite 650 KB still over H7 limit
- **1f confirms width adds nothing** without attention — 93.61% INT8, same as 1c
- **Attention ceiling (1g)**: 94.21% INT8 — only +0.6 pp over 1e, not H7 deployable
- **1h underperforms 1g**: wider attention model doesn't help on this task
- **1j×80 macOS complete**: mean 94.03% FP32 / 93.98% INT8 — below 1j×64 (94.49%), confirms n_mels=64 best for 1j
- **1r (EfficientNetB0 pretrained, Stage9 pipeline)**: FP32 **95.42%** = dataset ceiling (+0.5 pp over 1j); INT8 91.58% — 3.3 pp below 1j INT8; 267KB 1j beats 5MB EffNetB0 post-quantization; swish activations lose ~3.8 pp to quantization

---
# MEL SWEEP — Series 1 (all platforms)
---
## Mel sweep: n_mels ∈ {48, 80, 96} × models × 3 seeds

Split: 80:10:10 · dropout=0.05 · warmup=70 · mixup=0.2
Baseline (n_mels=64) from Linux/CUDA authoritative run.
Platform column: L = Linux/CUDA (RTX GPU), M = macOS Metal (M4 Pro).

### Per-seed results

| Model | n_mels | Platform | Seed | Linux FP32 % | Linux INT8 % | Linux Runtime | macOS FP32 % | macOS INT8 % | macOS Runtime |
| ----- | ------ | -------- | ---- | ------------ | ------------ | ------------- | ------------ | ------------ | ------------- |
| 1a    | 48     | M        | 42   |              |              |               | 93.06        | 93.19        | 43m           |
| 1a    | 48     | M        | 100  |              |              |               | 93.75        | 93.61        | 40m           |
| 1a    | 48     | M        | 786  |              |              |               | 93.06        | 92.78        | 40m           |
| 1a    | 48     | L        | 42   | 92.92        | 93.06        | 24m           |              |              |               |
| 1a    | 48     | L        | 100  | 93.75        | 93.75        | 40m           |              |              |               |
| 1a    | 48     | L        | 786  | 92.64        | 92.50        | 22m           |              |              |               |
| 1a    | 64†    | L        | 42   | 93.75        | 93.61        | 28m           |              |              |               |
| 1a    | 64†    | L        | 100  | 92.92        | 93.19        | 27m           |              |              |               |
| 1a    | 64†    | L        | 786  | 93.33        | 93.19        | 28m           |              |              |               |
| 1a    | 80     | M        | 42   |              |              |               | 92.22        | 91.94        | 37m           |
| 1a    | 80     | M        | 100  |              |              |               | 89.58        | 89.31        | 35m           |
| 1a    | 80     | M        | 786  |              |              |               | 94.17        | 93.89        | 1h04m         |
| 1a    | 80     | L        | 42   | 92.50        | 92.50        | 53m           |              |              |               |
| 1a    | 80     | L        | 100  | 91.94        | 91.94        | 1h13m         |              |              |               |
| 1a    | 80     | L        | 786  | 92.36        | 92.50        | 1h12m         |              |              |               |
| 1b    | 64†    | L        | 42   | 93.33        | 93.75        | 28m           |              |              |               |
| 1b    | 64†    | L        | 100  | 93.33        | 93.06        | 29m           |              |              |               |
| 1b    | 64†    | L        | 786  | 92.22        | 92.36        | 29m           |              |              |               |
| 1b    | 80     | L        | 42   | 92.08        | 92.08        | 39m           |              |              |               |
| 1b    | 80     | L        | 100  | 91.53        | 91.25        | 38m           |              |              |               |
| 1b    | 80     | L        | 786  | 91.53        | 91.81        | 24m           |              |              |               |
| 1c    | 64†    | L        | 42   | 94.72        | 94.44        | 39m           |              |              |               |
| 1c    | 64†    | L        | 100  | 93.89        | 93.33        | 39m           |              |              |               |
| 1c    | 64†    | L        | 786  | 92.64        | 93.06        | 21m           |              |              |               |
| 1c    | 80     | L        | 42   | 93.47        | 93.33        | 52m           | 93.75        | 93.19        | 1h25m         |
| 1c    | 80     | L        | 100  | pending      | pending      | —             | 92.08        | 91.94        | 1h23m         |
| 1c    | 80     | L        | 786  | missing      | missing      | —             | 92.64        | 93.33        | 1h22m         |
| 1c    | 96     | M        | 42   |              |              |               | 92.64        | 92.78        | 1h59m         |
| 1c    | 96     | M        | 100  |              |              |               | 92.78        | 93.19        | 1h52m         |
| 1c    | 96     | M        | 786  |              |              |               | 92.50        | 92.64        | 1h42m         |
| 1e    | 48     | L        | 42   | 93.89        | 94.17        | 31m           |              |              |               |
| 1e    | 48     | L        | 100  | 94.17        | 94.03        | 31m           |              |              |               |
| 1e    | 48     | L        | 786  | 93.33        | 93.06        | 29m           |              |              |               |
| 1e    | 64†    | L        | 42   | 94.86        | 94.44        | 40m           |              |              |               |
| 1e    | 64†    | L        | 100  | 93.75        | 93.75        | 40m           |              |              |               |
| 1e    | 64†    | L        | 786  | 93.75        | 94.17        | 40m           |              |              |               |
| 1e    | 80     | L        | 42   | 93.19        | 93.19        | 1h04m         |              |              |               |
| 1e    | 80     | L        | 100  | 93.47        | 93.19        | 1h42m         |              |              |               |
| 1e    | 80     | L        | 786  | 91.81        | 91.81        | 49m           |              |              |               |
| 1i    | 48     | L        | 42   | 94.44        | 94.58        | 20m           |              |              |               |
| 1i    | 48     | L        | 100  | 95.69        | 95.56        | 20m           |              |              |               |
| 1i    | 48     | L        | 786  | 93.47        | 93.75        | 20m           |              |              |               |
| 1i    | 64†    | L        | 42   | 93.89        | 94.44        | 24m           |              |              |               |
| 1i    | 64†    | L        | 100  | 94.31        | 94.17        | 24m           |              |              |               |
| 1i    | 64†    | L        | 786  | 93.47        | 93.47        | 20m           |              |              |               |
| 1i    | 80     | L        | 42   | 93.61        | 93.61        | 31m           | 94.17        | 93.75        | 53m           |
| 1i    | 80     | L        | 100  | 92.50        | 93.19        | 31m           | 92.22        | 92.92        | 55m           |
| 1i    | 80     | L        | 786  | 93.75        | 93.61        | 31m           | 94.58        | 94.86        | 1h0m          |
| 1i    | 96     | M        | 42   |              |              |               | 93.61        | 93.61        | 1h2m          |
| 1i    | 96     | M        | 100  |              |              |               | 89.58        | 89.86        | 29m           |
| 1i    | 96     | M        | 786  |              |              |               | 92.50        | 92.92        | 1h12m         |
| 1j    | 48     | M        | 42   |              |              |               | 95.00        | 95.00        | 34m           |
| 1j    | 48     | M        | 100  |              |              |               | 95.14        | 95.28        | 34m           |
| 1j    | 48     | M        | 786  |              |              |               | 95.28        | 95.28        | 35m           |
| 1j    | 48     | L        | 42   | 93.75        | 93.89        | 35m           |              |              |               |
| 1j    | 48     | L        | 100  | 95.42        | 95.28        | 41m           |              |              |               |
| 1j    | 48     | L        | 786  | 94.17        | 94.44        | 37m           |              |              |               |
| 1j    | 64†    | L        | 42   | 94.03        | 94.31        | 52m           |              |              |               |
| 1j    | 64†    | L        | 100  | 95.14        | 95.42        | 53m           |              |              |               |
| 1j    | 64†    | L        | 786  | 94.58        | 95.00        | 53m           |              |              |               |
| 1j    | 80     | M        | 42   |              |              |               | 93.47        | 93.47        | 55m           |
| 1j    | 80     | M        | 100  |              |              |               | 94.17        | 94.17        | 1h02m         |
| 1j    | 80     | M        | 786  |              |              |               | 94.44        | 94.31        | 1h01m         |
| 1j    | 80     | L        | 42   | 94.58        | 94.44        | 1h05m         |              |              |               |
| 1j    | 80     | L        | 100  | 94.86        | 94.72        | 1h08m         |              |              |               |
| 1j    | 80     | L        | 786  | 94.44        | 94.31        | 47m           |              |              |               |

### Summary (n=3 seeds)

| Model | n_mels | Platform | Linux FP32 mean±sd | Linux INT8 mean±sd | Linux Runtime | macOS FP32 mean±sd | macOS INT8 mean±sd | macOS Runtime |
| ----- | ------ | -------- | ------------------ | ------------------ | ------------- | ------------------ | ------------------ | ------------- |
| 1a    | 48     | M        |                    |                    |               | 93.29 ± 0.40       | 93.19 ± 0.42       | 41m           |
| 1a    | 48     | L        | 93.10 ± 0.58       | 93.10 ± 0.63       | 29m           |                    |                    |               |
| 1a    | 64†    | L        | 93.33 ± 0.20       | 93.33 ± 0.20       | 28m           |                    |                    |               |
| 1a    | 80     | M        |                    |                    |               | 91.99 ± 2.34       | 91.71 ± 2.34       | 45m           |
| 1a    | 80     | L        | 92.27 ± 0.29       | 92.31 ± 0.32       | 1h06m         |                    |                    |               |
| 1b    | 64†    | L        | 92.96 ± 0.52       | 93.06 ± 0.57       | 29m           |                    |                    |               |
| 1b    | 80     | L        | 91.71 ± 0.32       | 91.71 ± 0.42       | 34m           |                    |                    |               |
| 1c    | 64†    | L        | 93.75 ± 0.85       | 93.61 ± 0.60       | 33m           |                    |                    |               |
| 1c    | 80     | L        | partial (1/3)      | partial (1/3)      | —             | 92.82 ± 0.85       | 92.82 ± 0.77       | 1h23m         |
| 1c    | 96     | M        | 92.64 ± 0.14       | 92.87 ± 0.29       | 1h51m         |                    |                    |               |
| 1e    | 48     | L        | 93.80 ± 0.43       | 93.75 ± 0.60       | 30m           |                    |                    |               |
| 1e    | 64†    | L        | 94.12 ± 0.52       | 94.12 ± 0.28       | 40m           |                    |                    |               |
| 1e    | 80     | L        | 92.82 ± 0.89       | 92.73 ± 0.80       | 1h12m         |                    |                    |               |
| 1i    | 48     | L        | 94.53 ± 1.11       | **94.63** ± 0.91   | 20m           |                    |                    |               |
| 1i    | 64†    | L        | 93.89 ± 0.34       | 94.03 ± 0.41       | 23m           |                    |                    |               |
| 1i    | 80     | L        | 93.29 ± 0.69       | 93.47 ± 0.24       | 31m           | 93.66 ± 1.26       | 93.84 ± 0.97       | 56m           |
| 1i    | 96     | M        |                    |                    |               | 91.90 ± 2.08       | 92.13 ± 2.00       | 54m           |
| 1j    | 48     | M        |                    |                    |               | 95.14 ± 0.14       | **95.19** ± 0.16   | 34m           |
| 1j    | 48     | L        | 94.45 ± 0.87       | 94.54 ± 0.70       | 38m           |                    |                    |               |
| 1j    | 64†    | L        | 94.58 ± 0.55       | **94.91** ± 0.56   | 52m           |                    |                    |               |
| 1j    | 80     | M        |                    |                    |               | 94.03 ± 0.40       | 93.98 ± 0.37       | 59m           |
| 1j    | 80     | L        | 94.63 ± 0.21       | 94.49 ± 0.21       | 1h00m         |                    |                    |               |

> † Baseline n_mels=64 from Linux/CUDA authoritative run
> L = Linux/CUDA (RTX GPU) · M = macOS Metal (M4 Pro) · macOS runtimes ~3–4× longer
> 1i×48 seed=100 outlier: 95.69% FP32 / 95.56% INT8 — high variance (±0.91%) limits confidence
> **1j×48 new best**: 95.19% INT8 mean ± 0.16 (macOS) — beats 1j×64 (94.91%) by +0.28 pp, very consistent
> **1j×80 macOS complete**: 93.98% INT8 mean — below 1j×64 (94.49%); confirms n_mels=64 best for 1j
> Trend across all models: 64 ≥ 48 > 80 > 96 — denser mels consistently hurt
> Last updated: 2026-05-11 (1j×80 M done; 1k 3×seeds L+M done; 1c×80 L seed 100 pending, seed 786 missing)


---
# Architecture Summary
---

## TABLE 12: Architecture Summary

| Model | Key difference                  | INT8 KB | INT8 % (Linux) | MCU           |     |
| ----- | ------------------------------- | ------- | -------------- | ------------- | --- |
| 1a    | Baseline 2D CNN                 | 1629.9  | 93.33          | ✗†            |     |
| 1b    | DS-CNN (no SE, no residual)     | 289.7   | 93.06          | ✓             |     |
| 1c    | + SE block                      | 376.9   | 93.61          | ✓             |     |
| 1d    | + Residual only                 | 294.8   | 93.10          | ✓             |     |
| 1e    | + SE + Residual                 | 377.2   | **94.12**      | ✓             |     |
| 1f    | + wider channels, no attention  | 455.3   | 93.61          | ✓             |     |
| 1g    | + Multi-Head Self-Attention     | 371.8   | 94.21          | ✗             |     |
| 1h    | 1g + wider channels             | 529.3   | 93.89          | ✗‡            |     |
| 1i    | MBV2 inverted residual + SE     | 258.7   | **94.03**      | ✓             |     |
| 1j    | MBV3-SE (5×5 dw + hard-sigmoid) | 267.1   | **94.91**      | ✓             |     |
| 1k    | +stacked res, hard-swish, blk5  | 650.0†† | 93.75–94.44 (L)       | ✗††  |     |
| 1p    | 1j + hard-swish in blks 3–4     | 269.3   | 94.68                 | ✓    |     |
| 1r    | EfficientNetB0 pretrained       | 5126.6  | FP32 **95.42** (ceiling); INT8 91.58†††| ✗†††|  |
| 3b–3d | MBV3Small native (64–80 mel)    | ~1998   | ~81            | ✗ (4× over)   |     |
| 3e–3f | MBV3Small width×0.75            | ~1270   | ~77–78         | ✗ (2.5× over) |     |

>- Best H7-deployable: **1j** (94.91% INT8, 267 KB) — new best; **1i** (94.03% INT8, 259 KB — compact+fast); **1e** (94.12% INT8, 377 KB — best DS-CNN)
>- **1p**: 94.68% INT8 (Linux) — hard-swish alone −0.23 pp vs 1j; macOS pending
>- **1k**: 93.75–94.44% INT8 (Linux, augmentation-dependent) — TFLite 650 KB over H7 limit
>- **1r**: EfficientNetB0 pretrained (Stage9 pipeline) — FP32 **95.42%** dataset ceiling (+0.5 pp over 1j); INT8 91.58% (3.3 pp below 1j INT8 due to swish); 5127 KB; NOT MCU-deployable
>- 1f: width alone adds nothing without attention · 1g/1h: attention ceiling ~94.2%, not H7-deployable
>- No MBV3Small variant fits H7 512 KB — smallest is width×0.75 at ~1270 KB
>- †† 1k TFLite INT8 = 650 KB (buffer overhead despite 488 KB param count) — ✗ H7 flash
>- ††† 1r TFLite INT8 = 5127 KB (~10× H7 limit); swish activations cause INT8 degradation (91.58% vs 95.42% FP32)
>- 2-series: TCN variants · 3-series: MobileNet family · 4-series: archival

---
# SCRIPTS
---

## Scripts — Series 1 (DS-CNN family, 12-class mygardenbird)
Results → `results_mygardenbird_1_{platform}/`. TCN scripts moved to `../tcn/`; MobileNet/SqueezeNet to `../mobilenet-inspired/`.
- `1a_baseline_2dcnn.py` — trains 1a (Baseline 2D CNN)
- `1b_baseline_dscnn.py` — trains 1b (DS-CNN)
- `1c_dscnn_se.py` — trains 1c (DS+SE, no residual)
- `1d_dscnn_res.py` — trains 1d (DS+Res, no SE)
- `1e_dscnn_se_res.py` — trains 1e (DS+SE+Res, best DS-CNN MCU model)
- `1f_dscnn_se_res_wide.py` — trains 1f (DS+SE+Res+Wide)
- `1g_dscnn_se_res_att.py` — trains 1g (DS+SE+Res+Att, not H7-deployable)
- `1h_dscnn_se_res_att_wide.py` — trains 1h (DS+SE+Res+Att+Wide, not H7-deployable)
- `1i_mbv2_se.py` — trains 1i (MBV2 inverted residual + SE, MCU compatible)
- `1j_mbv3_se.py` — trains 1j (MBV3-SE: 5×5 dw + hard-sigmoid SE, MCU compatible, best overall)
- `1k_mbv3_se_v2.py` — trains 1k (MBV3-SE v2: stacked residuals at 3b+5, hard-swish blks 3–5, 5th block; ~488 KB INT8)
- `1m_nanodepthwise.py` — trains 1m (NanoDepthwise-CNN: 4-stage DW-sep, 32→64→128→192ch, no skip connections; ~87% @ <50 KB INT8)
- `1p_mbv3_se_hs.py` — trains 1p (MBV3-SE + hard-swish: identical to 1j, blocks 3–4 use hard-swish instead of ReLU6; ~270 KB INT8)
- `1q_efficientnet_se.py` — trains 1q (EfficientNet-SE: 1j + EfficientNet SE on output channels se_ratio=0.25; ~153 KB INT8)
- `1r_efficientnetb0.py` — trains 1r (pretrained EfficientNetB0: Stage9 pipeline, n_mels=224; FP32 95.42% ceiling, INT8 91.58%, NOT MCU-deployable, 5127 KB INT8)
- `run_seabird12_ablation.sh` — runs all experiments (1a–1r); supports per-model filter (1a, 1b, …, 1r)
- `run_mels_1c_1i.sh` — mel sweep: 1c+1i × n_mels ∈ {80,96} × 3 seeds
- `run_nmels_ablation.sh` — broader mel sweep: MCU models 1b–1e × {80,96} × 3 seeds
- `update_ablation_results.py` — parses results and updates this file

