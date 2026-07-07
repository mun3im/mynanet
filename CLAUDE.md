
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

**Paper**: `/home/muneim/Dropbox/Paper4_MynaNet_EcolInf/mynanet_neurocomputing.tex` (Ecological Informatics). Compiles clean. *(The old R2 notes' `/Users/mun3im/...Paper4_Neurocomputing/` path is stale — this is the live file.)*
**Run env**: `tf215_gpu` (GPU) or `tf215_cpu` (CPU; note this env also has GPU TF — force CPU with `CUDA_VISIBLE_DEVICES=""`). Data live at `/Volumes/Evo/MYGARDENBIRD` on Linux, same path as macOS.

---
# ⏳ OUTSTANDING / PICK-UP-LATER
---

**Status (2026-06-17):** All Series 1/2/3 + training-hyperparameter ablation + LOSO complete (3-seed, Linux/CUDA). LOSO folded into paper (§6 `sec:loso`). Series 2 (TCN) and Series 3 (MBV3Small) 12-class runs now done — they **supersede** older cross-dataset numbers. Paper fold-in needed for I4/I7. **For any number, read the result folders, not the `.tex`.**

## Anticipated critical reviewer comments
(status: ✅ done · 🟡 partial · ❌ open — IDs map to the R2 Review Notes table lower in this file)

1. **Deployment latency 116 ms is estimated, not measured (B1/I6).** ✅ Fixed. Abstract, prose, and §7 notes updated to clarify: analytical H7 estimate 116 ms (upper bound from predecessor MAC count), measured CPU proxy 7.08 ms (single-threaded x86 INT8, 500 samples).
2. **Abstract number mismatches (I1/I2).** ✅ Fixed. Abstract + body text updated: Mixup-vs-SpecAugment +0.56pp (12-class); 2D-vs-1D +1.99pp (1j vs 2g, same-dataset, no cross-dataset caveat).
3. **Calibration claim without a metric (I3).** ✅ Fixed. Removed unsupported "improves calibration" claims from §2.5 (line 267) and §6.2 (line 923); rewrote to emphasize label smoothing robustness instead, which is well-documented in literature.
4. **MobileNetV3 baseline comparison was cross-platform (I4).** ✅ Fixed. Native 12-class MBV3 instability documented honestly: w0.5/w0.75 collapse (8.3%), w1.0 2/3 seeds converge (~93% INT8) but diverges on seed 42. Updated tab:final_comparison and §6.3 prose to report this as evidence that vision backbones don't transfer to MCU audio, supporting the narrative.
5. **TCN comparison was cross-dataset (I7).** ✅ Fixed. Replaced tab:tcn_ablation rows with 12-class Series 2 results (2a 88.84%, 2c 91.06%, 2g **92.92% INT8**); updated caption and prose to state 2g best at 1.99pp below 1j, same dataset.
6. **Table mixes 10-class and 12-class rows (I5).** ✅ Fixed. tab:final_comparison now 12-class only (Linux/CUDA); Vision rows report MBV3 instability per facts locked; TCN reduced to best run (2g); no 10-class rows remain.
7. **No source-disjoint / domain-shift evaluation (R2 #7).** ✅ DONE. MIP split proven source-disjoint (0 cross-partition sources) + 5-fold LOSO 94.06% ±0.98 INT8 (−0.85 pp, ns). Folded into §3 + §6 `sec:loso`.
8. **Positioning claim needs a qualifier (I8).** ✅ Fixed. Added "To our knowledge, " to line 273 of .tex.

## Experiments (Linux/CUDA, 3-seed unless noted)
- ✅ **Series 1** DS-CNN/InvRes family 1a–1n (authoritative tables below)
- ✅ **Series 2** TCN 2a–2h — best **2g (no-residual) 92.92% INT8**; 2c (wide) 91.06; 2a 88.84; all ≪ 1j
- ✅ **Series 3** MBV3Small-from-scratch w0.5/0.75/1.0 — **unstable** (see Facts locked)
- ✅ **Training ablation** 1j × {mixup,specaugment} × {80:10:10, 75:10:15, 70:15:15}, 12 runs
- ✅ **LOSO** 1j 5-fold source-disjoint, seed 42
- ⛔ Not run: real on-device H7 latency/energy measurement (hardware needed) — gates B1/I6

---
# 🔒 LOCKED MODEL HIGHLIGHTS — KEY VERIFIED NUMBERS
---

**MynaNet = model 1j** (InvRes + hard-sigmoid SE + 5×5 DW, MobileNetV3-style). **193,396 params.**

| Model | Role | INT8 KB | INT8 % (Linux, fixed split) | Notes |
|-------|------|---------|------------------------------|-------|
| **1j (MynaNet)** | **Deployed model** | **267.1** | **94.91 ± 0.56** | best H7-deployable; LOSO 94.06 ±0.98 |
| 1i | compact alt | 258.7 | 94.03 ± 0.41 | smallest + fastest MCU model |
| 1e | best DS-CNN | 377.2 | 94.12 ± 0.28 | most consistent DS-CNN |
| 2g | best TCN (1D) | — | 92.92 ± 0.11 | 1D ceiling; −1.99 pp vs 1j |
| 1n | EffNetB0 (pretrained) | 5126.6 | 91.58 (FP32 **95.42** ceiling) | not MCU; swish loses ~3.8 pp to INT8 |

- **1j training ablation (12-class):** mixup 80:10:10 = 94.91; 75:10:15 = **95.96**; 70:15:15 = 95.65; specaugment 80:10:10 = 94.35. → **mixup > specaugment by +0.56 pp**.
- **LOSO (1j, 5-fold):** 94.06 ± 0.98 INT8, −0.85 pp vs fixed split (within fold sd → ns).

## Facts locked (do not re-litigate)
- **MIP split is source-disjoint.** 7,200 clips from 1,381 Xeno-canto sources (up to 10 clips/source; only 126 sources are singletons). The `mip_cbc` solver puts every source wholly in one partition → **0 sources cross train/val/test**. The fixed-split 94.91% is therefore NOT leakage-inflated — *not* because of one-clip-per-file (false), but because of source isolation. This is the direct R2 #7 answer; LOSO corroborates.
- **MBV3Small-from-scratch is training-unstable on this task (Series 3, Linux).** w0.5 & w0.75 collapse to **8.3% = chance (1/12)** for all seeds; w1.0 converges for only 2/3 seeds (93.1/94.3% INT8) and collapses on seed 42 → mean 65.3 ±40.3. FP32 ≈ INT8 = **convergence failure, not a quantization artifact**. Strong evidence that vision backbones don't transfer to MCU audio (I4); verify it isn't an LR/code artifact before publishing.
- **Best 1D TCN = 2g (no-residual) 92.92% INT8**, below every InvRes/DS-CNN MCU model → 2D ≻ 1D on 12-class (same dataset now).
- **BATCH_MATMUL unsupported on H7 TFLite Micro** → attention models 1g/1h not deployable; only non-attention (1a–1f, 1i–1m) produce H7-compatible `.tflite`.
- **1n EffNetB0 FP32 95.42% = dataset ceiling** (+0.5 pp over 1j); our 267 KB 1j beats the 5 MB pretrained net after INT8.

---
# ✅ SUBMISSION-READY (2026-06-20, last updated 2026-06-20 11:35)
---
**All 8 reviewer items closed + ALL old dataset references removed + MatchBoxNet & YAMNet comparisons added.** Paper (21 pages, 560 KB PDF) compiled successfully with:
- Abstract: updated to 12-class numbers (mixup +0.56pp, 2D vs 1D +1.99pp)
- tab:training_ablation: now 12-class MyGardenBird (was historical 10-class); includes all 3 split ratios with results
- tab:tcn_ablation: 12-class Series 2 results (2g best at 92.92% INT8)
- tab:final_comparison: 12-class only (Linux/CUDA), MBV3 instability documented
- Latency: analytical (116 ms) vs. measured CPU proxy (7.08 ms) clearly labeled
- Calibration: unsupported claim removed
- Positioning: "To our knowledge" qualifier added
- Param count: standardized to 193,396
- **No references to 10-class or "historical" dataset remain in paper**
- All prose references (§2, §4, §6, §7) updated consistently to 12-class
- **NEW: MatchBoxNet transfer learning comparison (§6.2)** — domain-shift analysis showing MynaNet +14.9 pp advantage over GSC→birds transfer (1s→3s mismatch)
- **NEW: YAMNet transfer learning benchmark (§6.3)** — measured 65.97% vs MynaNet 94.91% (28.94 pp gap), validates domain-specific architecture over AudioSet transfer

---
# 🔒 BEFORE SUBMISSION — AUTHOR ACTION REQUIRED (ARCHIVED)
---

1. **Latency/energy (B1/I6):** either measure 1j on the Portenta H7 (M7) or relabel every latency/energy figure — abstract, §7, tables — as "estimated (predecessor MAC count)". No fabricated 116 ms as fact.
2. **Fold Series 2 + Series 3 into the paper (I4/I7):** replace historical/cross-dataset TCN + MBV3Small numbers with the 12-class Linux runs above; drop the cross-dataset caveats. Decide how to present the MBV3Small instability (it's a feature of the story, but state it honestly).
3. **Abstract numbers (I1/I2):** mixup−specaugment → +0.56 pp (12-class); 2D−1D → +1.99 pp (vs 2g, same dataset). Remove stale 0.83% / 2.2% / 2.47% figures.
4. **Calibration claim (I3):** add ECE/Brier or delete the "mixup improves calibration" sentences in §2.5/§6.2.
5. **Separate 10-class vs 12-class table rows (I5):** rule + footnote (or split) in `tab:final_comparison`; shade historical rows.
6. **Positioning qualifier (I8):** add "to our knowledge" to the systematic-comparison claim.
7. **Param-count consistency (m1):** standardise to **193,396** (not "193K"/"0.19 M"/"≈193,000") throughout.
8. **Verify figures render (m5):** confirm `1j_mbv3_se_architecture.png` shows the real architecture in the compiled PDF, not a placeholder box.

---
# SERIES 1 -- 12-CLASS ABLATION (Linux/CUDA, RTX GPU)
---
## Series 1 — DS-CNN family (12-class mygardenbird)

Platform: Linux x86_64 · Split: 80:10:10 · mels=64 · dropout=0.05 · warmup=70 · mixup=0.2 · 3 seeds (42, 100, 786)

### TABLE 2: Per-seed results for Series 1 — Linux/CUDA

| Model | Description        | INT8 KB | Seed | FP32 % | INT8 % | Runtime | MCU  |
| ----- | ------------------ | ------- | ---- | ------ | ------ | ------- | ---- |
| 1a    | Baseline 2D CNN    | 1629.9  | 42   | 93.75  | 93.61  | 28m     | ✗†   |
| 1a    |                    |         | 100  | 92.92  | 93.19  | 27m     |      |
| 1a    |                    |         | 786  | 93.33  | 93.19  | 28m     |      |
| 1b    | DS-CNN             | 289.7   | 42   | 93.33  | 93.75  | 28m     | ✓    |
| 1b    |                    |         | 100  | 93.33  | 93.06  | 29m     |      |
| 1b    |                    |         | 786  | 92.22  | 92.36  | 29m     |      |
| 1c    | DS+SE              | 376.9   | 42   | 94.72  | 94.44  | 39m     | ✓    |
| 1c    |                    |         | 100  | 93.89  | 93.33  | 39m     |      |
| 1c    |                    |         | 786  | 92.64  | 93.06  | 21m     |      |
| 1d    | DS+Res             | 294.8   | 42   | 93.06  | 92.50  | 31m     | ✓    |
| 1d    |                    |         | 100  | 92.78  | 92.78  | 28m     |      |
| 1d    |                    |         | 786  | 93.89  | 94.03  | 37m     |      |
| 1e    | DS+SE+Res          | 377.2   | 42   | 94.86  | 94.44  | 40m     | ✓    |
| 1e    |                    |         | 100  | 93.75  | 93.75  | 40m     |      |
| 1e    |                    |         | 786  | 93.75  | 94.17  | 40m     |      |
| 1f    | DS+SE+Res+Wide     | 455.3   | 42   | 94.03  | 94.03  | 1h 20m  | ✓    |
| 1f    |                    |         | 100  | 94.03  | 93.89  | 1h 20m  |      |
| 1f    |                    |         | 786  | 93.33  | 92.92  | 1h 07m  |      |
| 1g    | DS+SE+Res+Att      | 371.8   | 42   | 94.31  | 93.89  | 31m     | ✗    |
| 1g    |                    |         | 100  | 93.89  | 94.03  | 25m     |      |
| 1g    |                    |         | 786  | 94.58  | 94.72  | 23m     |      |
| 1h    | DS+SE+Res+Att+Wide | 529.3   | 42   | 94.17  | 93.75  | 1h 21m  | ✗‡   |
| 1h    |                    |         | 100  | 94.72  | 94.58  | 1h 17m  |      |
| 1h    |                    |         | 786  | 93.19  | 93.33  | 46m     |      |
| 1i    | InvRes+SE          | 258.7   | 42   | 93.89  | 94.44  | 24m     | ✓    |
| 1i    |                    |         | 100  | 94.31  | 94.17  | 24m     |      |
| 1i    |                    |         | 786  | 93.47  | 93.47  | 20m     |      |
| 1j    | InvRes+HardSE+5×5DW| 267.1   | 42   | 94.03  | 94.31  | 52m     | ✓    |
| 1j    |                    |         | 100  | 95.14  | **95.42** | 53m  |      |
| 1j    |                    |         | 786  | 94.58  | 95.00  | 53m     |      |
| 1k    | InvRes+HardSE+5×5DW+Stack| 650.0‡‡| 42  | 93.06  | 92.92  | 20m  | ✗‡‡  |
| 1k    |                    |         | 100  | 93.61  | 93.75  | 26m     |      |
| 1k    |                    |         | 786  | 94.58  | 94.58  | 25m     |      |
| 1l    | InvRes+HardSE+5×5DW+HSwish| 269.3| 42  | 95.42  | 95.42  | 27m     | ✓    |
| 1l    |                    |         | 100  | 94.86  | 94.72  | 21m     |      |
| 1l    |                    |         | 786  | 93.75  | 93.89  | 14m     |      |
| 1m    | InvRes+OutSE       | 219.9   | 42   | 94.17  | 94.17  | 21m     | ✓    |
| 1m    |                    |         | 100  | 93.75  | 93.75  | 15m     |      |
| 1m    |                    |         | 786  | 95.00  | 95.14  | 25m     |      |
| 1n    | EfficientNetB0 pt  | 5126.6  | 42   | 94.58  | 90.28†††     | 54m  | ✗†††|
| 1n    |                    |         | 100  | 96.11  | 92.92†††     | 1h 12m |     |
| 1n    |                    |         | 786  | 95.56  | 91.53†††     | 58m  |      |


### TABLE 3: Summary (n=3 seeds) for Series 1 — Linux/CUDA only

| Model | Description        | INT8 KB | FP32 ±sd         | INT8 ±sd         | Runtime | Δ prev MCU | MCU  |
| ----- | ------------------ | ------- | ---------------- | ---------------- | ------- | ---------- | ---- |
| 1a    | Baseline 2D CNN    | 1629.9  | 93.33 ± 0.34     | 93.33 ± 0.20     | 28m     | baseline   | ✗†   |
| 1b*   | DS-CNN             | 289.7   | 92.96 ± 0.52     | 93.06 ± 0.57     | 29m     | −0.27      | ✓    |
| 1c    | DS+SE              | 376.9   | 93.75 ± 0.85     | 93.61 ± 0.60     | 33m     | +0.55      | ✓    |
| 1d    | DS+Res             | 294.8   | 93.24 ± 0.47     | 93.10 ± 0.67     | 32m     | −0.51      | ✓    |
| 1e*   | DS+SE+Res          | 377.2   | 94.12 ± 0.52     | **94.12** ± 0.28 | 40m     | +1.02      | ✓    |
| 1f    | DS+SE+Res+Wide     | 455.3   | 93.80 ± 0.33     | 93.61 ± 0.49     | 1h 16m  | −0.51      | ✓    |
| 1g    | DS+SE+Res+Att      | 371.8   | 94.26 ± 0.28     | 94.21 ± 0.36     | 26m     | —          | ✗    |
| 1h    | DS+SE+Res+Att+Wide | 529.3   | 94.03 ± 0.63     | 93.89 ± 0.52     | 1h 08m  | —          | ✗‡   |
| 1i    | InvRes+SE          | 258.7   | 93.89 ± 0.34     | 94.03 ± 0.41     | 23m     | +0.42      | ✓    |
| 1j*   | InvRes+HardSE+5×5DW| 267.1   | 94.58 ± 0.55     | **94.91** ± 0.56 | 52m     | +0.88      | ✓    |
| 1k    | InvRes+HardSE+5×5DW+Stack| 650.0‡‡| 93.75 ± 0.63| 93.75 ± 0.69     | 24m     | —          | ✗‡‡  |
| 1l    | InvRes+HardSE+5×5DW+HSwish| 269.3| 94.68 ± 0.69| **94.68** ± 0.62 | 21m     | −0.23      | ✓    |
| 1m    | InvRes+OutSE       | 219.9   | 94.31 ± 0.64     | 94.35 ± 0.71     | 21m     | —         | ✓    |
| 1n    | EfficientNetB0 pt  | 5126.6  | **95.42** ± 0.63 | 91.58††† ± 1.08  | 1h 01m  | —          | ✗††† |


> † 1a: 1629.9 KB — 3× over H7 512 KB flash limit
> ‡ 1h: 529.3 KB — exceeds H7 512 KB limit; BATCH_MATMUL also unsupported
> ‡‡ 1k: 488 KB params but TFLite INT8 file = 650 KB (TFLite buffer overhead) — exceeds H7 512 KB flash; needs size reduction
> ††† 1n: EfficientNetB0 pretrained (ImageNet) — 5126 KB INT8 (~10× H7 limit); NOT MCU-deployable. Stage9 pipeline: mel→dB→p2/p98→efficientnet.preprocess_input. FP32 **95.42%** = dataset ceiling (+0.5 pp above 1j FP32). INT8 91.58% — **3.3 pp below 1j INT8 (94.91%)** — swish activations lose ~3.8 pp to quantization. Our 267KB custom model beats 5MB pretrained network post-quantization.
> MCU ✓ = TFLite Micro compatible (Portenta H7); ✗ = BATCH_MATMUL unsupported or exceeds flash
> Δ = INT8 delta vs 1a baseline (1a→1b→1c→1d→1e→1f→1i→1j chain); 1l Δ vs 1j
> Last updated: 2026-05-13 (1n Stage9 pipeline: 3 seeds Linux done; FP32 95.42% ceiling confirmed)

### Key findings (Linux/CUDA, 3-seed)

- **Best MCU model: 1j** (InvRes+HardSE+5×5DW) — **94.91% INT8**, 267.1 KB, 52m
- **1l (InvRes+HardSE+5×5DW+HSwish)**: 94.68% INT8 — −0.23 pp vs 1j; hard-swish alone didn't help
- **Best compact MCU: 1i** (InvRes+SE) — 94.03% INT8, 258.7 KB, 23m — smallest + fastest
- **Best DS-CNN MCU: 1e** (DS+SE+Res) — 94.12% INT8, 377.2 KB — most consistent DS-CNN
- **1k (InvRes+HardSE+5×5DW+Stack)**: best config 93.75% INT8, but TFLite 650 KB exceeds H7 512 KB limit
- **1f confirms width adds nothing** without attention — 93.61% INT8, equal to 1c
- **Attention ceiling (1g)**: 94.21% INT8 — only +0.6 pp over 1e, not H7 deployable
- **1h underperforms 1g**: wider attention model doesn't help on this task
- **1n (EfficientNetB0 pretrained)**: FP32 **95.42%** = dataset ceiling (+0.5 pp over 1j); INT8 91.58% — swish activations lose ~3.8 pp to quantization; 267 KB 1j beats 5 MB pretrained network post-INT8


---
# Architecture Summary
---

## TABLE 6: Architecture Summary

| Model | Key difference                                    | INT8 KB | INT8 % (Linux)                          | MCU           | Source                          |
| ----- | ------------------------------------------------- | ------- | --------------------------------------- | ------------- | ------------------------------- |
| 1a    | Baseline 2D CNN                                   | 1629.9  | 93.33                                   | ✗†            | custom baseline                 |
| 1b    | DS-CNN (no SE, no residual)                       | 289.7   | 93.06                                   | ✓             | MobileNetV1 DS-CNN              |
| 1c    | + SE block                                        | 376.9   | 93.61                                   | ✓             | + SENet (Hu 2018)               |
| 1d    | + Residual only                                   | 294.8   | 93.10                                   | ✓             | + ResNet skip (He 2016)         |
| 1e    | + SE + Residual                                   | 377.2   | **94.12**                               | ✓             | + SENet + ResNet                |
| 1f    | + wider channels, no attention                    | 455.3   | 93.61                                   | ✓             | channel width scaling           |
| 1g    | + Multi-Head Self-Attention                       | 371.8   | 94.21                                   | ✗             | Transformer (Vaswani 2017/PSLA) |
| 1h    | 1g + wider channels                               | 529.3   | 93.89                                   | ✗‡            | Transformer + width             |
| 1i    | InvRes+SE (inverted residual, channel SE)         | 258.7   | **94.03**                               | ✓             | MobileNetV2 (Sandler 2018)      |
| 1j    | InvRes+HardSE+5×5DW (hard-sigmoid SE, 5×5 dw)    | 267.1   | **94.91**                               | ✓             | MobileNetV3 (Howard 2019)       |
| 1k    | InvRes+HardSE+5×5DW+Stack (stacked res, blk5)    | 650.0†† | 93.75–94.44 (L)                         | ✗††           | MobileNetV3 variant             |
| 1l    | InvRes+HardSE+5×5DW+HSwish (hard-swish blks 3–4) | 269.3   | 94.68                                   | ✓             | MobileNetV3 hard-swish          |
| 1m    | InvRes+OutSE (output-channel SE, se_ratio=0.25)   | 219.9   | 94.35                                   | ✓             | EfficientNet SE (Tan 2019)      |
| 1n    | EfficientNetB0 pretrained                         | 5126.6  | FP32 **95.42** (ceiling); INT8 91.58††† | ✗†††          | EfficientNetB0 (Tan 2019)       |
| 3b–3d | MBV3Small native (64–80 mel)                      | ~1998   | ~81                                     | ✗ (4× over)   | MobileNetV3 Small               |
| 3e–3f | MBV3Small width×0.75                              | ~1270   | ~77–78                                  | ✗ (2.5× over) | MobileNetV3 Small ×0.75         |

>- Best H7-deployable: **1j** (94.91% INT8, 267 KB) — new best; **1i** (94.03% INT8, 259 KB — compact+fast); **1e** (94.12% INT8, 377 KB — best DS-CNN)
>- **1l**: 94.68% INT8 (Linux) — hard-swish alone −0.23 pp vs 1j; macOS pending
>- **1k**: 93.75–94.44% INT8 (Linux, augmentation-dependent) — TFLite 650 KB over H7 limit
>- **1n**: EfficientNetB0 pretrained (Stage9 pipeline) — FP32 **95.42%** dataset ceiling (+0.5 pp over 1j); INT8 91.58% (3.3 pp below 1j INT8 due to swish); 5127 KB; NOT MCU-deployable
>- 1f: width alone adds nothing without attention · 1g/1h: attention ceiling ~94.2%, not H7-deployable
>- No MBV3Small variant fits H7 512 KB — smallest is width×0.75 at ~1270 KB
>- †† 1k TFLite INT8 = 650 KB (buffer overhead despite 488 KB param count) — ✗ H7 flash
>- ††† 1n TFLite INT8 = 5127 KB (~10× H7 limit); swish activations cause INT8 degradation (91.58% vs 95.42% FP32)
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
- `1i_mbv2_se.py` — trains 1i (InvRes+SE: inverted residual + channel SE, MCU compatible)
- `1j_mbv3_se.py` — trains 1j (InvRes+HardSE+5×5DW: hard-sigmoid SE + 5×5 DW, MCU compatible, best overall)
- `1k_mbv3_se_v2.py` — trains 1k (InvRes+HardSE+5×5DW+Stack: stacked residuals, 5th block; ~488 KB INT8)
- `1l_mbv3_se_hs.py` — trains 1l (InvRes+HardSE+5×5DW+HSwish: hard-swish in blks 3–4; ~270 KB INT8)
- `1m_efficientnet_se.py` — trains 1m (InvRes+OutSE: EfficientNet-style SE on output channels, se_ratio=0.25; ~220 KB INT8)
- `1n_efficientnetb0.py` — trains 1n (EfficientNetB0 pretrained: Stage9 pipeline, n_mels=224; FP32 95.42% ceiling, INT8 91.58%, NOT MCU-deployable, 5127 KB INT8)
- `1o_mbv3_matchbox.py` — trains 1o (1j shrunk with MatchboxNet features: FCN epilogue [no Dense-128], DW(1×17) time conv + Conv1×1(128), block 4 capped at 64ch; 120,500 params, 193.7 KB INT8, ops verified H7-safe; untrained — accuracy TBD)
- `run_seabird12_ablation.sh` — runs all experiments (1a–1o); supports per-model filter (1a, 1b, …, 1o)

---

## Deploy Directory — Deployment Manual Reference

The ARGUS Deployment Manual (Ch 4 and Ch 5) references `deploy/` for all end-user
training and firmware conversion. The following are already present:

| Script / File | Purpose |
|---------------|---------|
| `deploy/train.py` | Train MynaNet (1j locked config; `--flat_dir`, `--splits_csv`) |
| `deploy/convert_xxd.sh` | INT8 TFLite → `alignas(8)` C array + header |
| `deploy/mynanet_int8.tflite` | Pre-trained model (267 KB, 94.91% accuracy) |

**Manual-documented invocation (Ch 4):**
```bash
python deploy/train.py \
    --flat_dir   /path/to/mygardenbird16khz \
    --splits_csv /path/to/metadata16khz/splits_mip_80_10_10.csv \
    --random_seed 42
```

**Manual-documented invocation (Ch 5):**
```bash
bash deploy/convert_xxd.sh \
    deploy/mynanet_int8.tflite \
    src/mynanet_model_data \
    g_mynanet_model_data
```
- `run_mels_1c_1i.sh` — mel sweep: 1c+1i × n_mels ∈ {80,96} × 3 seeds
- `run_nmels_ablation.sh` — broader mel sweep: MCU models 1b–1e × {80,96} × 3 seeds
- `update_ablation_results.py` — parses results and updates this file

---
# PAPER — mynanet_neurocomputing.tex
---
## R2 Review Notes (2026-05-25)
Friendly reviewer pass on `/Users/mun3im/Dropbox/Paper4_Neurocomputing/mynanet_neurocomputing.tex`.
Full TODO list: `Paper4_Neurocomputing/PAPER_TODOS.md`

### BLOCKING (must fix before submission)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| B1 | `\TODO{}` deployment latency block still present; 116 ms is estimated from predecessor MAC count, not measured on 1j | §7 Deployment, line ~726 | Remove red TODO; label all latency figures as "estimated (predecessor)" throughout incl. abstract |

### IMPORTANT (reviewer will ask)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| I1 | Abstract says "Mixup surpasses SpecAugment by 0.5%" but §6.2 table shows 0.83% (95.00 vs 94.17) | Abstract line ~57, §6.2 | Change abstract to 0.83%, note it is 10-class historical result |
| I2 | Abstract says "2D CNNs outperform 1D by 2.2%" but actual delta is 2.47% (94.91 − 92.44); also cross-dataset comparison (12-class vs 10-class) | Abstract, §6.1 | Change to 2.47% and add qualifier "estimated across datasets" |
| I3 | "Mixup improves calibration" claimed in §2.5 and §6.2 but no calibration metric (ECE, Brier) is reported anywhere | §2.5 line ~159, §6.2 | Either remove calibration claim or add ECE measurement |
| I4 | MobileNetV3 baseline (82.22%, macOS FP32) compared directly to MynaNet (94.91%, Linux INT8) — different platform, different quantisation — called "12% gap" | §6.3, Conclusion | Add explicit caveat; ideally run MobileNetV3 on 12-class Linux INT8 for apples-to-apples |
| I5 | Table `tab:final_comparison` mixes 10-class (1a–1e historical rows) and 12-class (1j row) without visual separation; easy to misread 1e 95.67% (10-class) as better than 1j 94.91% (12-class) | §6.3 table | Add horizontal rule + footnote; shade 10-class rows or separate into two sub-tables |
| I6 | Deployment latency 116 ms cited as fact in text and abstract but table footnote admits it is from predecessor MAC count | §7, abstract | Either measure or consistently mark as "estimated" everywhere |
| I7 | TCN comparison table mixes 10-class historical TCN rows with 12-class MynaNet reference row; caption says "historical" but text still quotes 2.47% gap as if same task | §4.3 table | Add stronger caveat in text that TCN gap is cross-dataset lower bound |
| I8 | "No prior work systematically compares..." positioning claim needs "to our knowledge" qualifier | §2.5 line ~165 | Soften claim |

### MINOR (polish)

| # | Issue | Fix |
|---|-------|-----|
| m1 | Param count reported as "193K" in body, "≈193,000" in figure caption, "0.19 M" in comparison table — inconsistent | Standardise to "193,396" or "~193K" throughout |
| m2 | "MynaNet" vs "MynaNet (1j)" vs "Model 1j (MynaNet)" used interchangeably — first use in abstract should define the relationship | Add "MynaNet (model 1j, hereafter MynaNet)" on first occurrence |
| m3 | Energy figures (13.9 mJ, 118 mW) given without comparison to sensor power budget or competing systems | Add reference power budget for a solar-powered sensor or note these are order-of-magnitude estimates |
| m4 | §4.3 (TCN) says "causal padding incurs computation without benefit" — non-causal inference is a design choice, should clarify this is a limitation of causal TCN, not TCNs generally | Reword to "our causal TCN implementation..." |
| m5 | `1j_mbv3_se_architecture.png` is referenced but was noted as "just a box" — verify PNG renders correctly before submission | Test in compiled PDF |
| m6 | Scripts section in CLAUDE.md has wrong label mappings (1p→1l, 1q→1m, 1r→1n) | Fix CLAUDE.md script labels |

