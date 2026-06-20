# MynaNet Model Organization & Benchmarking Plan

**Date**: December 20, 2025
**Purpose**: Systematic model numbering for TCN paper benchmarking
**Input**: Fixed 64×300 mel spectrograms (except MobileNet pretrained)
**Dataset**: SEABird (6,000 samples, 10 species @ 16kHz)

---

## Systematic Numbering Scheme

### **1_baseline_2dcnn.py** - 2D CNN Baseline
**Architecture**: Standard Conv2D + MaxPool
**Source**: Reuse/modify `1a_base_64x300.py`
**Purpose**: Baseline to compare against temporal models

```python
Architecture:
Conv2D(32, 3×3) → MaxPool2D(2×2)
Conv2D(64, 3×3) → MaxPool2D(2×2)
Conv2D(128, 3×3) → MaxPool2D(2×2)
GlobalAveragePooling2D
Dense(128) → Dropout → Dense(10)
```

**Expected issues**: MaxPooling struggles with 300 frames
- 300 → 150 → 75 → 37.5 (needs special handling)

**File mapping**:
- `1a_base_64x300.py` → `1_baseline_2dcnn.py`

---

### **2a-2z_mobilenet_*.py** - MobileNet Family
**Architecture**: Depthwise Separable Convolutions
**Purpose**: State-of-practice lightweight CNN baseline
**Reference**: SEABird paper reports 90%+ accuracy with MobileNetV3-Small

#### Existing variants to organize:

**2a_mobilenet_64x300.py** (Custom input)
- Source: `2b_mobilenet_64x300.py`
- Input: 64×300 (custom, not pretrained)
- Purpose: Fair comparison with same input shape

**2b_mobilenetv3_64x300.py** (Custom input)
- Source: `2c_mobilenetv3_64x300.py`
- Input: 64×300 (custom, not pretrained)
- Architecture: MobileNetV3-Small/Large
- Purpose: Latest MobileNet version

**2c_mobilenetv3_224x224.py** (Pretrained - LOW RESOURCE)
- Source: `2d_mobilenetv3_224x224.py`
- Input: 224×224 (ImageNet pretrained, transfer learning)
- Purpose: Pretrained baseline (different input shape acceptable)
- Note: Uses pretrained weights, may have advantage

**File mapping**:
- `2b_mobilenet_64x300.py` → `2a_mobilenet_64x300.py`
- `2c_mobilenetv3_64x300.py` → `2b_mobilenetv3_64x300.py`
- `2d_mobilenetv3_224x224.py` → `2c_mobilenetv3_pretrained_224x224.py`

---

### **3a-3z_sota_*.py** - SOTA Architectures (Transformers, etc.)

**Purpose**: Compare against state-of-the-art sequence/audio models
**Input**: All use 64×300 (except where pretrained requires different size)

#### Recommended Model 3 variants:

#### **3a_transformer_encoder.py** ⭐ RECOMMENDED FIRST
**Architecture**: Pure Transformer Encoder (from scratch)
```python
Input: 64×300 mel spectrogram
↓
Reshape to sequence: (batch, 300, 64) [time steps, features]
↓
Positional Encoding (learnable or sinusoidal)
↓
Transformer Encoder × 4 blocks
  - Multi-head attention (4-8 heads)
  - Feed-forward network (256-512 units)
  - LayerNorm + residual connections
  - Dropout (0.1-0.2)
↓
Global Average Pooling (over time)
↓
Dense(128) → Dense(10)
```

**Pros**:
- True SOTA sequence model
- Self-attention captures long-range dependencies
- Good benchmark for TCN temporal modeling

**Cons**:
- Quadratic complexity O(300²)
- Large parameter count (~1-5M)
- May not fit Cortex-M7 (but good for comparison)

**Expected performance**: High accuracy, high latency

---

#### **3b_ast_audio_spectrogram_transformer.py** (Optional)
**Architecture**: Audio Spectrogram Transformer (AST)
**Reference**: Gong et al. 2021 - "AST: Audio Spectrogram Transformer"
**Pretrained**: AudioSet (if available)

```python
Input: 64×300 → Resize to 128×128 (AST standard)
↓
Patch embedding (16×16 patches)
↓
Vision Transformer (ViT) encoder × 12 layers
↓
Classification head
```

**Pros**:
- SOTA for audio classification
- Pretrained on AudioSet (if using transfer learning)
- Published benchmark

**Cons**:
- Requires different input size (128×128 typical)
- Very large model (86M parameters in base version)
- Not MCU-deployable (comparison only)

**Implementation**: Use Hugging Face `transformers` library
```python
from transformers import ASTForAudioClassification
```

**Decision**: Include if you want published SOTA baseline

---

#### **3c_conformer.py** (Optional - Advanced)
**Architecture**: Conformer (Convolution + Transformer)
**Reference**: Gulati et al. 2020 - "Conformer: Convolution-augmented Transformer"

```python
Input: 64×300
↓
Conformer Block × 6
  - Multi-head self-attention
  - Convolution module (depthwise conv)
  - Feed-forward module
  - LayerNorm
↓
Global pooling → Dense(10)
```

**Pros**:
- Combines CNN local features + Transformer global context
- SOTA in speech recognition (ASR)
- Interesting hybrid comparison

**Cons**:
- Complex implementation
- Large model size
- May be overkill for 10-class problem

**Decision**: Include if you have time and want comprehensive comparison

---

#### **3d_efficientnet_audio.py** (Optional - CNN baseline)
**Architecture**: EfficientNet adapted for audio
**Note**: CNN-based, not strictly "SOTA" (Transformers are current SOTA)

```python
EfficientNetB0/B1 (pretrained on ImageNet)
Fine-tune on 64×300 spectrograms
```

**Pros**:
- Strong CNN baseline
- Compound scaling (depth + width + resolution)
- Widely used

**Cons**:
- Not Transformer-based (less interesting for TCN comparison)
- Already have MobileNet as CNN baseline

**Decision**: Skip unless you want another CNN variant

---

### **RECOMMENDATION FOR MODEL 3**:

**Start with 3a_transformer_encoder.py** (Pure Transformer)
- Simplest to implement from scratch
- Clear SOTA sequence model comparison
- Good contrast with TCN (attention vs dilated convolutions)

**Optionally add 3b_ast** if:
- You want published SOTA with pretrained weights
- Willing to handle different input size (128×128)
- Have Hugging Face transformers library

---

### **4a-4z_tcn_*.py** - TCN Variants (Main Focus)

**Purpose**: Systematic TCN ablation for MynaNet paper
**Architecture**: Temporal Convolutional Network with dilated causal convolutions

#### Existing TCN files to reorganize:
```
4c_tcn_mel_64x300.py
4d_tcn_mel_64x300.py
5c_tcn_mel_64x300_journal_ready.py
6c_tcn_mel_64x300_journal_ready.py
7c_train_tweetcn.py
7d_train_tweetcn.py
```

#### Proposed TCN variant naming (based on ablations):

**4a_tcn_baseline.py** - Baseline TCN
- Source: Best of `4c` or `4d`
- Blocks: 7 (dilations: 1, 2, 4, 8, 16, 32, 64)
- Filters: 32 per block
- Kernel: 3
- Dropout: 0.2
- Residual: Yes
- RF: 509 time steps (covers 300 fully)

**4b_tcn_shallow.py** - Shallow TCN
- Blocks: 5 (dilations: 1, 2, 4, 8, 16)
- Filters: 32
- Kernel: 3
- RF: 65 time steps
- Purpose: Test if full receptive field needed

**4c_tcn_wide.py** - Wide TCN
- Blocks: 7
- Filters: 64 (double baseline)
- Kernel: 3
- Purpose: More capacity

**4d_tcn_deep.py** - Deep TCN
- Blocks: 9 (dilations: 1, 2, 4, 8, 16, 32, 64, 128, 256)
- Filters: 32
- Kernel: 3
- RF: 1023 (over-coverage)
- Purpose: Test very deep network

**4e_tcn_small_kernel.py** - Small Kernel TCN
- Blocks: 7
- Filters: 32
- Kernel: 2 (instead of 3)
- Purpose: Faster inference

**4f_tcn_large_kernel.py** - Large Kernel TCN
- Blocks: 7
- Filters: 32
- Kernel: 5
- Purpose: Larger receptive field per layer

**4g_tcn_no_residual.py** - No Residual Connections
- Blocks: 7
- Filters: 32
- Kernel: 3
- Residual: No
- Purpose: Test importance of skip connections

**4h_tcn_lightweight.py** - Lightweight TCN (MCU-optimized)
- Blocks: 5
- Filters: 16
- Kernel: 3
- Purpose: Smallest deployable variant

**4i_tcn_tweetcn.py** - TweetCN Architecture
- Source: `7c_train_tweetcn.py` or `7d_train_tweetcn.py`
- Purpose: Published TCN variant for audio

**4j_tcn_journal_ready.py** - Best Performer (Journal Version)
- Source: Best of `5c` or `6c`
- Purpose: Optimized configuration from initial experiments

#### Additional TCN ablations (if needed):

**4k_tcn_dropout01.py** - Low Dropout
**4l_tcn_dropout03.py** - Medium Dropout
**4m_tcn_dropout04.py** - High Dropout

**4n_tcn_batchnorm.py** - With Batch Normalization
**4o_tcn_layernorm.py** - With Layer Normalization

**4p_tcn_glu.py** - Gated Linear Units (like WaveNet)
**4q_tcn_causal_padding.py** - Strict causal padding

---

## Model Inventory & File Mapping

### Current files → New systematic names:

```
EXISTING FILE                           → NEW NAME                         STATUS
─────────────────────────────────────────────────────────────────────────────────
1a_base_64x300.py                       → 1_baseline_2dcnn.py              Rename
2b_mobilenet_64x300.py                  → 2a_mobilenet_64x300.py           Rename
2c_mobilenetv3_64x300.py                → 2b_mobilenetv3_64x300.py         Rename
2d_mobilenetv3_224x224.py               → 2c_mobilenetv3_pretrained.py     Rename
[NEW]                                   → 3a_transformer_encoder.py        Create
[NEW - optional]                        → 3b_ast.py                        Create
4c_tcn_mel_64x300.py                    → 4a_tcn_baseline.py               Decide
4d_tcn_mel_64x300.py                    → 4a_tcn_baseline.py (or 4b)      Decide
5c_tcn_mel_64x300_journal_ready.py      → 4j_tcn_journal_ready.py          Rename
6c_tcn_mel_64x300_journal_ready.py      → 4j_tcn_journal_ready.py (merge?) Decide
7c_train_tweetcn.py                     → 4i_tcn_tweetcn.py                Rename
7d_train_tweetcn.py                     → 4i_tcn_tweetcn.py (use latest)   Rename
[NEW]                                   → 4b-4h (TCN ablations)            Create
```

---

## Implementation Priority

### Phase 1: Reorganize Existing Models (High Priority)
1. ✅ Rename `1a_base_64x300.py` → `1_baseline_2dcnn.py`
2. ✅ Rename MobileNet variants → `2a-2c`
3. ✅ Decide on best TCN baseline from `4c/4d` → `4a`
4. ✅ Rename TweetCN `7d` → `4i`
5. ✅ Rename journal-ready `5c` or `6c` → `4j`

### Phase 2: Create SOTA Baseline (High Priority)
6. 🔧 Implement `3a_transformer_encoder.py` (pure Transformer)
7. 🔧 (Optional) Implement `3b_ast.py` (if want pretrained SOTA)

### Phase 3: Create TCN Ablations (Medium Priority)
8. 🔧 Implement `4b-4h` (shallow, wide, deep, kernel variants, etc.)
9. 🔧 Implement dropout/normalization variants `4k-4o`

### Phase 4: Unified Benchmarking (High Priority)
10. 🔧 Create `benchmark_all_models.py` script
11. 🔧 Create results comparison table generator

---

## Benchmarking Requirements

### All models must report:
- **Accuracy**: Top-1 classification accuracy
- **F1-Score**: Macro-averaged
- **Latency**: Mean inference time (ms) on target hardware
- **Model Size**: TFLite int8 file size (KB)
- **Parameters**: Total trainable parameters
- **MACs**: Multiply-accumulate operations
- **Quantization Degradation**: Float32 vs int8 accuracy loss

### Standard evaluation protocol:
- **Dataset**: SEABird (6,000 samples, 10 species)
- **Split**: 70/15/15 (train/val/test) - stratified
- **Input**: 64×300 mel spectrograms (except MobileNet pretrained: 224×224)
- **Training**: 100 epochs, early stopping (patience=15)
- **Optimizer**: Adam (lr=0.001, ReduceLROnPlateau)
- **Augmentation**: Test with/without (SpecAugment, Mixup)
- **Seeds**: Multi-seed (42, 123, 456) for stability

---

## Expected Paper Results Table

| Model | Type | Accuracy | F1 | Latency | Size | Params | MACs | Note |
|-------|------|----------|----|---------|----|--------|------|------|
| 1_baseline_2dcnn | CNN | [X]% | [X] | [X]ms | [X]KB | [X]K | [X]M | MaxPool issues w/ 300 frames |
| 2a_mobilenet | CNN | 90%+ | [X] | [X]ms | [X]KB | [X]K | [X]M | SEABird paper baseline |
| 2b_mobilenetv3 | CNN | [X]% | [X] | [X]ms | [X]KB | [X]K | [X]M | Latest MobileNet |
| 2c_mobilenetv3_pretrained | CNN | [X]% | [X] | [X]ms | [X]KB | [X]K | [X]M | Transfer learning (224×224) |
| **3a_transformer** | Transformer | [X]% | [X] | [X]ms | [X]KB | [X]M | [X]M | SOTA sequence model |
| 3b_ast | Transformer | [X]% | [X] | [X]ms | [X]KB | [X]M | [X]M | Pretrained AudioSet |
| **4a_tcn_baseline** | TCN | **[X]%** | **[X]** | **[X]ms** | **[X]KB** | **[X]K** | **[X]M** | **Main MynaNet** |
| 4b_tcn_shallow | TCN | [X]% | [X] | [X]ms | [X]KB | [X]K | [X]M | Faster, smaller RF |
| 4c_tcn_wide | TCN | [X]% | [X] | [X]ms | [X]KB | [X]K | [X]M | More capacity |
| 4i_tcn_tweetcn | TCN | [X]% | [X] | [X]ms | [X]KB | [X]K | [X]M | Published variant |
| 4j_tcn_journal_ready | TCN | [X]% | [X] | [X]ms | [X]KB | [X]K | [X]M | Optimized config |

---

## Key Paper Claims to Validate

### RQ1: TCN vs CNN
**Hypothesis**: TCN outperforms 2D CNN due to temporal modeling
**Comparison**: 4a_tcn vs 1_baseline_2dcnn
**Expected**: TCN higher accuracy, similar/better latency

### RQ2: TCN vs SOTA Transformer
**Hypothesis**: TCN matches Transformer accuracy with fewer parameters
**Comparison**: 4a_tcn vs 3a_transformer
**Expected**: TCN ~95% of Transformer accuracy, 5-10× fewer params, faster inference

### RQ3: Handling 300 frames (non-power-of-2)
**Hypothesis**: TCN handles arbitrary sequence length better than pooling-based CNN
**Evidence**: 1_baseline_2dcnn needs workarounds (GlobalAveragePooling instead of MaxPool)
**TCN**: No architectural constraints

### RQ4: Quantization robustness
**Hypothesis**: TCN quantizes well (<1% degradation)
**Comparison**: All models float32 vs int8
**Expected**: TCN <1%, competitive with CNNs

### RQ5: MCU deployment
**Hypothesis**: TCN fits Portenta H7 Cortex-M7
**Requirements**: <500KB model, <10ms latency, <512KB RAM
**Validation**: Deploy 4a_tcn_baseline on target

---

## Next Steps

### Immediate Actions:
1. **Review existing TCN files** (`4c, 4d, 5c, 6c, 7c, 7d`) - decide which to keep as baseline
2. **Rename files** systematically (create backup first)
3. **Implement 3a_transformer_encoder.py** - critical SOTA baseline
4. **Create benchmark script** - unified evaluation pipeline

### Questions to Decide:
- [ ] Which TCN file is best baseline? `4c` vs `4d` vs `5c` vs `6c`?
- [ ] Should we merge `5c` and `6c` (both "journal ready") or keep separate?
- [ ] Do we want AST (3b) or just pure Transformer (3a)?
- [ ] How many TCN ablations needed? (4b-4h: 7 variants, or more?)

---

**Status**: Ready to proceed with reorganization
**Blockers**: None - files identified, plan ready
**Estimated time**:
- Renaming: 30 min
- Implementing 3a_transformer: 2-3 hours
- Creating TCN ablations 4b-4h: 1-2 hours
- Benchmark script: 1 hour

---

*END OF ORGANIZATION PLAN*
