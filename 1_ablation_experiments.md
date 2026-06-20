# Ablation Study Experiments: 2D CNN vs TCN for Seabird Classification

## Overview
This document proposes a comprehensive ablation study comparing baseline 2D CNN architectures with the TCN model across various configurations. The goal is to establish whether the temporal modeling capability of TCNs justifies the complexity, or if simple 2D CNNs can achieve comparable performance.

---

## Motivation

Based on 7d_train_tweetcn.py analysis, the current system uses:
- **Input**: 64×300 mel-spectrograms (frequency × time)
- **Model**: Temporal Convolutional Network (TCN) with dilated causal convolutions
- **Complexity**: 200K-800K parameters depending on channel width
- **Performance**: 92-98% accuracy (INT8)

**Key Questions:**
1. Does temporal causality matter for 3-second seabird clips?
2. Can a simpler 2D CNN achieve similar accuracy with fewer parameters?
3. What is the optimal architecture for embedded deployment?

---

## Proposed Ablation Experiments

### Series 1: Baseline 2D CNN Architectures

#### 1a. Small 2D CNN (Baseline)
**Purpose**: Establish minimum viable model

**Architecture:**
```
Input: (64, 300, 1)
Conv2D(32, 3×3, relu6) → MaxPool2D(2×2)
Conv2D(64, 3×3, relu6) → MaxPool2D(2×2)
Conv2D(128, 3×3, relu6) → MaxPool2D(2×2)
GlobalAveragePooling2D
Dense(128, relu6) → Dropout
Dense(10, softmax)
```

**Parameters:** ~100K (target: <200 KB INT8)
**Receptive field:** 21×21 (spatial + temporal)
**Script:** `1a_base_64x300.py`

---

#### 1b. Medium 2D CNN
**Purpose**: More capacity for comparison

**Architecture:**
```
Input: (64, 300, 1)
Conv2D(64, 3×3, relu6) → MaxPool2D(2×2)
Conv2D(128, 3×3, relu6) → MaxPool2D(2×2)
Conv2D(256, 3×3, relu6) → MaxPool2D(2×2)
GlobalAveragePooling2D
Dense(256, relu6) → Dropout
Dense(128, relu6) → Dropout
Dense(10, softmax)
```

**Parameters:** ~400K (comparable to TCN-96)
**Script:** `1b_medium_64x300.py`

---

#### 1c. Deep 2D CNN
**Purpose**: Test depth vs width trade-off

**Architecture:**
```
Input: (64, 300, 1)
Conv2D(32, 3×3, relu6) → MaxPool2D(2×2)
Conv2D(64, 3×3, relu6) → MaxPool2D(2×2)
Conv2D(128, 3×3, relu6)
Conv2D(128, 3×3, relu6) → MaxPool2D(2×2)
Conv2D(256, 3×3, relu6)
GlobalAveragePooling2D
Dense(256, relu6) → Dropout
Dense(10, softmax)
```

**Parameters:** ~300K
**Script:** `1c_deep_64x300.py`

---

### Series 2: Specialized 2D CNN Variants

#### 2a. Frequency-Aware CNN
**Purpose**: Exploit known frequency structure of bird calls

**Architecture:**
```
Input: (64, 300, 1)
# Frequency-first convolutions
Conv2D(32, (7, 3), relu6)  # Tall kernels for frequency
MaxPool2D((2, 1))           # Pool only frequency
Conv2D(64, (5, 3), relu6)
MaxPool2D((2, 2))
Conv2D(128, (3, 5), relu6)  # Square kernels
MaxPool2D((2, 2))
Conv2D(256, (1, 7), relu6)  # Wide kernels for time
GlobalAveragePooling2D
Dense(128, relu6) → Dropout
Dense(10, softmax)
```

**Parameters:** ~250K
**Rationale:** Mimics mel-spectrogram structure (frequency vs time)

---

#### 2b. MobileNet-inspired CNN
**Purpose**: Depthwise separable convolutions for efficiency

**Architecture:**
```
Input: (64, 300, 1)
# Depthwise separable blocks
Conv2D(32, 3×3, relu6)
DepthwiseConv2D(3×3) → Conv2D(64, 1×1, relu6) → MaxPool2D(2×2)
DepthwiseConv2D(3×3) → Conv2D(128, 1×1, relu6) → MaxPool2D(2×2)
DepthwiseConv2D(3×3) → Conv2D(256, 1×1, relu6) → MaxPool2D(2×2)
GlobalAveragePooling2D
Dense(128, relu6) → Dropout
Dense(10, softmax)
```

**Parameters:** ~80K (smallest model)
**Script:** `2b_mobilenet_64x300.py`

---

### Series 3: TCN Variants (From 7d)

#### 3a. TCN-64 (Current Baseline)
**From:** `7d_train_tweetcn.py --tcn_channels 64`
**Parameters:** ~200K
**Architecture:** 2 TCN blocks, dilations [1,2,4,8,16,32]

#### 3b. TCN-96
**From:** `7d_train_tweetcn.py --tcn_channels 96`
**Parameters:** ~450K

#### 3c. TCN-128
**From:** `7d_train_tweetcn.py --tcn_channels 128`
**Parameters:** ~800K

---

### Series 4: Hybrid Architectures

#### 4a. CNN+TCN Hybrid
**Purpose**: 2D CNN for spatial features, 1D TCN for temporal

**Architecture:**
```
Input: (64, 300, 1)
# 2D CNN for frequency patterns
Conv2D(32, 3×3, relu6) → MaxPool2D(2×1)  # Pool frequency only
Conv2D(64, 3×3, relu6) → MaxPool2D(2×1)
# Reshape to temporal sequence
Reshape((-1, time_steps))
# TCN for temporal patterns
TCN blocks (channels=64, dilations=[1,2,4,8])
GlobalAveragePooling1D
Dense(128, relu6) → Dropout
Dense(10, softmax)
```

**Parameters:** ~150K
**Script:** `4a_hybrid_64x300.py`

---

## Experimental Protocol

### Fixed Hyperparameters (Same as 7d)
```python
# Data
TARGET_SR = 16000
AUDIO_LENGTH_SEC = 3
N_MELS = 64
TIME_FRAMES = 300
HOP_LENGTH = 160
N_FFT = 512
FMAX = 8000

# Split
TEST_SIZE_PER_CLASS = 90
VAL_SIZE_PER_CLASS = 60
TRAIN_SIZE_PER_CLASS = 450

# Training
WARMUP_EPOCHS = 50
FINETUNE_EPOCHS = 20
BATCH_SIZE = 32
WARMUP_LR = 1e-3
FINETUNE_LR = 1e-5
RANDOM_SEED = 786
```

### Variables to Test

#### Dropout Rates
- 0.0, 0.1, 0.2, 0.3

#### Augmentation Strategies
- **None**: Baseline (no augmentation)
- **Baseline**: Time shift ±100ms, pitch shift ±2 semitones
- **SpecAugment**: Freq mask 8 bins, time mask 20 frames
- **Mixup**: α = 0.2

#### LR Schedules
- **Cosine**: Smooth annealing
- **Plateau**: Adaptive reduction
- **Both**: Combined

---

## Experimental Matrix

### Phase 1: Architecture Comparison (No Augmentation)
**Goal:** Isolate architectural differences

| Model | Params | Dropout | Expected Time |
|-------|--------|---------|---------------|
| 1a_base (Small 2D CNN) | ~100K | 0.0, 0.2 | 10 min × 2 = 20 min |
| 1b_medium (Medium 2D CNN) | ~400K | 0.0, 0.2 | 15 min × 2 = 30 min |
| 1c_deep (Deep 2D CNN) | ~300K | 0.0, 0.2 | 15 min × 2 = 30 min |
| 2a_freq (Freq-Aware CNN) | ~250K | 0.0, 0.2 | 15 min × 2 = 30 min |
| 2b_mobile (MobileNet CNN) | ~80K | 0.0, 0.2 | 10 min × 2 = 20 min |
| 3a_tcn64 (TCN Baseline) | ~200K | 0.0, 0.2 | 20 min × 2 = 40 min |

**Total:** 12 experiments, ~3 hours on GPU

---

### Phase 2: Best Architecture + Augmentation
**Goal:** Optimize best 2-3 architectures from Phase 1

For each top model:
- Augmentation: None, Baseline, SpecAugment, Mixup (4 variants)
- Dropout: Best from Phase 1

**Total:** 3 models × 4 augmentations = 12 experiments, ~4 hours

---

### Phase 3: Fine-Tuning Best Configuration
**Goal:** Optimize hyperparameters for winner

- LR schedule: Cosine, Plateau, Both (3 variants)
- Warmup epochs: 50, 100 (2 variants)
- Multiple seeds: 786, 42, 100 (3 runs for statistics)

**Total:** 3 × 2 × 3 = 18 experiments, ~6 hours

---

## Success Criteria

### Model Performance Targets
- **Minimum viable**: 90% INT8 accuracy
- **Competitive**: 93% INT8 accuracy (match TCN-64)
- **Excellent**: 95% INT8 accuracy (match TCN-96)

### Deployment Constraints
- **Hard limit**: <512 KB INT8 model size
- **Target**: <200 KB INT8 model size
- **Ideal**: <100 KB INT8 model size

### Efficiency Metrics
- **Parameter efficiency**: Accuracy per 100K parameters
- **Size efficiency**: Accuracy per 100 KB
- **Inference speed**: Theoretical MACs (multiply-accumulate ops)

---

## Expected Outcomes

### Hypothesis 1: Simple 2D CNN Competitive
**Prediction:** Small 2D CNN (1a) achieves 90-93% with proper augmentation

**Reasoning:**
- 3-second clips are short, limited temporal dependencies
- Spectrogram structure is 2D image-like
- Pooling provides translation invariance

**If true:** Use 2D CNN for deployment (simpler, smaller, faster)

---

### Hypothesis 2: TCN Superior for Temporal
**Prediction:** TCN maintains 2-3% edge over 2D CNN

**Reasoning:**
- Dilated convolutions capture long-range dependencies
- Causal structure respects temporal order
- Skip connections aggregate multi-scale features

**If true:** TCN worth the complexity for production

---

### Hypothesis 3: Hybrid Best of Both
**Prediction:** CNN+TCN hybrid achieves highest accuracy/size ratio

**Reasoning:**
- 2D CNN extracts frequency patterns efficiently
- TCN models temporal evolution
- Smaller than pure TCN, smarter than pure CNN

**If true:** Hybrid architecture is optimal

---

## Metrics to Collect

### Primary Metrics
- INT8 test accuracy (primary objective)
- FP32 test accuracy
- Accuracy drop (FP32 → INT8)
- Model size (INT8 KB)

### Secondary Metrics
- Per-class F1 scores
- Confusion matrix patterns
- Training time (minutes)
- Convergence speed (epochs to 90%)
- Overfitting gap (train - val accuracy)

### Efficiency Metrics
- Parameters count
- Theoretical MACs
- Receptive field size
- Activation memory

---

## Analysis Plan

### Comparison Tables
1. **Architecture comparison** (Phase 1)
   - Rank by INT8 accuracy
   - Plot accuracy vs parameters
   - Plot accuracy vs size

2. **Augmentation comparison** (Phase 2)
   - Best config for each model type
   - Robustness to augmentation

3. **Statistical validation** (Phase 3)
   - Mean ± std across seeds
   - McNemar's test for significance
   - Confusion matrix analysis

### Visualization
- Training curves (all models overlaid)
- Confusion matrices (best models)
- Accuracy vs size scatter plot
- Augmentation impact bar chart

---

## Implementation Notes

### Code Reuse from 7d
All scripts should inherit from `7d_train_tweetcn.py`:
- Same preprocessing pipeline
- Same data split strategy (90/60/450)
- Same training protocol (warmup + finetune)
- Same logging and evaluation
- Same quantization (PTQ INT8)

**Only change:** Model architecture (replace `create_tcn()`)

### Naming Convention
```
{series}{letter}_{name}_64x300.py
```
Examples:
- `1a_base_64x300.py` - Small 2D CNN
- `2b_mobilenet_64x300.py` - MobileNet-style
- `4a_hybrid_64x300.py` - CNN+TCN hybrid

### Output Directories
```
results_{script}_ptq_drop{XX}_rand{SEED}_{aug}_{platform}/
```
Example:
- `results_1a_base_ptq_drop00_rand786_mixup0.2_linux/`

---

## Timeline Estimate

| Phase | Experiments | GPU Time | Elapsed |
|-------|-------------|----------|---------|
| Setup | Script creation | 0 | 2 hours |
| Phase 1 | 12 (architecture) | 3 hours | 1 day |
| Phase 2 | 12 (augmentation) | 4 hours | 1 day |
| Phase 3 | 18 (fine-tuning) | 6 hours | 1 day |
| Analysis | Tables, plots | 0 | 4 hours |
| **Total** | **42 experiments** | **13 hours** | **3-4 days** |

---

## Expected Results Summary

### Likely Ranking (by INT8 accuracy)
1. **TCN-96 + Mixup** - 95-97% (baseline)
2. **Hybrid + SpecAugment** - 94-96% (best efficiency)
3. **TCN-64 + Mixup** - 93-95%
4. **Medium 2D CNN + Mixup** - 91-94%
5. **Freq-Aware CNN + SpecAugment** - 90-93%
6. **Small 2D CNN + Mixup** - 89-92%
7. **MobileNet CNN + Mixup** - 88-91%

### Key Insights Expected
- **Augmentation matters more than architecture** for this problem
- **2D CNNs competitive** if properly regularized
- **TCN edge diminishes** with strong augmentation
- **Hybrid balances** accuracy and efficiency

---

## Reproducibility

All experiments use:
- **Fixed seeds**: 786 (default), 42, 100 (ensemble)
- **Same splits**: Per-class 450/60/90
- **Same preprocessing**: 16kHz, 64 mels, 300 frames
- **Documented hyperparameters**: All in script headers
- **CSV export**: Automatic batch comparison

---

## References

### Baseline Architectures
- VGGNet (Simonyan & Zisserman, 2014): Simple stacked convolutions
- MobileNet (Howard et al., 2017): Depthwise separable convolutions
- ResNet (He et al., 2016): Skip connections

### Temporal Models
- TCN (Bai et al., 2018): Temporal Convolutional Networks
- WaveNet (van den Oord et al., 2016): Dilated causality

### Audio Classification Baselines
- YAMNet (Google, 2020): 2D CNN for audio
- BirdNET (Kahl et al., 2021): ResNet for bird calls
- PANNs (Kong et al., 2020): Pre-trained audio networks

---

## Document Metadata

**Version:** 1.0
**Created:** 2025-12-03
**Purpose:** Ablation study design for seabird classification
**Status:** Proposed - ready for implementation
**Next Step:** Implement `1a_base_64x300.py` baseline
