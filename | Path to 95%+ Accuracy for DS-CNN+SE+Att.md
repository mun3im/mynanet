
---

## Gap Analysis

| Metric        | DS-CNN+SE+Att | MobileNetV3 | Delta            |
| ------------- | ------------- | ----------- | ---------------- |
| INT8 Accuracy | 94.11%        | 95.33%      | -1.22%           |
| Parameters    | 295K          | 1.7M        | 5.7× smaller     |
| Model Size    | 388 KB        | 1996 KB     | 5.1× smaller     |
| Best Val Acc  | 91.33%        | 92.00%      | -0.67%           |
| Train–Val Gap | 4.79%         | 8.50%       | Less overfitting |

---

## Key Insight

The DS-CNN model is underfitting (4.79% train–val gap vs 8.50% for MobileNetV3).  
This means you have room to increase model capacity without causing generalization issues.

---

# Recommended Experiments (Priority Order)

## 1. Widen Channel Widths (+0.5–1.0% expected)

**Current:**  
`64 → 128 → 256 → 512`

**Proposed:**  
`80 → 160 → 320 → 640` _(25% wider)_

- Minimal size increase (~480 KB, still under 512 KB)    
- Better feature representation    

---

## 2. Add More Residual Connections (+0.3–0.5% expected)

**Current:** Only Block 1 has residual  
**Proposed:** Add residuals to Blocks 2–4 with 1×1 projection

- Already implemented in MobileNetV3    
- Improves gradient flow    

---

## 3. Enhance Attention Module (+0.2–0.4% expected)

**Current:** 2 heads, 32 `key_dim`, 64-dim projection  
**Proposed:** 4 heads, 64 `key_dim`, 128-dim projection

- Current attention is too lightweight    
- Double the attention capacity    

---

## 4. Combine Augmentations (+0.3–0.5% expected)

Try **Mixup + SpecAugment together**:

```bash
python 1d_dscnn_se_res_att.py --mixup 0.2 --specaugment
```

- Currently only using one at a time    
- Combined augmentation improves diversity    

---

## 5. Label Smoothing (+0.2–0.3% expected)

```python
loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
```

- Reduces overconfidence    
- Pairs well with mixup    

---

## 6. Try Seed 786 (+0–0.5% expected)

```bash
python 1d_dscnn_se_res_att.py --random_seed 786 --mixup 0.2
```

- MobileNetV3 used seed 786    
- Different initialization can find better minima    

---

## 7. Knowledge Distillation (+0.5–1.0% expected)

Train DS-CNN to match MobileNetV3 soft labels:

- **Teacher:** MobileNetV3 (95.33%)    
- **Student:** DS-CNN+SE+Att    
- **Loss:**  
    `α * CE(student, labels) + (1-α) * KL(student, teacher)`
- Most promising for closing the gap    
- Transfers MobileNetV3’s learned representations
    

---

# Quick Win Experiment

Try this command first (combines 3 improvements):

```bash
python 1d_dscnn_se_res_att.py \
    --dropout 0.0 \
    --mixup 0.2 \
    --specaugment \
    --random_seed 786 \
    --warmup_epochs 70 \
    --finetune_epochs 30
```

This combines:

- Different seed (786)    
- Dual augmentation (mixup + specaugment)    
- Longer training (100 total epochs)    

---

# Architecture Modification Priority

If you're willing to modify the model code, prioritize in this order:

1. Wider channels (80→160→320→640) — **biggest impact**    
2. Residuals in all blocks — **free improvement**    
3. Enhanced attention (4 heads, 64 `key_dim`)    
4. Hard-swish activation instead of ReLU6    

---

## Expected Combined Improvement

**+1.0 to 1.5%**, reaching **95–95.5%** while staying under **512 KB**.

---
