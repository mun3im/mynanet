# Model 1e vs MynaNet v1 Analysis

**Date:** February 8, 2026
**Status:** Investigating why 1e (95.67%) outperforms v1 (94.67%)

---

## The Mystery

User observed that **Model 1e achieved 95.67%**, which is **1.0% higher than MynaNet v1's 94.67%**.

This contradicts our earlier conclusion that v1 was the best model. Let's investigate.

---

## Architecture Comparison

| Component | Model 1e (Original) | MynaNet v1 (Optimized) | Difference |
|-----------|---------------------|------------------------|------------|
| **Channel Progression** | [80, 160, 320, 640] | [120, 180, 146, 360] | 1e wider at later layers |
| **MHSA Configuration** | 4 heads, 48 key_dim | 2 heads, 32 key_dim, 88 dims | 1e has more attention |
| **Dense Head** | 192 units | 176 units | 1e wider |
| **Total Parameters** | ~461K | ~395K | 1e has +17% params |
| **Model Size (INT8)** | **579KB** | **434KB** | 1e is +33% larger |
| **INT8 Accuracy** | **95.67%** | **94.67%** | 1e is +1.0% better |
| **Deployment Status** | ❌ OVER 512KB | ✓ UNDER 512KB | v1 meets target |

---

## Key Finding: Accuracy vs Size Trade-off

### Model 1e Strengths:
1. **Wider late-stage channels:** 640 channels in Block 4 (vs v1's 360)
   - More capacity for complex pattern composition
   - Better feature representation in final layers

2. **Enhanced MHSA:** 4 heads with 48 key_dim (vs v1's 2 heads, 32 key_dim)
   - More attention capacity for temporal dependencies
   - Better modeling of long-range patterns

3. **Wider dense head:** 192 units (vs v1's 176)
   - More classification capacity

### Model 1e Weakness:
- **Size: 579KB** → **67KB over 512KB target** (13% over budget)
- Cannot deploy to Cortex-M7 with 512KB constraint

### MynaNet v1 Strengths:
- **Size: 434KB** → **78KB under 512KB target** (15% margin)
- Deployable to target hardware
- Good accuracy (94.67%) exceeds 90% requirement by +4.67%

### MynaNet v1 Trade-off:
- Sacrificed 1.0% accuracy to meet deployment constraint
- Strategic channel reduction: [120, 180, 146, 360] instead of [80, 160, 320, 640]
- Lighter MHSA: 2 heads instead of 4

---

## Why Was v1 Created?

Looking at the results directories:

```
results_macos/1e_..._darwin_9567/
  - INT8 Accuracy: 95.67%
  - Model Size: 579.3 KB  ← OVER 512KB TARGET

results_linux/1e_..._linux/
  - INT8 Accuracy: 94.83%
  - Model Size: 528.9 KB  ← OVER 512KB TARGET
```

**Model 1e consistently exceeds the 512KB deployment constraint**, even though it achieves higher accuracy.

**MynaNet v1 was created to meet the deployment constraint** while maintaining competitive accuracy.

---

## The Real Comparison

| Metric | Model 1e | MynaNet v1 | Winner |
|--------|----------|------------|--------|
| **INT8 Accuracy** | 95.67% | 94.67% | **1e** (+1.0%) |
| **Model Size** | 579KB | 434KB | **v1** (-145KB, -25%) |
| **Deployment Ready** | ❌ No | ✓ Yes | **v1** |
| **Size Margin** | -67KB (over) | +78KB (under) | **v1** |
| **Production Viable** | ❌ No | ✓ Yes | **v1** |

---

## Question: Can We Get Both?

**Hypothesis:** What if 1e's 95.67% is reproducible across seeds?

If yes, we could explore:

### Option A: Accept 1e if deployment allows
- If target platform has >579KB available
- Trade size for +1.0% accuracy
- Risk: Might not fit with future features

### Option B: Create "MynaNet v1.5" - Middle Ground
- Channels: [100, 150, 240, 480] (between 1e and v1)
- MHSA: 3 heads, 40 key_dim (between 1e and v1)
- Dense: 184 units (between 1e and v1)
- Target: ~500KB, aiming for 95%+ accuracy

### Option C: Stick with v1 (Current Decision)
- 94.67% already exceeds 90% target by +4.67%
- Proven deployment-ready
- 78KB safety margin for future optimizations

---

## Multiseed Verification Plan

Running **Model 1e with seeds [42, 100, 786]** to verify:

1. **Reproducibility:** Does 1e consistently achieve 95.67% ±0.2%?
2. **Stability:** Is 1e's performance robust across random seeds?
3. **Variance:** How does 1e compare to v1's stability?

**Command:**
```bash
chmod +x run_1e_multiseed.sh
nohup ./run_1e_multiseed.sh > 1e_multiseed_master.log 2>&1 &
```

**Expected Training Time:** ~9 hours (3 hours × 3 seeds)

---

## Decision Framework

After multiseed results:

### If 1e averages 95.5%+:
- **Consider Option B** (MynaNet v1.5 compromise)
- Target: 95%+ @ ~500KB
- Worth the optimization effort

### If 1e averages 95.0-95.5%:
- **Stick with Option C** (MynaNet v1)
- Marginal gain (<1%) not worth 145KB size increase
- v1 is production-ready now

### If 1e shows high variance (>0.5% std):
- **Definitely Option C** (MynaNet v1)
- Unstable models are deployment risks
- v1's proven stability preferred

---

## Conclusion (Current Understanding)

**Model 1e is NOT better than v1 in deployment context:**

1. **1e achieves 95.67%** ← Higher accuracy ✓
2. **1e uses 579KB** ← Exceeds deployment constraint ❌
3. **v1 achieves 94.67%** ← Slightly lower accuracy (still excellent)
4. **v1 uses 434KB** ← Meets deployment constraint ✓

**The "better" model depends on the constraint:**
- If size unconstrained: **1e wins** (+1.0% accuracy)
- If <512KB required: **v1 wins** (only deployable option)
- If <600KB allowed: **1e becomes viable** (accuracy-focused choice)

**Our original decision stands:** MynaNet v1 is the **production model** because it meets the deployment constraint while maintaining excellent accuracy.

**Next step:** Verify 1e's 95.67% is reproducible via multiseed training, then decide if a middle-ground architecture (v1.5) is worth exploring.

---

*Analysis created: February 8, 2026, 10:15 PM*
*Status: Launching multiseed verification*
