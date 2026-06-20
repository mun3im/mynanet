# MobileNet Variants Comparison (2b vs 2c)

## Quick Summary

| Aspect | 2b_mobilenet | 2c_mobilenetv3 |
|--------|--------------|----------------|
| **Architecture** | Basic MobileNet | MobileNetV3-Small |
| **Core Block** | Depthwise Separable Conv | Inverted Residual + SE |
| **Activation** | ReLU6 | Hard-swish |
| **Params** | ~80K (smallest) | ~1-2M (more capacity) |
| **Innovation** | 2017 (MobileNet v1) | 2019 (MobileNetV3) |
| **Complexity** | Simple, 3 blocks | Advanced, 7+ blocks |

---

## Architecture Details

### **2b_mobilenet_64x300.py** - Basic MobileNet

```
Input: (64, 300, 1)
↓
Conv2D(32, 3×3, relu6) → Dropout
↓
Block 1: DepthwiseConv(3×3) → Conv2D(64, 1×1, relu6) → MaxPool(2×2) → Dropout
         (64×300 → 32×150)
↓
Block 2: DepthwiseConv(3×3) → Conv2D(128, 1×1, relu6) → MaxPool(2×2) → Dropout
         (32×150 → 16×75)
↓
Block 3: DepthwiseConv(3×3) → Conv2D(256, 1×1, relu6) → MaxPool(2×2) → Dropout
         (16×75 → 8×37)
↓
GlobalAveragePooling2D
↓
Dense(128, relu6) → Dropout → Dense(10, softmax)
```

**Characteristics:**
- **Depthwise Separable**: DepthwiseConv (spatial) + Pointwise Conv (channels)
- **9× parameter reduction** vs standard Conv2D
- **Simple and fast**: Minimal overhead
- **Good baseline**: Classic MobileNet architecture

**Pros:**
- Smallest model (~80K params)
- Fast inference
- Well-understood architecture

**Cons:**
- Less capacity than MobileNetV3
- No channel attention (SE blocks)
- Older activation (ReLU6 vs Hard-swish)

---

### **2c_mobilenetv3_64x300.py** - MobileNetV3-Small (Adapted)

```
Input: (64, 300, 1)
↓
Conv2D(16, 3×3, stride=1, hard-swish) + BatchNorm  [ADAPTED: stride=1 not 2]
↓
Inverted Residual Blocks (7+ blocks):
  Block 1: exp=1,   filters=16, kernel=3, stride=2, SE=4,    relu      → 32×150
  Block 2: exp=4.5, filters=24, kernel=3, stride=2, SE=None, relu      → 16×75
  Block 3: exp=3.67,filters=24, kernel=3, stride=1, SE=None, relu      → 16×75
  Block 4: exp=4,   filters=40, kernel=5, stride=2, SE=4,    h-swish   → 8×37
  Block 5: exp=6,   filters=40, kernel=5, stride=1, SE=4,    h-swish   → 8×37
  Block 6: exp=6,   filters=40, kernel=5, stride=1, SE=4,    h-swish   → 8×37
  Block 7: exp=3,   filters=48, kernel=5, stride=1, SE=4,    h-swish   → 8×37
  ... (more blocks)
↓
GlobalAveragePooling2D
↓
Classification head → Dense(10, softmax)
```

**Characteristics:**
- **Inverted Residual**: Expand → DepthwiseConv → Project (MobileNetV2 innovation)
- **SE Blocks**: Squeeze-Excitation for channel attention
- **Hard-swish**: More efficient activation than ReLU6
- **NAS-optimized**: Neural Architecture Search tuned
- **Adaptive stride**: First conv stride=1 (adapted for 64×300, not 224×224)

**Pros:**
- SOTA mobile architecture (2019)
- Channel attention (SE blocks) improves accuracy
- Hard-swish activation better for quantization
- More capacity (~1-2M params)

**Cons:**
- Larger model than 2b
- More complex (harder to interpret)
- Slightly slower inference

---

## Key Differences

### 1. **Activation Functions**

**2b (ReLU6):**
```python
f(x) = min(max(0, x), 6)
```
- Simple, quantization-friendly
- Clips at 6 to prevent explosion

**2c (Hard-swish):**
```python
f(x) = x * ReLU6(x + 3) / 6
```
- More expressive than ReLU6
- Better gradient flow
- MobileNetV3 innovation

### 2. **Block Architecture**

**2b (Depthwise Separable):**
```
DepthwiseConv(3×3) → Pointwise(1×1)
```

**2c (Inverted Residual + SE):**
```
Expand(1×1) → DepthwiseConv(k×k) → SE Block → Project(1×1) → Residual
```
- Expansion: Increase channels first (opposite of bottleneck)
- SE Block: Channel attention (weighted features)
- Residual: Skip connection for gradient flow

### 3. **Squeeze-Excitation (SE) Blocks**

**2b:** No SE blocks

**2c:** SE blocks in selected layers
```
Input → GlobalAvgPool → Dense(reduce) → ReLU → Dense(expand) → Sigmoid → Scale
```
- **Purpose**: Channel attention (which features matter?)
- **Reduction ratio**: 4 (e.g., 40 channels → 10 → 40)
- **Cost**: Minimal overhead (~1-2% latency)
- **Benefit**: 1-3% accuracy improvement

### 4. **Parameter Count**

**2b:** ~80K parameters
- 3 depthwise blocks (64, 128, 256 filters)
- Dense(128) classification head

**2c:** ~1-2M parameters
- 7+ inverted residual blocks
- SE attention modules
- More sophisticated architecture

### 5. **Adaptation for 64×300**

**2b:** Natural fit
- Designed from scratch for 64×300
- MaxPool handles non-square input well

**2c:** Requires adaptation
- Standard MobileNetV3: 224×224 input, stride=2 first conv
- **Adapted**: stride=1 first conv to preserve resolution
- **Hypothesis**: 224×224 resize loses temporal/frequency detail

---

## Expected Performance

| Metric | 2b_mobilenet | 2c_mobilenetv3 |
|--------|--------------|----------------|
| **Accuracy** | ~88-92% | ~92-95% (expected) |
| **Params** | 80K | 1-2M |
| **Latency** | Fastest | Slower (3-5× vs 2b) |
| **Model Size (int8)** | ~20-30 KB | ~500KB-1MB |
| **Quantization** | Good (ReLU6 friendly) | Excellent (Hard-swish optimized) |

---

## Use Cases in TCN Paper

### **2b_mobilenet**: Lightweight CNN Baseline
- **Role**: Smallest efficient CNN baseline
- **Comparison**: "Can TCN match 2b accuracy with similar params?"
- **Expectation**: TCN should outperform due to temporal modeling

### **2c_mobilenetv3**: SOTA Mobile CNN
- **Role**: State-of-practice mobile architecture
- **Comparison**: "Does TCN match SOTA CNN with fewer params?"
- **Expectation**: TCN competitive accuracy, fewer params

### **Combined Story**:
1. **2b** shows TCN beats simple depthwise CNN
2. **2c** shows TCN matches SOTA mobile CNN (parameter-efficient)
3. **3a (Transformer)** shows TCN matches SOTA sequence model
4. **4a (TCN)** wins on temporal modeling + efficiency

---

## Implementation Notes

### 2b Key Code:
```python
def create_mobilenet_cnn(num_classes, input_shape, dropout=0.2):
    inputs = layers.Input(shape=input_shape)

    # Initial conv
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu6')(inputs)

    # Depthwise separable blocks
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu6')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu6')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ... repeat for 128, 256 filters
```

### 2c Key Code:
```python
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6

def create_mobilenetv3_small_64x300(num_classes, input_shape, dropout=0.2):
    inputs = layers.Input(shape=input_shape)

    # Adapted first conv (stride=1 instead of 2)
    x = layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(hard_swish)(x)

    # Inverted residual blocks with SE
    for exp, filters, kernel, stride, se_ratio, act in block_configs:
        x = inverted_residual_block(x, exp, filters, kernel, stride, se_ratio, act)
```

---

## Recommendation for TCN Paper

**Include both:**
- **2b**: Lightweight baseline (compare params with TCN)
- **2c**: SOTA mobile CNN (compare accuracy with TCN)

**Expected results table:**
| Model | Type | Accuracy | Params | Latency | Size |
|-------|------|----------|--------|---------|------|
| 2b_mobilenet | CNN | ~90% | 80K | Fast | 25KB |
| 2c_mobilenetv3 | CNN | ~94% | 1.5M | Medium | 800KB |
| **4a_tcn** | TCN | **~95%** | **300K** | **Medium** | **150KB** |

**Narrative**: TCN matches MobileNetV3 accuracy with 5× fewer params due to superior temporal modeling.

---

*Summary: 2b is simple/fast, 2c is SOTA/accurate. Both valuable baselines for TCN comparison.*
