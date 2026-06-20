# MCU Deployment Analysis: MobileNetV3Small on Cortex-M7 @ 480 MHz

## 🔧 1. Model Overview
- **Architecture**: Custom MobileNetV3Small adapted for **64×300×1** input.
- **Key features**:
  - Depthwise separable convolutions (efficient).
  - Squeeze-and-Excitation (SE) blocks (lightweight global attention).
  - Hard-swish activations.
  - First convolution uses **stride=1** to preserve full 64×300 spatial resolution.
- **Final layers**:  
  GlobalAveragePooling → 1×1 Conv (1024) → Dropout → 1×1 Conv (num_classes).

---

## 📊 2. Model Size (INT8)
- **Estimated total parameters**: ~1.5 million.
- **INT8 model size**: ~1.5 MB / 4 ≈ **375 KB**.
- ✅ **Fits comfortably** within the 512 KB flash target for Cortex-M7 deployment.

---

## 🧮 3. Computational Complexity (MACs)
- **MACs (Multiply-Accumulate Operations)** dominate inference cost.
- Input size (**64×300 = 19,200 pixels**) is ~2.6× smaller than standard MobileNetV3 (224×224 = 50,176), but early layers operate at full resolution due to **stride=1** stem.

### Estimated MACs Breakdown:
| Layer Type                     | Output Size (H×W×C)      | Approx. MACs |
|-------------------------------|--------------------------|--------------|
| Stem conv (3×3, 16 filters)   | 64×300×16                | ~2.8M        |
| Inverted residual blocks (~11)| Gradual spatial reduction to ~4×18×96 | ~25–28M |
| Final 1×1 convs               | 1×1×1024 → num_classes   | ~0.6–1M      |

> ✅ **Total estimated INT8 MACs: ~30 million**

---

## ⏱️ 4. Inference Latency on Cortex-M7 @ 480 MHz

### Assumptions:
- Uses **CMSIS-NN** (optimized INT8 kernels for TFLite Micro).
- No dedicated NPU; relies on ARMv7E-M **DSP/SIMD extensions**.
- **Effective throughput**: ~0.7 MACs/cycle (realistic average accounting for depthwise, pointwise, SE, pooling, and memory overhead).

### Calculation:
- Total cycles ≈ 30,000,000 MACs ÷ 0.7 ≈ **43 million cycles**
- Latency = 43e6 ÷ 480e6 ≈ **0.089 seconds**

> ✅ **Estimated inference latency: 80–100 ms per frame**

> 📌 **Note**: Latency may increase if model/activations reside in external RAM (vs. TCM). Optimal performance requires TCM for weights and activations.

---

## ✅ Summary

| Metric                        | Estimate                     |
|------------------------------|------------------------------|
| **INT8 model size**          | ~375 KB                      |
| **Total MACs (INT8)**        | ~30 million                  |
| **Inference latency**        | **80–100 ms**                |
| **RAM (activation buffers)** | ~150–250 KB (TFLite Micro)   |
| **Feasible on Cortex-M7?**   | ✅ Yes (with CMSIS-NN + TCM) |

---

> ℹ️ For production: Profile with **TFLite Micro + CMSIS-NN** on target hardware to validate latency and memory usage.
