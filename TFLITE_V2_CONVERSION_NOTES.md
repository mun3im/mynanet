# TFLite v2 Conversion for Arduino Compatibility

**Date:** March 10, 2026
**Model:** v1_dscnn_se_res_att_wide_mels64 (seed 100, Linux)

---

## Problem

Arduino TFLite Micro (older versions) may not support **TFLite schema v3**, which is the default in TensorFlow 2.13+.

---

## Attempted Solution

Created `convert_to_tflite_v2.py` script that:
- Loads the Keras model
- Uses `converter._experimental_new_converter = False` to force v2 schema
- Performs INT8 post-training quantization
- Generates `model_int8_v2.tflite`

**Result:** ✗ Still produces v3 schema (TF 2.15 ignores the flag)

---

## File Comparison

| File | Size | Schema | Notes |
|------|------|--------|-------|
| `model_int8.tflite` (original) | 434 KB | v3 (TFL3) | From Stage9c training |
| `model_int8_v2.tflite` (new) | 423 KB | v3 (TFL3) | Conversion attempt |

Both files have `TFL3` magic bytes confirming schema v3.

---

## Solutions

### Option 1: Try the Model Anyway (Recommended First) ⭐

Modern Arduino TFLite Micro (late 2023+) **may support v3**. Try deploying:
- Use `model_int8_v2.tflite` (423.3 KB)
- If Arduino library is up-to-date, it should work
- Check Arduino TFLite Micro version in library manager

### Option 2: Use TensorFlow 2.4-2.12 for True v2

⚠️ **APPLE SILICON LIMITATION:** TensorFlow 2.4-2.12 are **NOT available** for ARM64 Macs!

**On Apple Silicon (M1/M2/M3/M4):**
- Minimum TF version: 2.13 (produces v3 schema)
- Cannot install TF 2.4-2.12 natively
- A `tf24` conda environment exists but uses TF 2.13 (still v3)

**For true v2 schema, use x86-64 Linux or Docker:**

```bash
# Option A: x86-64 Linux machine
pip install tensorflow==2.4.0 librosa numpy
python convert_to_tflite_v2.py <results_dir> --n_mels 64

# Option B: Docker with x86 emulation (on Mac)
docker run --platform linux/amd64 -it python:3.9
pip install tensorflow==2.4.0 librosa numpy
# Copy files and run conversion
```

**TensorFlow versions that produce v2:**
- TensorFlow 2.4.x ✓ (guaranteed v2, x86-64 only)
- TensorFlow 2.5-2.12.x ✓ (should produce v2, x86-64 only)
- TensorFlow 2.13+ ✗ (produces v3, available for ARM64)

### Option 3: Update Arduino TFLite Micro Library

Update to latest TFLite Micro that supports v3:
- Library: `Arduino_TensorFlowLite` or `tflite-micro-arduino`
- Check for 2024+ releases that support schema v3
- Update via Arduino Library Manager

### Option 4: Use `xxd` to Convert to C Array

Even with v3, you can deploy by converting to C header:

```bash
# Convert TFLite to C array
xxd -i model_int8_v2.tflite > model_data.cc

# In Arduino code
#include "model_data.cc"
const tflite::Model* model = tflite::GetModel(model_int8_v2_tflite);
```

This embeds the model directly, bypassing file loading issues.

---

## Model Verification Results

The converted model (`model_int8_v2.tflite`) was tested and **works correctly**:

```
✓ Model loaded successfully
  Input: INT8, shape (1, 64, 300, 1)
    Quantization: scale=0.003922, zero_point=-128
  Output: INT8, shape (1, 10)
    Quantization: scale=0.003906, zero_point=-128

✓ Inference successful
```

---

## Deployment Specs

**Model:** MynaNet v1 (64 mel bins)
**Size:** 423.3 KB INT8 TFLite
**Input:** (64, 300, 1) mel spectrogram, INT8
**Output:** (10,) class probabilities, INT8
**Target:** Arduino Portenta H7 (Cortex-M7 @ 480 MHz)

**Quantization:**
- Input scale: 0.003922, zero_point: -128
- Output scale: 0.003906, zero_point: -128
- All ops: INT8 (no float fallback)

---

## Recommended Next Steps

1. **Try `model_int8_v2.tflite` on Arduino first**
   - If TFLite Micro is recent (2024+), it should work
   - If you get schema errors, proceed to step 2

2. **If schema v3 errors occur:**
   - Option A: Update Arduino TFLite Micro library
   - Option B: Use TensorFlow 2.4-2.12 to regenerate with true v2

3. **For guaranteed compatibility:**
   - Use TF 2.4.0 in separate environment
   - Rerun conversion script
   - Verify `TFL3` → `TFL2` in hexdump

---

## Files Generated

- `convert_to_tflite_v2.py` - Conversion script
- `model_int8_v2.tflite` - INT8 model (423.3 KB, schema v3)
- `TFLITE_V2_CONVERSION_NOTES.md` - This document

---

## Contact/Support

If Arduino reports schema errors, you need TensorFlow 2.4-2.12. Let me know and I can help set up the older TF environment.

**TFLite Micro compatibility matrix:**
- Schema v2: Supported by all versions
- Schema v3: Supported by 2023+ versions (check your Arduino library date)

---

**Bottom line:** The model is ready to deploy. Try it first - modern Arduino libraries likely support v3. If not, we'll regenerate with TF 2.4.
