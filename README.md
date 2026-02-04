# MynaNet

Lightweight audio-focused CNN for bird call classification on edge devices (Cortex-M7 MCUs).

## Model Architecture

**DS-CNN + Squeeze-Excitation + Residual + Attention**
- ~420K parameters
- <512KB INT8 model size
- 64x300 mel spectrogram input (3 seconds @ 16kHz)

## Best Results

| Split | Train | Val | Test | INT8 Accuracy |
|-------|-------|-----|------|---------------|
| 80:10:10 | 4800 | 600 | 600 | **94.50%** |
| 75:10:15 | 4500 | 600 | 900 | 94.11% |
| 70:15:15 | 4200 | 900 | 900 | 92.56% |

## Training

```bash
python 1e_dscnn_se_res_att_wide.py \
  --splits_csv /path/to/splits.csv \
  --flat_dir /path/to/seabird16k_flat \
  --n_mels 64 --dropout 0.05 --mixup 0.2 --warmup_epochs 70
```

## Dataset

10 Southeast Asian bird species, 600 samples per class (6000 total).
Dataset creation and validation scripts are in [mun3im/seabird](https://github.com/mun3im/seabird).

## Results Directory Structure

```
results_linux/
  1e_dscnn_..._split80:10:10_linux/
    model_int8.tflite          # Quantized model for MCU
    model_fp32.keras           # Full precision model
    classification_report_*.txt
    confusion_matrix_*.png
    training_history.png
    seabird_splits_*_seed42.csv  # Reproducible split
```

## License

MIT
