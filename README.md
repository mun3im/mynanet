# MynaNet

Lightweight CNN for bird call classification targeting deployment on **Arduino Portenta H7** (Cortex-M7, 512 KB flash).

## Adopted Model: MynaNet = `1j` (MBV3-SE)

![MynaNet architecture](deploy/mynanet_architecture.svg)

**`1j_mbv3_se.py`** is the production MynaNet model.

- Architecture: MobileNetV3-style inverted residual blocks with 5×5 depthwise convolutions and hard-sigmoid Squeeze-Excitation
- **94.91% INT8** mean accuracy (3 seeds, Linux/CUDA authoritative)
- **267 KB INT8** — well within the 512 KB H7 flash limit
- MCU-deployable: all ops supported by TFLite Micro on Portenta H7

## Dataset

**mygardenbird16khz** — 12 garden bird species, 16 kHz  
Fixed 80:10:10 train/val/test split (CSV-based, no leakage)

## Repository layout

```
deploy/          ← production: train MynaNet, convert to firmware C array
  train.py           MynaNet (1j) training + INT8 quantization
  convert_xxd.sh     TFLite → alignas(8) C array for Portenta H7 firmware
  README.md          End-to-end guide: download → train → quantize → deploy

develop/         ← ablation: all Series 1 model scripts + sweep runner
  1a_baseline_2dcnn.py … 1n_efficientnetb0.py
  run_seabird12_ablation.sh
  README.md          Full ablation results and model comparison
```

## Quick start (deploy)

See [`deploy/README.md`](deploy/README.md) for the full end-to-end guide.

```bash
# Train MynaNet on mygardenbird16khz
python deploy/train.py \
  --flat_dir /path/to/mygardenbird16khz \
  --splits_csv /path/to/metadata16khz/splits_mip_80_10_10.csv

# Convert trained INT8 TFLite → firmware C array
bash deploy/convert_xxd.sh model_int8.tflite src/mynanet_model_data g_mynanet_model_data
```

## Ablation

See [`develop/README.md`](develop/README.md) for the full model comparison, results table, and key findings.

## License

MIT
