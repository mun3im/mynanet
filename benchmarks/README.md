# MynaNet Benchmarking Scripts

Comprehensive benchmark suite comparing MynaNet (domain-specific, 267 KB) against:
- **Transfer learning models**: YAMNet (AudioSet), MatchBoxNet (Google Speech Commands)
- **Peer lightweight architectures**: SqueezeNet, ShuffleNetV2 (designed for edge deployment)
- **Analysis tools**: Per-class accuracy, SNR sensitivity, statistical significance

All benchmarks use the **12-class MyGardenBird dataset** (80:10:10 split, MIP source-disjoint).

## Datasets

### MyGardenBird 12-Class Dataset

**Location:** `/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz/`

**Species (12 classes):**
- Asian Koel
- Coppersmith Barbet
- Common Iora
- Common Tailorbird
- Pied Fantail
- Spotted Dove
- White-breasted Waterhen
- White-throated Kingfisher
- Collared Kingfisher
- Large-tailed Nightjar
- Yellow-vented Bulbul
- Olive-backed Sunbird

**Structure:**
```
mygardenbird16khz/
├── Asian Koel/
├── Coppersmith Barbet/
├── Common Iora/
├── ...
└── Olive-backed Sunbird/
```

Each directory contains `.wav` files at 16 kHz, 3-second duration clips.

## Train/Val/Test Splits

**Splits CSV:** `/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv`

**Format:**
```
# split_ratio=80:10:10 seed=42 objective=0 solver=mip_cbc
file_id,split
xc1002657_2860,train
xc1003831_2642,test
...
```

- `file_id`: Xeno-canto recording ID (e.g., `xc1002657_2860`)
- `split`: `train`, `val`, or `test`
- **MIP solver guarantee:** Each recording's source is isolated into a single split (0 leakage)

**Available split ratios:**
- `splits_mip_80_10_10.csv` (80% train, 10% val, 10% test) — **standard**
- `splits_mip_75_10_15.csv` (75% train, 10% val, 15% test)
- `splits_mip_70_15_15.csv` (70% train, 15% val, 15% test)

## Benchmark Scripts

### YAMNet Benchmarks

YAMNet is a pre-trained CNN for general audio classification (trained on AudioSet, 527 classes).

#### `yamnet_waveform_benchmark.py` (Recommended)

**Purpose:** Transfer learning benchmark using YAMNet embeddings + Random Forest classifier

**Approach:**
1. Load pre-trained YAMNet from TensorFlow Hub
2. Extract 5 overlapping 0.96s windows from each 3s clip (sliding window with 51-sample strides)
3. Run YAMNet on each window to get 128-dimensional embeddings
4. Aggregate embeddings via arithmetic mean
5. Train Random Forest (100 estimators) on pooled embeddings
6. Evaluate on held-out test set

**Results (12-class MyGardenBird):**
- YAMNet transfer learning: **65.97%** accuracy
- MynaNet domain-specific: **94.91%** INT8 accuracy
- **Gap: 28.94 percentage points** → domain specialization significantly outperforms transfer learning

**Usage:**
```bash
conda activate tf215_gpu
python benchmarks/yamnet_waveform_benchmark.py
```

**Requirements:**
- TensorFlow 2.15+ with GPU support
- tensorflow_hub
- librosa
- scikit-learn

**Output:**
- Prints detailed results, per-class accuracy breakdown
- Saves CSV results to `yamnet_benchmark_results.csv`

#### `yamnet_debug.py`

**Purpose:** Verify YAMNet input/output shapes and tensor compatibility

**Usage:**
```bash
python benchmarks/yamnet_debug.py
```

**Output:** Tests different input shapes, confirms YAMNet expects raw audio waveforms (1D float32), not mel spectrograms.

### Peer Architecture Benchmarks (Lightweight MCU-class models)

Lightweight CNN architectures designed for edge deployment.

#### `peer_architectures.py`

**Purpose:** Unified benchmark framework for SqueezeNet, ShuffleNetV2, and other lightweight models

**Supported architectures:**
- **4a: SqueezeNet v1.1** (58K py) — depthwise-separable with Fire modules
- **4b: ShuffleNetV2 v1.1** (50K py) — channel shuffling for efficiency
- **4c: ShuffleNetV2 Compact** (50K py) — reduced width variant

**Usage:**
```bash
python peer_architectures.py --model 4a --dataset mygardenbird --splits splits_mip_80_10_10.csv
```

**Output:**
- Per-model accuracy, F1-score, model size
- Comparison table vs MynaNet

#### `4a_squeezenet_v11.py`, `4b_shufflenetv2_v11.py`, `4c_shufflenetv2_compact.py`

**Purpose:** Individual training scripts for peer architectures on mygardenbird

**Usage:**
```bash
python 4a_squeezenet_v11.py --random_seed 42
python 4b_shufflenetv2_v11.py --random_seed 42
python 4c_shufflenetv2_compact.py --random_seed 42
```

### Analysis Scripts

#### `analyze_perclass_multiseed.py`

**Purpose:** Per-class accuracy analysis across multiple seeds

**Usage:**
```bash
python analyze_perclass_multiseed.py results_mygardenbird_4_linux/
```

**Output:** Per-species accuracy, variance, class-wise performance comparison

#### `analyze_perclass_snr.py`

**Purpose:** Analyze per-class accuracy as function of audio signal-to-noise ratio

**Usage:**
```bash
python analyze_perclass_snr.py results_mygardenbird_4_linux/
```

**Output:** SNR impact on classification accuracy by species

#### `analyze_significance.py`

**Purpose:** Statistical significance testing (t-tests, confidence intervals)

**Usage:**
```bash
python analyze_significance.py results_mygardenbird_4_linux/
```

**Output:** Significance results CSV with p-values, 95% CIs

### Batch Run Scripts

#### `run_peer_benchmarks.sh`

**Purpose:** Execute all peer architecture benchmarks with fixed configuration

**Usage:**
```bash
bash run_peer_benchmarks.sh
```

**Configuration (edit script):**
```bash
DATASET_DIR="/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SPLITS_CSV="/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
SEEDS=(42 100 786)  # 3-seed evaluation
```

#### `run_on_device_latency.sh`

**Purpose:** Measure inference latency on target device (Portenta H7 or similar)

**Usage:**
```bash
bash run_on_device_latency.sh mynanet_int8.tflite
```

**Output:** Latency statistics (mean, median, p95) in milliseconds

### MatchBoxNet Benchmarks

MatchBoxNet is a depthwise-separable CNN trained on Google Speech Commands (1-second speech, 30 classes).

#### `matchboxnet_transfer_analysis.py`

**Purpose:** Analytical transfer learning gap estimate (no actual training required)

**Approach:**
- Estimates MatchBoxNet GSC accuracy (~96% on 1-second speech)
- Applies domain shift penalty (speech → birds): -5 to -10pp
- Applies duration mismatch penalty (1s → 3s): -5 to -10pp
- Combined estimated penalty: ~16pp
- Predicted MatchBoxNet on 3s birds: ~80% accuracy

**Results (12-class MyGardenBird):**
- MatchBoxNet transfer (estimated): **~80%** accuracy
- MynaNet domain-specific: **94.91%** INT8 accuracy
- **Gap: ~14.9 percentage points** → domain-specific architecture advantage

**Usage:**
```bash
python benchmarks/matchboxnet_transfer_analysis.py
```

**Output:**
- Narrative text explaining transfer gap
- CSV file with transfer gap analysis

## Configuration

Edit the following constants in scripts to customize:

```python
# Dataset paths
DATASET_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SPLITS_CSV = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"

# Audio parameters
SAMPLE_RATE = 16000  # Hz
CLIP_DURATION = 3.0  # seconds
WINDOW_DURATION = 0.96  # seconds (YAMNet default ~1s)

# Training parameters
YAMNET_WINDOW_STARTS_S = [0.0, 0.51, 1.02, 1.53, 2.04]  # 5-window overlapping
```

## Results Summary

### Transfer Learning vs Domain-Specific (Main Finding)

| Model | Training | Approach | Accuracy | Model Size | Gap vs MynaNet |
|-------|----------|----------|----------|---|---|
| YAMNet | AudioSet | Transfer learning (embeddings + RF) | 65.97% | N/A | -28.94pp |
| MatchBoxNet | GSC | Transfer learning (estimated) | ~80% | N/A | -14.9pp |
| **MynaNet (1j)** | **MyGardenBird** | **Domain-specific, INT8** | **94.91%** | **267 KB** | **baseline** |

### Peer Lightweight Architectures (MCU-class)

| Model | Training | Accuracy (INT8) | Model Size | H7 Fit | Gap vs MynaNet |
|-------|----------|---|---|---|---|
| 4a SqueezeNet v1.1 | MyGardenBird | 91.81% | 809.5 KB | ✗ Over | -3.10pp |
| 4c ShuffleNetV2 Compact | MyGardenBird | 90.14% | 476.3 KB | ✓ Fits | -4.77pp |
| 4b ShuffleNetV2 v1.1 | MyGardenBird | 89.44% | 2299.2 KB | ✗ Way over | -5.47pp |
| **1j MynaNet** | **MyGardenBird** | **94.91%** | **267 KB** | **✓ Fits** | **baseline** |

**Key insight:** Among H7-deployable models (≤512 KB), MynaNet achieves the highest accuracy (94.91%) with the smallest footprint (267 KB), demonstrating optimal architecture design for the embedded bird classification task.

**Key Finding:** Domain-specific architecture design significantly outperforms transfer learning from general-purpose audio models, even when those models achieve state-of-the-art performance on their native tasks (AudioSet, GSC).

## Citation

If using these benchmarks, please cite:

```bibtex
@inproceedings{zabidi2026mynanet,
  title={MynaNet: A Compact, Deployable Deep Neural Network for Bird Sound Classification on Edge Devices},
  author={Zabidi, Muhammad Mun'im Ahmad},
  journal={Ecological Informatics},
  year={2026}
}

@inproceedings{gemmeke2017audio,
  title={AudioSet: An ontology and human-labeled dataset for research on large-scale sound event recognition},
  author={Gemmeke, Jort F. and others},
  booktitle={ICASSP},
  year={2017}
}

@inproceedings{majumdar2020matchboxnet,
  title={MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition},
  author={Majumdar, Somshubra and Ginsburg, Boris},
  booktitle={Proc.\ Interspeech},
  year={2020}
}
```

## Notes

- All benchmarks use the **80:10:10 train/val/test split** (MIP-optimized, source-disjoint)
- YAMNet inference is GPU-accelerated; CPU fallback available but slow (~30-60s per sample)
- Random Forest training uses parallel processing (`n_jobs=-1`); adjust for system RAM constraints
- Results are deterministic given fixed random seeds (numpy/sklearn `random_state=42`)

## Troubleshooting

**"FileNotFoundError: Dataset not found"**
- Verify `/Volumes/Evo/MYGARDENBIRD/` exists and is mounted
- Check that species folder names match exactly (e.g., "Asian Koel" not "Asian.Koel")

**"Module 'tensorflow_hub' not found"**
- Install: `pip install tensorflow_hub`

**"YAMNet expects 1D waveform, got 2D array"**
- Do NOT pass mel spectrograms to YAMNet; pass raw audio samples only
- See `yamnet_debug.py` for correct input shapes

**"No embeddings extracted (0 samples)"**
- Check that audio files load correctly with librosa
- Verify mel spectrogram extraction parameters match your audio processing pipeline

## License

Same as MynaNet paper (Ecological Informatics, 2026)
