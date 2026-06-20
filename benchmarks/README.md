# MynaNet Benchmarking Scripts

Transfer learning benchmarks comparing MynaNet (domain-specific) against pre-trained general-purpose audio models.

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

| Model | Training | Duration | Approach | Accuracy | Gap vs MynaNet |
|-------|----------|----------|----------|----------|---|
| YAMNet | AudioSet | 3s | Transfer learning (embeddings + RF) | 65.97% | -28.94pp |
| MatchBoxNet | GSC | 3s | Transfer learning (estimated) | ~80% | -14.9pp |
| **MynaNet (1j)** | **MyGardenBird** | **3s** | **Domain-specific, INT8** | **94.91%** | **baseline** |

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
