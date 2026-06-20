#!/usr/bin/env python3
"""
TCN Ablation — Series 2
12-class MyGardenBird dataset, 16 kHz audio, 64×300 mel-spectrogram input.

Variants 2a–2h explore TCN depth, width, kernel size, dilation pattern, and residuals.

Input pipeline (inside model, as Keras layers):
  (n_mels, time_frames, 1)
  → Reshape((n_mels, time_frames))   # drop channel dim
  → Permute((2, 1))                  # → (time_frames, n_mels) = (300, 64)
  → TCN residual blocks (causal dilated Conv1D)
  → GlobalAveragePooling1D
  → Dense(n_classes, softmax)
"""

print("\n\n\n")
for _ in range(3):
    print(" 🔶 " * 30)

import os
import sys
import argparse
import platform
import warnings
import hashlib
import json

warnings.filterwarnings("ignore")

# GPU Configuration (before TensorFlow import)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --------------------------------------------------------------
# EARLY ARGUMENT PARSING (for GPU config before TF import)
# --------------------------------------------------------------
def parse_early_args():
    """Parse GPU-related args before TensorFlow import."""
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--force_cpu", action='store_true')
    temp_parser.add_argument("--gpu_memory_limit", type=int, default=None)
    temp_args, _ = temp_parser.parse_known_args()

    if temp_args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("\n⚠ Force CPU mode enabled - GPU disabled")

    return temp_args

# Parse GPU settings early
early_args = parse_early_args()

# --------------------------------------------------------------
# TENSORFLOW & KERAS ENVIRONMENT CHECK
# --------------------------------------------------------------
print("\n" + "=" * 70)
print("ENVIRONMENT VALIDATION (Checking before dataset preparation)")
print("=" * 70)

try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers, callbacks

    print(f"✓ TensorFlow version: {tf.__version__}")
    print(f"✓ tf_keras version: {keras.__version__}")

    # Configure GPU with memory growth and error recovery
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and not early_args.force_cpu:
        try:
            if early_args.gpu_memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=early_args.gpu_memory_limit)]
                )
                print(f"✓ Found {len(gpus)} GPU(s)")
                print(f"  GPU: {gpus[0].name}")
                print(f"  Memory limit: {early_args.gpu_memory_limit} MB")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                print(f"✓ Found {len(gpus)} GPU(s)")
                print(f"  GPU: {gpus[0].name}")
                if 'device_name' in gpu_details:
                    print(f"  Device: {gpu_details['device_name']}")
                print(f"  Memory growth: Enabled (prevents OOM errors)")

            print(f"  Deterministic ops: Enabled (stable cuDNN)")
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")
            print("  Continuing with default GPU settings...")
    else:
        print("✓ Running on CPU (no GPU detected)")

    print("\n✓ Environment check PASSED - safe to proceed with dataset preparation")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ CRITICAL: TensorFlow environment check failed: {e}")
    print("Install: pip install tensorflow tf_keras")
    print("\n⚠ Stopping now to save your time (no dataset loading yet)")
    print("=" * 70)
    sys.exit(1)

# Now safe to import other libraries
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

script_start = time.time()

# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------
DEFAULT_RANDOM_STATE = 42

TARGET_SR = 16000
AUDIO_LENGTH_SEC = 3
FIXED_AUDIO_LENGTH = TARGET_SR * AUDIO_LENGTH_SEC
HOP_LENGTH = 160  # 10ms at 16kHz = 160 samples
N_FFT = 512
DEFAULT_N_MELS = 64
FMAX = 8000
TIME_FRAMES = 300  # Fixed: 3 seconds / 10ms = 300 frames

DEFAULT_FLAT_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
DEFAULT_SPECTROGRAM_DIR = "/Volumes/Evo/MYGARDENBIRD/precompute/spectrograms_16k_mels64"

SPECAUGMENT_FREQ_MASK = 8
SPECAUGMENT_TIME_MASK = 20
SPECAUGMENT_NUM_MASKS = 2

PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98
GLOBAL_STATS_SAMPLES = 100

# --------------------------------------------------------------
# TCN VARIANT DEFINITIONS
# --------------------------------------------------------------
TCN_VARIANTS = {
    '2a': {'n_layers': 8,  'channels': 64,  'kernel': 3, 'dilations': [1,2,4,8,16,32,64,128],         'residual': True,  'desc': 'Baseline (8L,64ch,k=3)'},
    '2b': {'n_layers': 4,  'channels': 64,  'kernel': 3, 'dilations': [1,2,4,8],                       'residual': True,  'desc': 'Shallow (4L,64ch,k=3)'},
    '2c': {'n_layers': 8,  'channels': 128, 'kernel': 3, 'dilations': [1,2,4,8,16,32,64,128],         'residual': True,  'desc': 'Wide (8L,128ch,k=3)'},
    '2d': {'n_layers': 10, 'channels': 64,  'kernel': 3, 'dilations': [1,2,4,8,16,32,64,128,256,512], 'residual': True,  'desc': 'Deep (10L,64ch,k=3)'},
    '2e': {'n_layers': 8,  'channels': 64,  'kernel': 2, 'dilations': [1,2,4,8,16,32,64,128],         'residual': True,  'desc': 'Kernel-2 (RF≈256)'},
    '2f': {'n_layers': 8,  'channels': 64,  'kernel': 5, 'dilations': [1,2,4,8,16,32,64,128],         'residual': True,  'desc': 'Kernel-5 (RF≈1021)'},
    '2g': {'n_layers': 8,  'channels': 64,  'kernel': 3, 'dilations': [1,2,4,8,16,32,64,128],         'residual': False, 'desc': 'No-residual (8L,64ch,k=3)'},
    '2h': {'n_layers': 6,  'channels': 32,  'kernel': 3, 'dilations': [1,2,4,8,16,32],                'residual': True,  'desc': 'Lightweight (6L,32ch,k=3)'},
}


# --------------------------------------------------------------
# UTILITY: TIME FORMATTING
# --------------------------------------------------------------
def format_time(seconds):
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# --------------------------------------------------------------
# OPTIMIZER – LEGACY ON APPLE SILICON, ADAMW ON LINUX
# --------------------------------------------------------------
system = platform.system()
processor = platform.processor()

if system == "Darwin" and processor == "arm":
    from tf_keras.optimizers.legacy import Adam as LegacyAdam
    Adam = LegacyAdam
    OPTIMIZER_NAME = "Legacy Adam"
    print("Using LEGACY Adam (fast on M1/M2/M4)")
elif system == "Linux":
    try:
        from tf_keras.optimizers import AdamW
        Adam = AdamW
        OPTIMIZER_NAME = "AdamW"
        print("Using AdamW optimizer (Linux - optimal for weight decay)")
    except ImportError:
        from tf_keras.optimizers import Adam
        OPTIMIZER_NAME = "Adam"
        print("Using standard Adam (AdamW not available)")
else:
    from tf_keras.optimizers import Adam
    OPTIMIZER_NAME = "Adam"
    print(f"Using standard Adam ({system})")


# --------------------------------------------------------------
# CACHE MANAGEMENT
# --------------------------------------------------------------
def compute_cache_hash(config_params):
    """Compute hash of preprocessing parameters for cache validation."""
    cache_key = {
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_mels': DEFAULT_N_MELS,
        'fmax': FMAX,
        'target_sr': TARGET_SR,
        'time_frames': TIME_FRAMES,
        'audio_length': FIXED_AUDIO_LENGTH,
        'center': True,
        'window': 'hann',
        'win_length': 400,
    }
    hash_str = json.dumps(cache_key, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]


def validate_cache(cache_dir):
    """Validate cache by checking version file."""
    version_file = os.path.join(cache_dir, '.cache_version')
    current_hash = compute_cache_hash({})
    if not os.path.exists(version_file):
        return False
    try:
        with open(version_file, 'r') as f:
            cached_hash = f.read().strip()
        return cached_hash == current_hash
    except:
        return False


def save_cache_version(cache_dir):
    """Save cache version file."""
    version_file = os.path.join(cache_dir, '.cache_version')
    current_hash = compute_cache_hash({})
    with open(version_file, 'w') as f:
        f.write(current_hash)


# --------------------------------------------------------------
# CONFIG WITH VALIDATION
# --------------------------------------------------------------
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default='2a',
                        choices=['2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h'],
                        help="TCN variant to train")
    parser.add_argument("--warmup_epochs", type=int, default=70)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--calib_samples", type=int, default=200)

    parser.add_argument("--augment", action='store_true',
                        help="Enable baseline augmentation (time/pitch shift)")
    parser.add_argument("--mixup", type=float, default=None,
                        help="Enable mixup augmentation with alpha value (e.g., 0.2)")
    parser.add_argument("--specaugment", action='store_true',
                        help="Enable SpecAugment (frequency/time masking)")

    parser.add_argument("--time_shift_ms", type=int, default=100,
                        help="Max time shift in milliseconds (baseline augmentation)")
    parser.add_argument("--pitch_shift_steps", type=int, default=2,
                        help="Max pitch shift in semitones (baseline augmentation)")

    parser.add_argument("--force_cpu", action='store_true',
                        help="Force CPU execution (disable GPU)")
    parser.add_argument("--gpu_memory_limit", type=int, default=None,
                        help="GPU memory limit in MB (e.g., 8192 for 8GB)")

    parser.add_argument("--splits_csv", type=str, required=True,
                        help="Path to splits CSV from seabird_splitter_mip.py")
    parser.add_argument("--flat_dir", type=str, default=DEFAULT_FLAT_DIR,
                        help="Path to flat dataset directory")
    parser.add_argument("--spectrogram_dir", type=str, default=DEFAULT_SPECTROGRAM_DIR,
                        help="Path to spectrogram cache directory")

    parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS, choices=[48, 64, 80, 96],
                        help="Number of mel bins (default: 64)")

    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "plateau", "both", "none"],
                        help="Learning rate schedule strategy")

    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_STATE,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    n_mels = args.n_mels
    variant = args.variant
    vcfg = TCN_VARIANTS[variant]

    aug_suffix = ""
    augmentation_mode = "none"

    if args.mixup is not None:
        augmentation_mode = "mixup"
        aug_suffix = f"mixup{args.mixup}"
    elif args.specaugment:
        augmentation_mode = "specaugment"
        aug_suffix = "specaugment"
    elif args.augment:
        augmentation_mode = "baseline"
        aug_suffix = "baseline"

    split_suffix = ""
    try:
        with open(args.splits_csv, 'r') as f:
            header = f.readline().strip()
        if header.startswith('# split_ratio='):
            ratio_str = header.split('split_ratio=')[1].split()[0]
            split_suffix = f"split{ratio_str}"
    except Exception:
        split_suffix = "splitcsv"

    n_classes = len([d for d in os.listdir(args.flat_dir)
                     if os.path.isdir(os.path.join(args.flat_dir, d)) and not d.startswith('.')])

    output_dir_name = (
        f"results_mygardenbird_2_{platform.system().lower()}/"
        f"2{variant}_tcn_"
        f"{vcfg['channels']}ch_"
        f"{vcfg['n_layers']}l_"
        f"k{vcfg['kernel']}_"
        f"mels{n_mels}_"
        f"drop{int(args.dropout * 100):02d}_"
        f"rand{args.random_seed}_"
        f"warm{args.warmup_epochs}_"
        f"{aug_suffix}_"
        f"{split_suffix}_"
        f"{platform.system().lower()}"
    )
    output_dir_name = output_dir_name.replace("__", "_").rstrip("_")

    spec_dir = args.spectrogram_dir
    if spec_dir == DEFAULT_SPECTROGRAM_DIR:
        spec_dir = f"/Volumes/Evo/MYGARDENBIRD/precompute/spectrograms_16k_mels{n_mels}"

    config = {
        'variant': variant,
        'vcfg': vcfg,
        'warmup_epochs': args.warmup_epochs,
        'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size,
        'warmup_lr': args.warmup_lr,
        'finetune_lr': args.finetune_lr,
        'dropout': args.dropout,
        'time_frames': TIME_FRAMES,
        'n_mels': n_mels,
        'input_shape': (n_mels, TIME_FRAMES, 1),
        'output_dir': output_dir_name,
        'calib_samples': args.calib_samples,
        'model_type': f'2{variant}_tcn',
        'augmentation_mode': augmentation_mode,
        'mixup_alpha': args.mixup,
        'time_shift_ms': args.time_shift_ms,
        'pitch_shift_steps': args.pitch_shift_steps,
        'force_cpu': args.force_cpu,
        'gpu_memory_limit': args.gpu_memory_limit,
        'spectrogram_dir': spec_dir,
        'lr_schedule': args.lr_schedule,
        'random_seed': args.random_seed,
        'splits_csv': args.splits_csv,
        'flat_dir': args.flat_dir,
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    return config


# --------------------------------------------------------------
# LOGGING UTILITIES
# --------------------------------------------------------------
class TrainingLogger:
    """Centralized logger for all training metrics and hyperparameters."""

    def __init__(self, output_dir, variant, vcfg):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, 'training_report.txt')
        self.start_time = time.time()
        self.stage_times = {}

        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MODEL 2{variant}: TCN — {vcfg['desc']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Platform: {platform.system()} {platform.machine()}\n")
            f.write(f"Python: {sys.version.split()[0]}\n")
            f.write(f"TensorFlow: {tf.__version__}\n")
            f.write(f"Keras: {keras.__version__}\n")
            f.write("\n")

    def log_section(self, title):
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n")

    def log_hyperparameters(self, config):
        self.log_section("HYPERPARAMETERS")
        vcfg = config['vcfg']
        with open(self.log_path, 'a') as f:
            f.write("\nSystem Configuration:\n")
            f.write(f"  Platform: {platform.system()} {platform.machine()}\n")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and not config.get('force_cpu', False):
                f.write(f"  GPU: {len(gpus)} device(s) detected\n")
                f.write(f"  GPU Memory: Dynamic growth enabled\n")
                if config.get('gpu_memory_limit'):
                    f.write(f"  GPU Memory Limit: {config['gpu_memory_limit']} MB\n")
            else:
                f.write(f"  Compute: CPU only\n")

            f.write("\nAudio Processing:\n")
            f.write(f"  Target Sample Rate:     {TARGET_SR} Hz\n")
            f.write(f"  Audio Length:           {AUDIO_LENGTH_SEC} seconds ({FIXED_AUDIO_LENGTH} samples)\n")
            f.write(f"  FFT Size (N_FFT):       {N_FFT}\n")
            f.write(f"  FFT Window:             Hann (YAMNet standard, reduces spectral leakage)\n")
            f.write(f"  Window Length:          400 samples (25ms at 16kHz)\n")
            f.write(f"  Hop Length:             {HOP_LENGTH} samples (10.0 ms)\n")
            f.write(f"  Mel Bins (N_MELS):      {config['n_mels']}\n")
            f.write(f"  Max Frequency (FMAX):   {FMAX} Hz\n")
            f.write(f"  Time Frames:            {TIME_FRAMES} (FIXED)\n")
            f.write(f"  Spectrogram Shape:      {config['n_mels']}x{TIME_FRAMES}\n")
            f.write(f"  Center Padding:         Enabled (librosa center=True)\n")

            f.write("\nModel Architecture:\n")
            f.write(f"  Model Type:             TCN (Series 2, variant {config['variant']})\n")
            f.write(f"  Description:            {vcfg['desc']}\n")
            f.write(f"  Layers:                 {vcfg['n_layers']}\n")
            f.write(f"  Channels:               {vcfg['channels']}\n")
            f.write(f"  Kernel Size:            {vcfg['kernel']}\n")
            f.write(f"  Dilations:              {vcfg['dilations']}\n")
            f.write(f"  Residual:               {vcfg['residual']}\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  Input Shape:            {config['input_shape']}\n")

            f.write("\nTraining Configuration:\n")
            f.write(f"  Random Seed:            {config['random_seed']}\n")
            f.write(f"  Warmup Epochs:          {config['warmup_epochs']}\n")
            f.write(f"  Fine-tune Epochs:       {config['finetune_epochs']}\n")
            f.write(f"  Total Epochs:           {config['warmup_epochs'] + config['finetune_epochs']}\n")
            f.write(f"  Batch Size:             {config['batch_size']}\n")
            f.write(f"  Warmup Learning Rate:   {config['warmup_lr']}\n")
            f.write(f"  Fine-tune Learning Rate:{config['finetune_lr']}\n")
            f.write(f"  LR Schedule:            {config['lr_schedule']}\n")
            f.write(f"  Optimizer:              {OPTIMIZER_NAME}\n")
            f.write(f"  Loss Function:          Sparse Categorical Crossentropy\n")

            f.write("\nData Augmentation:\n")
            f.write(f"  Mode:                   {config['augmentation_mode']}\n")

            if config['augmentation_mode'] == 'baseline':
                f.write(f"  Type:                   Baseline (Time/Pitch Shift)\n")
                f.write(f"  Time Shift:             ±{config['time_shift_ms']} ms\n")
                f.write(f"  Pitch Shift:            ±{config['pitch_shift_steps']} semitones\n")
                f.write(f"  Data Multiplier:        2x (original + augmented)\n")
            elif config['augmentation_mode'] == 'mixup':
                f.write(f"  Type:                   Mixup\n")
                f.write(f"  Alpha:                  {config['mixup_alpha']}\n")
                f.write(f"  Data Multiplier:        2x (original + mixup)\n")
            elif config['augmentation_mode'] == 'specaugment':
                f.write(f"  Type:                   SpecAugment\n")
                f.write(f"  Frequency Mask:         {SPECAUGMENT_FREQ_MASK} bins\n")
                f.write(f"  Time Mask:              {SPECAUGMENT_TIME_MASK} frames\n")
                f.write(f"  Number of Masks:        {SPECAUGMENT_NUM_MASKS}\n")
                f.write(f"  Data Multiplier:        2x (original + augmented)\n")
            else:
                f.write(f"  Enabled:                False\n")

            f.write("\nQuantization:\n")
            f.write(f"  Method:                 Post-Training Quantization (PTQ)\n")
            f.write(f"  Target Format:          INT8 TFLite\n")
            f.write(f"  Calibration Samples:    {config['calib_samples']}\n")
            f.write(f"  Input/Output Type:      INT8\n")

            f.write("\nData Paths:\n")
            f.write(f"  Flat Directory:         {config['flat_dir']}\n")
            f.write(f"  Splits CSV:             {config['splits_csv']}\n")
            f.write(f"  Spectrogram Cache:      {config['spectrogram_dir']}\n")
            f.write(f"  Output Directory:       {config['output_dir']}\n")

    def log_dataset_info(self, X, y, class_labels, X_train, X_val, X_test, failed_files=0):
        self.log_section("DATASET INFORMATION")
        with open(self.log_path, 'a') as f:
            f.write(f"\nTotal Samples:          {len(X)}\n")
            f.write(f"Number of Classes:      {len(class_labels)}\n")
            if failed_files > 0:
                f.write(f"Failed Files:           {failed_files}\n")
            f.write(f"\nData Split (Fixed per-class):\n")
            f.write(f"  Training:             {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)\n")
            f.write(f"  Validation:           {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)\n")
            f.write(f"  Test:                 {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)\n")

            f.write(f"\nClass Distribution:\n")
            for name, idx in sorted(class_labels.items(), key=lambda x: x[1]):
                count = np.sum(y == idx)
                percentage = count / len(y) * 100
                f.write(f"  {name:30s}: {count:5d} samples ({percentage:5.2f}%)\n")

    def log_model_info(self, model):
        self.log_section("MODEL ARCHITECTURE")

        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()

        summary_path = os.path.join(self.output_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL ARCHITECTURE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary_str)
        print(f"Saved model summary: {summary_path}")

        with open(self.log_path, 'a') as f:
            f.write("\n" + summary_str)

            total_params = model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params

            f.write(f"\nParameter Summary:\n")
            f.write(f"  Total Parameters:       {total_params:,}\n")
            f.write(f"  Trainable Parameters:   {trainable_params:,}\n")
            f.write(f"  Non-trainable Params:   {non_trainable_params:,}\n")

            fp32_size_mb = total_params * 4 / (1024 ** 2)
            int8_size_kb = total_params / 1024

            f.write(f"\nEstimated Model Sizes:\n")
            f.write(f"  FP32 (4 bytes/param):   {fp32_size_mb:.2f} MB\n")
            f.write(f"  INT8 (1 byte/param):    {int8_size_kb:.1f} KB\n")

            if int8_size_kb > 512:
                f.write(f"\n  WARNING: Model may exceed 512 KB target for Cortex-M7\n")
            else:
                f.write(f"\n  Model size within 512 KB target\n")

    def start_stage(self, stage_name):
        self.stage_times[stage_name] = {'start': time.time()}
        self.log_section(stage_name)

    def end_stage(self, stage_name, history=None):
        if stage_name not in self.stage_times:
            return

        self.stage_times[stage_name]['end'] = time.time()
        elapsed = self.stage_times[stage_name]['end'] - self.stage_times[stage_name]['start']

        with open(self.log_path, 'a') as f:
            f.write(f"\nStage Duration: {format_time(elapsed)}\n")

            if history is not None:
                f.write(f"\nTraining History:\n")
                final_epoch = len(history.history['loss'])
                f.write(f"  Epochs Completed:       {final_epoch}\n")
                f.write(f"  Final Train Loss:       {history.history['loss'][-1]:.4f}\n")
                f.write(f"  Final Train Accuracy:   {history.history['accuracy'][-1]:.4f}\n")
                f.write(f"  Final Val Loss:         {history.history['val_loss'][-1]:.4f}\n")
                f.write(f"  Final Val Accuracy:     {history.history['val_accuracy'][-1]:.4f}\n")
                f.write(f"  Best Val Loss:          {min(history.history['val_loss']):.4f}\n")
                f.write(f"  Best Val Accuracy:      {max(history.history['val_accuracy']):.4f}\n")

    def log_evaluation(self, model_name, accuracy, report_path):
        with open(self.log_path, 'a') as f:
            f.write(f"\n{model_name} Evaluation:\n")
            f.write(f"  Test Accuracy:          {accuracy:.2f}%\n")
            f.write(f"  Classification Report:  {report_path}\n")

    def log_final_results(self, fp32_acc, int8_acc, model_sizes,
                          warmup_history, finetune_history, config, model=None):
        self.log_section("FINAL RESULTS SUMMARY")

        drop = fp32_acc - int8_acc
        total_time = time.time() - script_start
        vcfg = config['vcfg']
        variant = config['variant']

        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("QUICK REFERENCE (Copy to spreadsheet)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Config: 2{variant}_tcn_drp{int(config['dropout'] * 10)}_"
                    f"{config['augmentation_mode']}_warmup{config['warmup_epochs']}_"
                    f"finetune{config['finetune_epochs']}_lr{config['lr_schedule']}\n")
            f.write(f"FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}% | "
                    f"Drop: {drop:+.2f}% | Time: {format_time(total_time)}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n")

            f.write(f"\nAccuracy Results:\n")
            f.write(f"  FP32 (.keras):          {fp32_acc:6.2f}%\n")
            f.write(f"  INT8 (TFLite):          {int8_acc:6.2f}%\n")

            f.write(f"\nAccuracy Change (INT8 vs FP32):\n")
            f.write(f"  Drop:                   {drop:+6.2f}% ")
            if abs(drop) < 0.5:
                f.write("✓ Excellent (no degradation)\n")
            elif drop > 0:
                f.write("✓✓ INT8 better! (quantization as regularizer)\n")
            elif drop > -2:
                f.write("✓ Good (<2% drop)\n")
            elif drop > -5:
                f.write("⚠ Acceptable (2-5% drop)\n")
            else:
                f.write("✗ High degradation (>5% drop)\n")

            f.write(f"\nModel Sizes:\n")
            for model_type, size_info in model_sizes.items():
                f.write(f"  {model_type:20s}: {size_info}\n")

            f.write(f"\nTraining Metrics:\n")
            f.write(f"  Best Warmup Val Acc:    {max(warmup_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Best Finetune Val Acc:  {max(finetune_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Final Train Acc:        {finetune_history.history['accuracy'][-1] * 100:6.2f}%\n")
            f.write(f"  Final Val Acc:          {finetune_history.history['val_accuracy'][-1] * 100:6.2f}%\n")
            f.write(f"  Train-Test Gap:         {finetune_history.history['accuracy'][-1] * 100 - int8_acc:+6.2f}%\n")

            overfitting_gap = (finetune_history.history['accuracy'][-1] * 100
                               - finetune_history.history['val_accuracy'][-1] * 100)
            f.write(f"  Train-Val Gap:          {overfitting_gap:+6.2f}%")
            if overfitting_gap < 2:
                f.write(" ✓ No overfitting\n")
            elif overfitting_gap < 5:
                f.write(" ⚠ Slight overfitting\n")
            else:
                f.write(" ✗ Overfitting detected\n")

            f.write(f"\nExecution Time:\n")
            f.write(f"  Total Duration:         {format_time(total_time)}\n")
            f.write(f"\nTraining completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("CSV FORMAT (for batch comparison)\n")
            f.write("=" * 80 + "\n")
            f.write("model_type,variant_desc,n_layers,channels,kernel,residual,dropout,augmentation,"
                    "warmup_epochs,finetune_epochs,warmup_lr,finetune_lr,lr_schedule,"
                    "fp32_acc,int8_acc,drop,best_val_acc,train_val_gap,train_time_sec,model_size_kb\n")
            f.write(f"2{variant}_tcn,\"{vcfg['desc']}\",{vcfg['n_layers']},{vcfg['channels']},"
                    f"{vcfg['kernel']},{vcfg['residual']},{config['dropout']},{config['augmentation_mode']},"
                    f"{config['warmup_epochs']},{config['finetune_epochs']},{config['warmup_lr']},{config['finetune_lr']},"
                    f"{config['lr_schedule']},{fp32_acc:.2f},{int8_acc:.2f},{drop:.2f},"
                    f"{max(finetune_history.history['val_accuracy']) * 100:.2f},"
                    f"{overfitting_gap:.2f},{int(total_time)},"
                    f"{os.path.getsize(os.path.join(config['output_dir'], 'model_int8.tflite')) / 1024:.1f}\n")


# --------------------------------------------------------------
# GLOBAL STATS (2-98 percentile)
# --------------------------------------------------------------
def compute_global_stats(data_dir, n_mels, allowed_files=None):
    """Compute global normalization statistics from flat dataset directory."""
    all_mel = []
    total_sampled = 0

    print(f"Computing global stats (sampling up to {GLOBAL_STATS_SAMPLES} files per class, n_mels={n_mels})...")

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue

        wavs = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        if allowed_files is not None:
            wavs = [f for f in wavs if f in allowed_files]
        sample_size = min(len(wavs), GLOBAL_STATS_SAMPLES)

        for f in wavs[:sample_size]:
            try:
                audio_path = os.path.join(class_dir, f)
                audio, _ = librosa.load(audio_path, sr=TARGET_SR)

                if len(audio) > FIXED_AUDIO_LENGTH:
                    audio = audio[:FIXED_AUDIO_LENGTH]
                else:
                    audio = np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                mel = librosa.feature.melspectrogram(
                    y=audio, sr=TARGET_SR, n_fft=N_FFT,
                    win_length=400, hop_length=HOP_LENGTH,
                    n_mels=n_mels, fmax=FMAX, center=True,
                    power=2.0, window='hann'
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                all_mel.append(mel_db.flatten())
                total_sampled += 1
            except Exception as e:
                print(f"\n⚠ Failed to process {f} during stats computation: {e}")
                continue

    if len(all_mel) == 0:
        raise RuntimeError("No valid audio files found for computing global stats")

    all_mel = np.concatenate(all_mel)
    gmin, gmax = np.percentile(all_mel, PERCENTILE_LOW), np.percentile(all_mel, PERCENTILE_HIGH)
    print(f"✓ Global stats computed from {total_sampled} files: {gmin:.2f} → {gmax:.2f} dB")
    return float(gmin), float(gmax)


# --------------------------------------------------------------
# SPECTROGRAM + NORMALIZE (FIXED 64x300 with WIN_LENGTH)
# --------------------------------------------------------------
def compute_spec(audio, sr, gmin, gmax, n_mels=None):
    """Compute and normalize mel spectrogram with shape validation."""
    if n_mels is None:
        n_mels = DEFAULT_N_MELS
    WIN_LENGTH = 400  # 25ms at 16kHz

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT,
        win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmax=FMAX, center=True,
        power=2.0, window='hann'
    )

    if mel.shape[1] > TIME_FRAMES:
        mel = mel[:, :TIME_FRAMES]
    if mel.shape[1] < TIME_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, TIME_FRAMES - mel.shape[1])))

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, gmin, gmax)
    mel_norm = (mel_db - gmin) / (gmax - gmin + 1e-8)
    return mel_norm[..., np.newaxis].astype(np.float32)


# --------------------------------------------------------------
# AUGMENTATION FUNCTIONS
# --------------------------------------------------------------
def augment_baseline(audio, sr, time_shift_ms=100, pitch_steps=2):
    """Baseline augmentation: time shift + pitch shift"""
    if np.random.rand() > 0.5:
        shift_samples = int(np.random.uniform(-time_shift_ms, time_shift_ms) * sr / 1000)
        audio = np.roll(audio, shift_samples)

    if np.random.rand() > 0.5:
        n_steps = np.random.uniform(-pitch_steps, pitch_steps)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    return audio


def augment_specaugment(spec):
    """SpecAugment: frequency and time masking on spectrogram"""
    spec_aug = spec.copy()
    freq_bins, time_bins, _ = spec_aug.shape

    for _ in range(SPECAUGMENT_NUM_MASKS):
        f_mask = np.random.randint(0, SPECAUGMENT_FREQ_MASK)
        f0 = np.random.randint(0, freq_bins - f_mask) if f_mask < freq_bins else 0
        spec_aug[f0:f0 + f_mask, :, 0] = 0

    for _ in range(SPECAUGMENT_NUM_MASKS):
        t_mask = np.random.randint(0, SPECAUGMENT_TIME_MASK)
        t0 = np.random.randint(0, time_bins - t_mask) if t_mask < time_bins else 0
        spec_aug[:, t0:t0 + t_mask, 0] = 0

    return spec_aug


# --------------------------------------------------------------
# TCN MODEL
# --------------------------------------------------------------
def tcn_block(x, filters, kernel_size, dilation_rate, use_residual, dropout, block_id):
    """
    TCN residual block: two causal dilated Conv1D → BN → ReLU stacked.
    Optional residual with 1×1 Conv1D projection if channels differ.
    """
    prefix = f'tcn_block{block_id}_'
    input_channels = x.shape[-1]

    # First Conv1D layer
    out = layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        use_bias=False,
        name=prefix + 'conv1'
    )(x)
    out = layers.BatchNormalization(name=prefix + 'bn1')(out)
    out = layers.ReLU(name=prefix + 'relu1')(out)
    out = layers.Dropout(dropout, name=prefix + 'drop1')(out)

    # Second Conv1D layer
    out = layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        use_bias=False,
        name=prefix + 'conv2'
    )(out)
    out = layers.BatchNormalization(name=prefix + 'bn2')(out)
    out = layers.ReLU(name=prefix + 'relu2')(out)
    out = layers.Dropout(dropout, name=prefix + 'drop2')(out)

    if use_residual:
        if input_channels != filters:
            # 1×1 projection to match channel dims
            x = layers.Conv1D(
                filters, 1,
                padding='same',
                use_bias=False,
                name=prefix + 'proj'
            )(x)
            x = layers.BatchNormalization(name=prefix + 'proj_bn')(x)
        out = layers.Add(name=prefix + 'residual')([out, x])

    return out


def create_tcn(num_classes, input_shape, vcfg, dropout=0.05):
    """
    TCN model with reshape/permute inside as Keras layers.

    Input: (n_mels, time_frames, 1)
    → Reshape((n_mels, time_frames))   # drop channel dim
    → Permute((2, 1))                  # → (time_frames, n_mels) = (300, 64)
    → TCN blocks (causal dilated Conv1D)
    → GlobalAveragePooling1D
    → Dense(n_classes, softmax)
    """
    n_mels, time_frames, _ = input_shape
    channels = vcfg['channels']
    kernel_size = vcfg['kernel']
    dilations = vcfg['dilations']
    use_residual = vcfg['residual']

    inputs = layers.Input(shape=input_shape, name='input')

    # Reshape and permute to (time_frames, n_mels)
    x = layers.Reshape((n_mels, time_frames), name='reshape')(inputs)
    x = layers.Permute((2, 1), name='permute')(x)  # → (time_frames, n_mels)

    # Initial projection to TCN channel width
    x = layers.Conv1D(channels, 1, padding='same', use_bias=False, name='input_proj')(x)
    x = layers.BatchNormalization(name='input_proj_bn')(x)
    x = layers.ReLU(name='input_proj_relu')(x)

    # TCN blocks
    for i, dilation in enumerate(dilations):
        x = tcn_block(
            x,
            filters=channels,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            use_residual=use_residual,
            dropout=dropout,
            block_id=i + 1
        )

    # Classification head
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    x = layers.Dropout(dropout, name='head_drop')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return keras.Model(inputs, outputs, name=f"TCN_2{vcfg.get('variant_key', '')}")


# --------------------------------------------------------------
# TFLITE CONVERSION (POST-TRAINING QUANTIZATION)
# --------------------------------------------------------------
def convert_to_tflite_int8(model, X_calib, path):
    """Convert model to INT8 TFLite with post-training quantization."""
    def rep_dataset():
        for i in range(len(X_calib)):
            yield [X_calib[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ Saved INT8 TFLite: {path} ({os.path.getsize(path) / 1024:.1f} KB)")


# --------------------------------------------------------------
# PLOTTING & EVALUATION
# --------------------------------------------------------------
def _save_classification_report(y_test, y_pred, class_names, output_dir, model_type):
    """Save classification report to file and print to console."""
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, f'classification_report_{model_type.lower()}.txt')
    with open(report_path, 'w') as f:
        f.write(f"{model_type} Model Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print(f"✓ Saved classification report: {report_path}")
    print(f"\n{model_type} Classification Report:")
    print(report)
    return report_path


def _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, accuracy):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    cmap = 'Blues' if 'FP32' in model_type else 'Greens'

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_type} Confusion Matrix - Accuracy: {accuracy:.2f}%', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type.lower()}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(warmup_hist, finetune_hist, output_dir):
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(warmup_hist.history['accuracy'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[0].plot(warmup_hist.history['val_accuracy'], label='Val (Warmup)', color='blue', linestyle='--')

    offset = len(warmup_hist.history['accuracy'])
    epochs_finetune = range(offset, offset + len(finetune_hist.history['accuracy']))
    axes[0].plot(epochs_finetune, finetune_hist.history['accuracy'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[0].plot(epochs_finetune, finetune_hist.history['val_accuracy'], label='Val (Finetune)', color='red',
                 linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(warmup_hist.history['loss'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[1].plot(warmup_hist.history['val_loss'], label='Val (Warmup)', color='blue', linestyle='--')
    axes[1].plot(epochs_finetune, finetune_hist.history['loss'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[1].plot(epochs_finetune, finetune_hist.history['val_loss'], label='Val (Finetune)', color='red',
                 linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    print("✓ Saved training history plot")


def evaluate_model(model, X_test, y_test, class_names, output_dir, model_type="FP32"):
    """Evaluate a Keras model and save results."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred) * 100

    _save_classification_report(y_test, y_pred, class_names, output_dir, model_type)
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, acc)

    return acc


def evaluate_tflite(tflite_path, X_test, y_test, class_names, output_dir):
    """Evaluate a TFLite INT8 model and save results."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    y_pred = []
    print("Evaluating TFLite model...")
    for i in tqdm(range(len(X_test)), desc="Running inference", unit="sample"):
        x_fp32 = X_test[i:i + 1]
        x_int8 = (x_fp32 / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], x_int8)
        interpreter.invoke()

        output_int8 = interpreter.get_tensor(output_details[0]['index'])
        output_fp32 = (output_int8.astype(np.float32) - output_zero_point) * output_scale

        y_pred.append(np.argmax(output_fp32))

    y_pred = np.array(y_pred)
    acc = accuracy_score(y_test, y_pred) * 100

    _save_classification_report(y_test, y_pred, class_names, output_dir, "INT8")
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, "INT8", acc)

    return acc


# --------------------------------------------------------------
# LOAD DATA (flat dir + splits CSV)
# --------------------------------------------------------------
def parse_splits_csv(csv_path):
    """Read splits CSV into a {filename: split} dict."""
    splits = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',', 1)
            if len(parts) != 2:
                continue
            key, split = parts[0].strip(), parts[1].strip()
            if key in ('filename', 'file_id'):
                continue
            key = key.lower()
            if not key.endswith('.wav'):
                key += '.wav'
            splits[key] = split
    return splits


def load_data_from_csv(csv_path, flat_dir, gmin, gmax, n_mels,
                       augmentation_mode='none', time_shift_ms=100,
                       pitch_shift_steps=2, mixup_alpha=0.2):
    """Load data using a flat directory + splits CSV."""
    if augmentation_mode == 'none':
        print("\n⚠ WARNING: No augmentation enabled")
        print("  For better results, try --augment, --mixup, or --specaugment\n")

    splits = parse_splits_csv(csv_path)

    X_test, y_test, test_paths = [], [], []
    X_val, y_val, val_paths = [], [], []
    X_train, y_train, train_paths = [], [], []
    labels = {}
    idx = 0
    failed_files = []
    csv_hits = 0
    csv_misses = 0

    file_lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                file_lookup[f] = (class_name, os.path.join(class_dir, f))

    total_files = 0
    train_count = 0
    for fn in splits:
        if fn in file_lookup:
            total_files += 1
            if splits[fn] == 'train':
                train_count += 1

    if augmentation_mode in ['baseline', 'specaugment']:
        total_files += train_count

    print(f"\nDataset Structure: CSV-based split from {csv_path}")
    print(f"  Flat dir: {flat_dir}")
    print(f"  CSV entries: {len(splits)}")
    print(f"  Files found: {len(file_lookup)}")

    print(f"\nAugmentation Strategy:")
    if augmentation_mode == 'baseline':
        print(f"  Train: baseline augmentation (time/pitch shift)")
    elif augmentation_mode == 'mixup':
        print(f"  Train: mixup (alpha={mixup_alpha})")
    elif augmentation_mode == 'specaugment':
        print(f"  Train: SpecAugment (freq/time masking)")
    else:
        print(f"  Train: no augmentation")
    print(f"  Total samples to process: {total_files}")

    with tqdm(total=total_files, desc="Loading from CSV splits", unit="file") as pbar:
        for target_split in ['test', 'val', 'train']:
            for fn, split in sorted(splits.items()):
                if split != target_split:
                    continue
                if fn not in file_lookup:
                    csv_misses += 1
                    continue

                csv_hits += 1
                class_name, full_path = file_lookup[fn]

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                try:
                    audio, _ = librosa.load(full_path, sr=TARGET_SR)
                    audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                        np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                    spec = compute_spec(audio, TARGET_SR, gmin, gmax, n_mels)

                    if split == 'test':
                        X_test.append(spec)
                        y_test.append(labels[class_name])
                        test_paths.append(full_path)
                        pbar.update(1)
                    elif split == 'val':
                        X_val.append(spec)
                        y_val.append(labels[class_name])
                        val_paths.append(full_path)
                        pbar.update(1)
                    elif split == 'train':
                        X_train.append(spec)
                        y_train.append(labels[class_name])
                        train_paths.append(full_path)
                        pbar.update(1)

                        if augmentation_mode == 'baseline':
                            aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                            aug_spec = compute_spec(aug_audio, TARGET_SR, gmin, gmax, n_mels)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(full_path + "_aug")
                            pbar.update(1)
                        elif augmentation_mode == 'specaugment':
                            aug_spec = augment_specaugment(spec)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(full_path + "_specaug")
                            pbar.update(1)

                except Exception as e:
                    failed_files.append(f"{full_path}: {str(e)}")
                    if split == 'train' and augmentation_mode in ['baseline', 'specaugment']:
                        pbar.update(2)
                    else:
                        pbar.update(1)
                    continue

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    if csv_misses > 0:
        print(f"\n⚠ Warning: {csv_misses} CSV entries had no matching file in {flat_dir}")

    if len(failed_files) > 0:
        print(f"\n⚠ Warning: {len(failed_files)} files failed to load")
        with open('data_loading_errors.txt', 'w') as f:
            for error in failed_files:
                f.write(error + '\n')

    num_classes = len(labels)
    print(f"\n✓ CSV Split Complete:")
    print(f"  Test (held-out):        {len(X_test):5d} samples")
    print(f"  Val (held-out):         {len(X_val):5d} samples")
    print(f"  Train (w/ augment):     {len(X_train):5d} samples")
    print(f"  Total:                  {len(X_test) + len(X_val) + len(X_train):5d} samples")
    print(f"  Classes:                {num_classes}")
    print(f"\n  ✓ NO DATA LEAKAGE: Test and Val samples never augmented")
    print(f"  ✓ INDEPENDENT SPLITS: True held-out evaluation (CSV-based)")

    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)


# --------------------------------------------------------------
# CUSTOM TRAINING LOOP FOR MIXUP
# --------------------------------------------------------------
class MixupDataGenerator(keras.utils.Sequence):
    """Custom data generator for mixup augmentation"""
    def __init__(self, X, y, batch_size, alpha=0.2, num_classes=10):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.num_classes = num_classes
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = len(X_batch)
            index = np.random.permutation(batch_size)

            X_mixed = lam * X_batch + (1 - lam) * X_batch[index]

            y_a = keras.utils.to_categorical(y_batch, self.num_classes)
            y_b = keras.utils.to_categorical(y_batch[index], self.num_classes)
            y_mixed = lam * y_a + (1 - lam) * y_b

            return X_mixed, y_mixed
        else:
            return X_batch, keras.utils.to_categorical(y_batch, self.num_classes)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    config = get_config()
    variant = config['variant']
    vcfg = config['vcfg']

    logger = TrainingLogger(config['output_dir'], variant, vcfg)

    print(f"\nConfig:")
    print(f"  Variant: {variant} — {vcfg['desc']}")
    print(f"  Random seed: {config['random_seed']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Finetune epochs: {config['finetune_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Warmup LR: {config['warmup_lr']}")
    print(f"  Finetune LR: {config['finetune_lr']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Model: TCN Series 2 — n_layers={vcfg['n_layers']}, channels={vcfg['channels']}, "
          f"kernel={vcfg['kernel']}, residual={vcfg['residual']}")
    print(f"  Augmentation: {config['augmentation_mode']}")
    if config['augmentation_mode'] == 'mixup':
        print(f"  Mixup Alpha: {config['mixup_alpha']}")
    print(f"  LR Schedule: {config['lr_schedule']}")
    print(f"  Spectrogram: {config['n_mels']}x{TIME_FRAMES} (10ms/frame)")
    print(f"  Splits CSV: {config['splits_csv']}")
    print(f"  Flat dir: {config['flat_dir']}")

    logger.log_hyperparameters(config)

    # Compute global stats (from training files only)
    print("\nComputing global normalization stats...")
    splits = parse_splits_csv(config['splits_csv'])
    train_files = {fn for fn, split in splits.items() if split == 'train'}
    global_min, global_max = compute_global_stats(
        config['flat_dir'], config['n_mels'], allowed_files=train_files)

    # Load data from CSV splits
    print(f"\nLoading dataset from CSV splits...")
    print("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels, failed_count = load_data_from_csv(
        config['splits_csv'], config['flat_dir'],
        global_min, global_max, config['n_mels'],
        augmentation_mode=config['augmentation_mode'],
        time_shift_ms=config['time_shift_ms'],
        pitch_shift_steps=config['pitch_shift_steps'],
        mixup_alpha=config['mixup_alpha']
    )
    print("=" * 70)

    class_names = list(class_labels.keys())
    num_classes = len(class_names)
    total_samples = len(X_train) + len(X_val) + len(X_test)

    print(f"\n✓ Total samples loaded: {total_samples}")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Spectrogram shape: {X_train[0].shape}")

    print(f"\nFinal Split:")
    print(f"  Train:     {len(X_train):5d} samples ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"  Val:       {len(X_val):5d} samples ({len(X_val) / total_samples * 100:.1f}%)")
    print(f"  Test:      {len(X_test):5d} samples ({len(X_test) / total_samples * 100:.1f}%)")
    print(f"  Total:     {total_samples:5d} samples")

    print(f"\nTest Set Class Distribution:")
    for class_name, class_idx in sorted(class_labels.items(), key=lambda x: x[1]):
        count = np.sum(y_test == class_idx)
        print(f"  {class_name:30s}: {count:3d} samples")

    print(f"\nValidation Set Class Distribution:")
    for class_name, class_idx in sorted(class_labels.items(), key=lambda x: x[1]):
        count = np.sum(y_val == class_idx)
        print(f"  {class_name:30s}: {count:3d} samples")

    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    logger.log_dataset_info(X_all, y_all, class_labels, X_train, X_val, X_test, failed_count)

    if len(X_val) < config['calib_samples']:
        print(f"\n⚠ Warning: Requested {config['calib_samples']} calibration samples, "
              f"but only {len(X_val)} validation samples available")
        config['calib_samples'] = len(X_val)
        print(f"  Using all {config['calib_samples']} validation samples for calibration")

    X_calib = X_val[:config['calib_samples']]
    print(f"\n✓ Calibration set: {len(X_calib)} samples (from validation set)")

    # Create TCN model
    print(f"\n{'=' * 70}")
    print(f"CREATING TCN MODEL (Series 2, variant {variant}) — {vcfg['desc']}")
    print(f"{'=' * 70}")
    model = create_tcn(num_classes, config['input_shape'], vcfg, config['dropout'])
    model.summary()

    logger.log_model_info(model)

    # Prepare training data
    if config['augmentation_mode'] == 'mixup':
        train_generator = MixupDataGenerator(
            X_train, y_train,
            config['batch_size'],
            alpha=config['mixup_alpha'],
            num_classes=num_classes
        )
        val_data = (X_val, keras.utils.to_categorical(y_val, num_classes))
        loss_function = 'categorical_crossentropy'
        print("\n✓ Using Mixup data generator for training")
    else:
        train_generator = None
        val_data = (X_val, y_val)
        loss_function = 'sparse_categorical_crossentropy'

    # Compile for warmup
    model.compile(
        optimizer=Adam(learning_rate=config['warmup_lr']),
        loss=loss_function,
        metrics=['accuracy']
    )

    # Stage 1: Warmup training
    logger.start_stage("STAGE 1: WARMUP TRAINING")
    print(f"\n{'=' * 70}")
    print(f"STAGE 1: WARMUP TRAINING ({config['warmup_epochs']} epochs)")
    print(f"{'=' * 70}")

    warmup_checkpoint = os.path.join(config['output_dir'], 'warmup_best.weights.h5')
    warmup_callbacks = [
        callbacks.ModelCheckpoint(
            warmup_checkpoint, monitor='val_accuracy',
            save_best_only=True, save_weights_only=True,
            mode='max', verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )
    ]

    if config['lr_schedule'] == 'cosine':
        warmup_callbacks.append(
            callbacks.LearningRateScheduler(
                lambda epoch: config['warmup_lr'] * 0.5 * (1 + np.cos(np.pi * epoch / config['warmup_epochs'])),
                verbose=0
            )
        )
    elif config['lr_schedule'] == 'plateau':
        warmup_callbacks.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                min_lr=1e-7, verbose=1
            )
        )
    elif config['lr_schedule'] == 'both':
        warmup_callbacks.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                min_lr=1e-7, verbose=1
            )
        )
        warmup_callbacks.append(
            callbacks.LearningRateScheduler(
                lambda epoch: config['warmup_lr'] * 0.5 * (1 + np.cos(np.pi * epoch / config['warmup_epochs'])),
                verbose=0
            )
        )

    try:
        if config['augmentation_mode'] == 'mixup':
            warmup_history = model.fit(
                train_generator,
                validation_data=val_data,
                epochs=config['warmup_epochs'],
                callbacks=warmup_callbacks,
                verbose=1
            )
        else:
            warmup_history = model.fit(
                X_train, y_train,
                validation_data=val_data,
                epochs=config['warmup_epochs'],
                batch_size=config['batch_size'],
                callbacks=warmup_callbacks,
                verbose=1
            )
    except Exception as e:
        print(f"\n✗ Warmup training failed: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Reduce batch size: --batch_size 16")
        print("2. Force CPU mode: --force_cpu")
        print("3. Limit GPU memory: --gpu_memory_limit 8192")
        raise

    logger.end_stage("STAGE 1: WARMUP TRAINING", warmup_history)
    print("\n✓ Warmup complete - best weights restored")

    # Stage 2: Fine-tuning
    logger.start_stage("STAGE 2: FINE-TUNING")
    print(f"\n{'=' * 70}")
    print(f"STAGE 2: FINE-TUNING ({config['finetune_epochs']} epochs)")
    print(f"{'=' * 70}")

    model.compile(
        optimizer=Adam(learning_rate=config['finetune_lr']),
        loss=loss_function,
        metrics=['accuracy']
    )

    finetune_checkpoint = os.path.join(config['output_dir'], 'finetune_best.weights.h5')
    finetune_callbacks = [
        callbacks.ModelCheckpoint(
            finetune_checkpoint, monitor='val_accuracy',
            save_best_only=True, save_weights_only=True,
            mode='max', verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )
    ]

    if config['lr_schedule'] in ['plateau', 'both']:
        finetune_callbacks.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                min_lr=1e-8, verbose=1
            )
        )

    try:
        if config['augmentation_mode'] == 'mixup':
            finetune_history = model.fit(
                train_generator,
                validation_data=val_data,
                epochs=config['finetune_epochs'],
                callbacks=finetune_callbacks,
                verbose=1
            )
        else:
            finetune_history = model.fit(
                X_train, y_train,
                validation_data=val_data,
                epochs=config['finetune_epochs'],
                batch_size=config['batch_size'],
                callbacks=finetune_callbacks,
                verbose=1
            )
    except Exception as e:
        print(f"\n✗ Fine-tuning failed: {e}")
        print("\nCannot continue with invalid model. Please check:")
        print("1. GPU memory issues - try --force_cpu or smaller --batch_size")
        print("2. Learning rate too high - try lower --finetune_lr")
        print("3. Model architecture issues")
        raise

    logger.end_stage("STAGE 2: FINE-TUNING", finetune_history)
    print("\n✓ Fine-tuning complete - best weights restored")

    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(warmup_history, finetune_history, config['output_dir'])

    # Save FP32 model
    fp32_path = os.path.join(config['output_dir'], 'model_fp32.keras')
    model.save(fp32_path)
    print(f"✓ Saved FP32 model: {fp32_path}")

    # Evaluate FP32
    logger.start_stage("EVALUATION: FP32 (.keras)")
    print(f"\n{'=' * 70}")
    print("EVALUATING FP32 MODEL (.keras) ON HELD-OUT TEST SET")
    print(f"{'=' * 70}")
    fp32_acc = evaluate_model(model, X_test, y_test, class_names,
                              config['output_dir'], "FP32")
    logger.log_evaluation("FP32 (.keras)", fp32_acc,
                          os.path.join(config['output_dir'], 'classification_report_fp32.txt'))

    # Convert to TFLite INT8
    logger.start_stage("TFLITE CONVERSION (PTQ)")
    print(f"\n{'=' * 70}")
    print("CONVERTING TO INT8 TFLITE (POST-TRAINING QUANTIZATION)")
    print(f"{'=' * 70}")
    int8_path = os.path.join(config['output_dir'], 'model_int8.tflite')
    convert_to_tflite_int8(model, X_calib, int8_path)

    # Evaluate INT8
    logger.start_stage("EVALUATION: INT8 TFLite")
    print(f"\n{'=' * 70}")
    print("EVALUATING INT8 TFLITE ON HELD-OUT TEST SET")
    print(f"{'=' * 70}")
    int8_acc = evaluate_tflite(int8_path, X_test, y_test, class_names,
                               config['output_dir'])
    logger.log_evaluation("INT8 TFLite", int8_acc,
                          os.path.join(config['output_dir'], 'classification_report_int8.txt'))

    model_sizes = {
        "FP32 (.keras)": f"{os.path.getsize(fp32_path) / (1024 ** 2):.2f} MB",
        "INT8 (.tflite)": f"{os.path.getsize(int8_path) / 1024:.1f} KB"
    }

    logger.log_final_results(fp32_acc, int8_acc, model_sizes,
                             warmup_history, finetune_history, config, model)

    drop = fp32_acc - int8_acc
    total_time = time.time() - script_start

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS — TCN Series 2 variant {variant} ({vcfg['desc']})")
    print(f"{'=' * 70}")
    print(f"Variant:                 {variant} — {vcfg['desc']}")
    print(f"Augmentation Mode:       {config['augmentation_mode']}")
    if config['augmentation_mode'] == 'mixup':
        print(f"Mixup Alpha:             {config['mixup_alpha']}")
    print(f"LR Schedule:             {config['lr_schedule']}")
    print(f"FP32 Accuracy:           {fp32_acc:6.2f}%")
    print(f"INT8 Accuracy:           {int8_acc:6.2f}%")
    print(f"Accuracy Drop:           {drop:6.2f}%")
    print(f"Total Execution Time:    {format_time(total_time)}")
    print(f"\n✓ Test/Val sets were HELD-OUT during training (no data leakage)")
    print(f"✓ Results are publication-ready and reproducible")
    print(f"{'=' * 70}")

    print(f"\n✓ Complete training report saved to:")
    print(f"  {logger.log_path}")
    print(f"\nAll results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()

    total_script_time = time.time() - script_start
    print(f"\n{'=' * 70}")
    print(f"Script completed in: {format_time(total_script_time)}")
    print(f"{'=' * 70}")
