#!/usr/bin/env python3
"""
Model 2i: SqueezeNet v1.1 for Mygardenbird Classification
Ultra-compact architecture for STM32H747 deployment

SqueezeNet Key Features:
- Fire modules: Squeeze (1x1) + Expand (1x1 + 3x3)
- Very small: ~1.2M params → ~300 KB INT8
- Designed for embedded deployment
- Simple architecture, no complex attention

Input: 64×300 mel-spectrogram (native, no resize)
Expected: 85-89% accuracy @ ~300 KB (very small!)
Target: Smallest viable model for Portenta H7

Version 1.1 improvements over 1.0:
- Bypass connections for better gradient flow
- Reduced parameter count (~50% fewer than v1.0)
- Same training: Warmup + finetune, early stopping
- Same augmentation: Baseline, SpecAugment, or Mixup
- Same quantization: PTQ INT8 TFLite

Expected outcome: Validate if 224x224 resize is bottleneck or if MobileNet
architecture itself is suboptimal for spectrograms.
"""

print("\n\n\n")
for _ in range(3):
    print(" ❤️ " * 30)

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
# CONSTANTS (Same as 7d)
# --------------------------------------------------------------
DEFAULT_RANDOM_STATE = 42

TARGET_SR = 16000
AUDIO_LENGTH_SEC = 3
FIXED_AUDIO_LENGTH = TARGET_SR * AUDIO_LENGTH_SEC
HOP_LENGTH = 160  # 10ms at 16kHz = 160 samples
N_FFT = 512
N_MELS = 64
FMAX = 8000
TIME_FRAMES = 300  # Fixed: 3 seconds / 10ms = 300 frames

# Default paths (flat dir + CSV splits, matching 3a)
DEFAULT_FLAT_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
DEFAULT_SPLITS_CSV = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"

# SpecAugment settings
SPECAUGMENT_FREQ_MASK = 8
SPECAUGMENT_TIME_MASK = 20
SPECAUGMENT_NUM_MASKS = 2



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
    print("Using LEGACY Adam (fast on M1/M2/M4)")
elif system == "Linux":
    try:
        # Prefer AdamW if available (TF >= 2.14)
        from tf_keras.optimizers import AdamW
        Adam = AdamW
        print("Using AdamW optimizer (Linux)")
    except ImportError:
        # Fallback to standard Adam
        from tf_keras.optimizers import Adam
        print("Using standard Adam (AdamW not available)")
else:
    from tf_keras.optimizers import Adam
    print(f"Using standard Adam ({system})")


# --------------------------------------------------------------
# CACHE MANAGEMENT (from 7d/4d)
# --------------------------------------------------------------
def compute_cache_hash(config_params):
    """Compute hash of preprocessing parameters for cache validation."""
    cache_key = {
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_mels': N_MELS,
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
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--calib_samples", type=int, default=200)

    # Augmentation flags
    parser.add_argument("--augment", action='store_true',
                        help="Enable baseline augmentation (time/pitch shift)")
    parser.add_argument("--mixup", type=float, default=None,
                        help="Enable mixup augmentation with alpha value (e.g., 0.2)")
    parser.add_argument("--specaugment", action='store_true',
                        help="Enable SpecAugment (frequency/time masking)")

    # Baseline augmentation parameters
    parser.add_argument("--time_shift_ms", type=int, default=100,
                        help="Max time shift in milliseconds (baseline augmentation)")
    parser.add_argument("--pitch_shift_steps", type=int, default=2,
                        help="Max pitch shift in semitones (baseline augmentation)")

    # GPU parameters
    parser.add_argument("--force_cpu", action='store_true',
                        help="Force CPU execution (disable GPU)")
    parser.add_argument("--gpu_memory_limit", type=int, default=None,
                        help="GPU memory limit in MB (e.g., 8192 for 8GB)")

    # Configurable paths (flat dir + CSV splits, matching 3a)
    parser.add_argument("--splits_csv", type=str, default=DEFAULT_SPLITS_CSV,
                        help="Path to splits CSV")
    parser.add_argument("--flat_dir", type=str, default=DEFAULT_FLAT_DIR,
                        help="Path to flat dataset directory (class_name/file.wav)")

    # LR schedule
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "plateau", "both", "none"],
                        help="Learning rate schedule strategy")

    # Random seed
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_STATE,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Determine augmentation mode and folder name suffix
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

    output_dir_name = (
        f"results_mygardenbird_4_{platform.system().lower()}/"
        f"4a_squeezenet_v11_"
        f"drop{int(args.dropout * 100):02d}_"
        f"rand{args.random_seed}_"
        f"warm{args.warmup_epochs}_"
        f"{aug_suffix}"
    )

    # Clean up double underscores if aug_suffix is empty
    output_dir_name = output_dir_name.replace("__", "_").rstrip("_")

    n_classes = len([d for d in os.listdir(args.flat_dir)
                     if os.path.isdir(os.path.join(args.flat_dir, d)) and not d.startswith('.')])

    config = {
        'warmup_epochs': args.warmup_epochs,
        'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size,
        'warmup_lr': args.warmup_lr,
        'finetune_lr': args.finetune_lr,
        'dropout': args.dropout,
        'time_frames': TIME_FRAMES,
        'input_shape': (N_MELS, TIME_FRAMES, 1),
        'output_dir': output_dir_name,
        'calib_samples': args.calib_samples,
        'augmentation_mode': augmentation_mode,
        'mixup_alpha': args.mixup,
        'time_shift_ms': args.time_shift_ms,
        'pitch_shift_steps': args.pitch_shift_steps,
        'force_cpu': args.force_cpu,
        'gpu_memory_limit': args.gpu_memory_limit,
        'splits_csv': args.splits_csv,
        'flat_dir': args.flat_dir,
        'n_classes': n_classes,
        'lr_schedule': args.lr_schedule,
        'random_seed': args.random_seed,
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    return config


# --------------------------------------------------------------
# LOGGING UTILITIES (reused from 7d/1a)
# --------------------------------------------------------------
class TrainingLogger:
    """Centralized logger for all training metrics and hyperparameters."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, 'training_report.txt')
        self.start_time = time.time()
        self.stage_times = {}

        # Initialize log file
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION 2C: MOBILENETV3SMALL @ 64×300 NATIVE (Version 2c)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Platform: {platform.system()} {platform.machine()}\n")
            f.write(f"Python: {sys.version.split()[0]}\n")
            f.write(f"TensorFlow: {tf.__version__}\n")
            f.write(f"Keras: {keras.__version__}\n")
            f.write("\n")

    def log_section(self, title):
        """Log a section header."""
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n")

    def log_hyperparameters(self, config):
        """Log all hyperparameters."""
        self.log_section("HYPERPARAMETERS")
        with open(self.log_path, 'a') as f:
            # System info
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
            f.write(f"  Mel Bins (N_MELS):      {N_MELS}\n")
            f.write(f"  Max Frequency (FMAX):   {FMAX} Hz\n")
            f.write(f"  Time Frames:            {TIME_FRAMES} (FIXED)\n")
            f.write(f"  Spectrogram Shape:      {N_MELS}x{TIME_FRAMES}\n")
            f.write(f"  Center Padding:         Enabled (librosa center=True)\n")

            f.write("\nModel Architecture:\n")
            f.write(f"  Model Type:             SqueezeNet v1.1 (Model 5a)\n")
            f.write(f"  Input Shape:            {config['input_shape']} (NATIVE, NO RESIZE)\n")
            f.write(f"  Hypothesis:             Standard 224×224 resize is suboptimal\n")
            f.write(f"  Adaptation:             First conv stride=1 (preserve resolution)\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  Blocks:                 Fire modules (squeeze + expand)\n")
            f.write(f"  Activation:             ReLU (SqueezeNet standard)\n")
            f.write(f"  Global Pooling:         GlobalAveragePooling2D\n")
            f.write(f"  Quantization Target:    INT8 optimized architecture\n")

            f.write("\nTraining Configuration:\n")
            f.write(f"  Random Seed:            {config['random_seed']}\n")
            f.write(f"  Warmup Epochs:          {config['warmup_epochs']}\n")
            f.write(f"  Fine-tune Epochs:       {config['finetune_epochs']}\n")
            f.write(f"  Total Epochs:           {config['warmup_epochs'] + config['finetune_epochs']}\n")
            f.write(f"  Batch Size:             {config['batch_size']}\n")
            f.write(f"  Warmup Learning Rate:   {config['warmup_lr']}\n")
            f.write(f"  Fine-tune Learning Rate:{config['finetune_lr']}\n")
            f.write(f"  LR Schedule:            {config['lr_schedule']}\n")
            f.write(f"  Optimizer:              Adam (Legacy on Apple Silicon)\n")
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

            f.write("\nDeployment Target:\n")
            f.write(f"  Platform:               ARM Cortex-M7\n")
            f.write(f"  Memory Target:          <512 KB (50% of 1MB)\n")

            f.write("\nData Paths:\n")
            f.write(f"  Flat Dir:               {config['flat_dir']}\n")
            f.write(f"  Splits CSV:             {config['splits_csv']}\n")
            f.write(f"  Output Directory:       {config['output_dir']}\n")

    def log_dataset_info(self, X, y, class_labels, X_train, X_val, X_test, failed_files=0):
        """Log dataset statistics."""
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
        """Log model architecture summary."""
        self.log_section("MODEL ARCHITECTURE")
        with open(self.log_path, 'a') as f:
            import io
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_str = stream.getvalue()
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
                f.write(f"\n  ⚠ WARNING: Model may exceed 512 KB target for Cortex-M7\n")
            else:
                f.write(f"\n  ✓ Model size within 512 KB target\n")

    def start_stage(self, stage_name):
        """Mark the start of a training stage."""
        self.stage_times[stage_name] = {'start': time.time()}
        self.log_section(stage_name)

    def end_stage(self, stage_name, history=None):
        """Mark the end of a training stage and log results."""
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
        """Log model evaluation results."""
        with open(self.log_path, 'a') as f:
            f.write(f"\n{model_name} Evaluation:\n")
            f.write(f"  Test Accuracy:          {accuracy:.2f}%\n")
            f.write(f"  Classification Report:  {report_path}\n")

    def log_final_results(self, fp32_acc, int8_acc, model_sizes,
                          warmup_history, finetune_history, config):
        """Log final comparison results."""
        self.log_section("FINAL RESULTS SUMMARY")

        drop = fp32_acc - int8_acc
        total_time = time.time() - script_start

        with open(self.log_path, 'a') as f:
            # Quick reference card
            f.write("\n" + "=" * 80 + "\n")
            f.write("QUICK REFERENCE (Copy to spreadsheet)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: 5a_squeezenet_v11 | Dropout: {config['dropout']} | "
                    f"Aug: {config['augmentation_mode']} | Seed: {config['random_seed']}\n")
            f.write(f"FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}% | "
                    f"Drop: {drop:+.2f}% | Time: {format_time(total_time)}\n")

            # Detailed results
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

            # Training metrics
            f.write(f"\nTraining Metrics:\n")
            f.write(f"  Best Warmup Val Acc:    {max(warmup_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Best Finetune Val Acc:  {max(finetune_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Final Train Acc:        {finetune_history.history['accuracy'][-1] * 100:6.2f}%\n")
            f.write(f"  Final Val Acc:          {finetune_history.history['val_accuracy'][-1] * 100:6.2f}%\n")
            f.write(f"  Train-Test Gap:         {finetune_history.history['accuracy'][-1] * 100 - int8_acc:+6.2f}%\n")

            overfitting_gap = finetune_history.history['accuracy'][-1] * 100 - finetune_history.history['val_accuracy'][-1] * 100
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

            # CSV format
            f.write("\n" + "=" * 80 + "\n")
            f.write("CSV FORMAT (for batch comparison)\n")
            f.write("=" * 80 + "\n")
            f.write("model,dropout,augmentation,warmup_epochs,finetune_epochs,warmup_lr,finetune_lr,"
                    "lr_schedule,fp32_acc,int8_acc,drop,best_val_acc,train_val_gap,train_time_sec,model_size_kb\n")
            f.write(f"5a_squeezenet_v11,{config['dropout']},{config['augmentation_mode']},"
                    f"{config['warmup_epochs']},{config['finetune_epochs']},{config['warmup_lr']},{config['finetune_lr']},"
                    f"{config['lr_schedule']},{fp32_acc:.2f},{int8_acc:.2f},{drop:.2f},"
                    f"{max(finetune_history.history['val_accuracy']) * 100:.2f},"
                    f"{overfitting_gap:.2f},{int(total_time)},"
                    f"{os.path.getsize(os.path.join(config['output_dir'], 'squeezenet_int8.tflite')) / 1024:.1f}\n")

            # Hypothesis validation
            f.write("\n" + "=" * 80 + "\n")
            f.write("HYPOTHESIS VALIDATION: 224×224 RESIZE IS SUBOPTIMAL\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nNative Input: 64×300 (no resize)\n")
            f.write(f"INT8 Accuracy: {int8_acc:.2f}%\n")
            f.write(f"\nComparison needed with 224×224 resized version to validate hypothesis.\n")
            f.write(f"Expected outcome: Native 64×300 should outperform resized 224×224\n")
            f.write(f"due to preservation of temporal/frequency resolution.\n")


# Data loading: CSV-based splits from flat directory (matching 3a)
def compute_spec(audio, sr, gmin=None, gmax=None):
    """Compute mel spectrogram with per-sample percentile normalisation → [0, 1].

    gmin/gmax are accepted for API compatibility but ignored;
    per-sample normalisation is used instead (matching 3a).
    """
    WIN_LENGTH = 400

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT,
        win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX, center=True,
        power=2.0, window='hann'
    )

    # Ensure exact TIME_FRAMES
    if mel.shape[1] > TIME_FRAMES:
        mel = mel[:, :TIME_FRAMES]
    if mel.shape[1] < TIME_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, TIME_FRAMES - mel.shape[1])))

    mel_db = librosa.power_to_db(mel, top_db=None)

    # Per-sample percentile normalisation to [0, 1]
    p2, p98 = np.percentile(mel_db, (2, 98))
    if p98 > p2 + 1e-8:
        mel_norm = np.clip(mel_db, p2, p98)
        mel_norm = (mel_norm - p2) / (p98 - p2)
    else:
        mel_norm = np.full_like(mel_db, 0.5)

    return mel_norm[..., np.newaxis].astype(np.float32)


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


def load_data_from_csv(csv_path, flat_dir, gmin=None, gmax=None, augmentation_mode='none',
                       time_shift_ms=100, pitch_shift_steps=2, mixup_alpha=0.2):
    """Load data using a flat directory + splits CSV (matching 3a)."""
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
    csv_misses = 0

    file_lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                file_lookup[f] = (class_name, os.path.join(class_dir, f))

    total_files = sum(1 for fn in splits if fn in file_lookup)
    train_count = sum(1 for fn, sp in splits.items() if fn in file_lookup and sp == 'train')
    if augmentation_mode in ['baseline', 'specaugment']:
        total_files += train_count

    print(f"\nDataset Structure: CSV-based split from {csv_path}")
    print(f"  Flat dir: {flat_dir}")
    print(f"  CSV entries: {len(splits)} | Files found: {len(file_lookup)}")

    with tqdm(total=total_files, desc="Loading from CSV splits", unit="file") as pbar:
        for target_split in ['test', 'val', 'train']:
            for fn, split in sorted(splits.items()):
                if split != target_split:
                    continue
                if fn not in file_lookup:
                    csv_misses += 1
                    continue

                class_name, full_path = file_lookup[fn]
                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                try:
                    audio, _ = librosa.load(full_path, sr=TARGET_SR)
                    audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                        np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))
                    spec = compute_spec(audio, TARGET_SR)

                    if split == 'test':
                        X_test.append(spec); y_test.append(labels[class_name])
                        test_paths.append(full_path); pbar.update(1)
                    elif split == 'val':
                        X_val.append(spec); y_val.append(labels[class_name])
                        val_paths.append(full_path); pbar.update(1)
                    elif split == 'train':
                        X_train.append(spec); y_train.append(labels[class_name])
                        train_paths.append(full_path); pbar.update(1)
                        if augmentation_mode == 'baseline':
                            aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                            X_train.append(compute_spec(aug_audio, TARGET_SR))
                            y_train.append(labels[class_name])
                            train_paths.append(full_path + "_aug"); pbar.update(1)
                        elif augmentation_mode == 'specaugment':
                            X_train.append(augment_specaugment(spec))
                            y_train.append(labels[class_name])
                            train_paths.append(full_path + "_specaug"); pbar.update(1)
                except Exception as e:
                    failed_files.append(f"{full_path}: {str(e)}")
                    pbar.update(2 if split == 'train' and augmentation_mode in ['baseline', 'specaugment'] else 1)
                    continue

    X_test = np.array(X_test, dtype=np.float32);   y_test = np.array(y_test, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32);     y_val = np.array(y_val, dtype=np.int32)
    X_train = np.array(X_train, dtype=np.float32); y_train = np.array(y_train, dtype=np.int32)

    if csv_misses > 0:
        print(f"\n⚠ Warning: {csv_misses} CSV entries had no matching file in {flat_dir}")
    if failed_files:
        print(f"\n⚠ Warning: {len(failed_files)} files failed to load")
        with open('data_loading_errors.txt', 'w') as f:
            f.write('\n'.join(failed_files))

    print(f"\n✓ CSV Split: test={len(X_test)} val={len(X_val)} train={len(X_train)} classes={len(labels)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)


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

    # Apply frequency masks
    for _ in range(SPECAUGMENT_NUM_MASKS):
        f_mask = np.random.randint(0, SPECAUGMENT_FREQ_MASK)
        f0 = np.random.randint(0, freq_bins - f_mask) if f_mask < freq_bins else 0
        spec_aug[f0:f0 + f_mask, :, 0] = 0

    # Apply time masks
    for _ in range(SPECAUGMENT_NUM_MASKS):
        t_mask = np.random.randint(0, SPECAUGMENT_TIME_MASK)
        t0 = np.random.randint(0, time_bins - t_mask) if t_mask < time_bins else 0
        spec_aug[:, t0:t0 + t_mask, 0] = 0

    return spec_aug


# --------------------------------------------------------------
# MOBILENETV3SMALL ADAPTED FOR 64×300 INPUT
# --------------------------------------------------------------
def hard_swish(x):
    """Hard-swish activation (MobileNetV3)"""
    return x * tf.nn.relu6(x + 3) / 6


def se_block(x, ratio=4):
    """Squeeze-and-Excitation block"""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Conv2D(filters // ratio, 1, activation='relu')(se)
    se = layers.Conv2D(filters, 1, activation='hard_sigmoid')(se)
    return layers.Multiply()([x, se])


def inverted_residual_block(x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id):
    """Inverted residual block (MobileNetV2/V3)"""
    shortcut = x
    prefix = f'block{block_id}_'

    # Expansion
    if expansion != 1:
        x = layers.Conv2D(expansion * x.shape[-1], 1, padding='same',
                         use_bias=False, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_act')(x)

    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same',
                              use_bias=False, name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(name=prefix + 'depthwise_bn')(x)
    x = layers.Activation(activation, name=prefix + 'depthwise_act')(x)

    # SE block
    if se_ratio is not None:
        x = se_block(x, ratio=se_ratio)

    # Project
    x = layers.Conv2D(filters, 1, padding='same',
                     use_bias=False, name=prefix + 'project')(x)
    x = layers.BatchNormalization(name=prefix + 'project_bn')(x)

    # Skip connection
    if stride == 1 and shortcut.shape[-1] == filters:
        x = layers.Add(name=prefix + 'add')([shortcut, x])

    return x


    """
    SqueezeNet Fire module.

    Architecture:
    - Squeeze layer: 1x1 conv to reduce channels
    - Expand layer: parallel 1x1 and 3x3 convs, concatenated
    - Optional bypass connection (for v1.1)

    Args:
        x: Input tensor
        squeeze_filters: Number of filters in squeeze layer
        expand_filters: Number of filters in each expand layer (1x1 and 3x3)
        name: Module name prefix
        use_bypass: Whether to add bypass connection (v1.1 feature)
    """

def fire_module(x, squeeze_filters, expand_filters, name, use_bypass=False):
    input_tensor = x

    squeeze = layers.Conv2D(squeeze_filters, 1, padding='same',
                            use_bias=False, name=f'{name}_squeeze')(x)
    squeeze = layers.BatchNormalization(name=f'{name}_squeeze_bn')(squeeze)
    squeeze = layers.Activation('relu', name=f'{name}_squeeze_act')(squeeze)

    expand_1x1 = layers.Conv2D(expand_filters, 1, padding='same',
                               use_bias=False, name=f'{name}_expand1x1')(squeeze)
    expand_1x1 = layers.BatchNormalization(name=f'{name}_expand1x1_bn')(expand_1x1)
    expand_1x1 = layers.Activation('relu', name=f'{name}_expand1x1_act')(expand_1x1)

    expand_3x3 = layers.Conv2D(expand_filters, 3, padding='same',
                               use_bias=False, name=f'{name}_expand3x3')(squeeze)
    expand_3x3 = layers.BatchNormalization(name=f'{name}_expand3x3_bn')(expand_3x3)
    expand_3x3 = layers.Activation('relu', name=f'{name}_expand3x3_act')(expand_3x3)

    x = layers.Concatenate(name=f'{name}_concat')([expand_1x1, expand_3x3])

    if use_bypass and input_tensor.shape[-1] == x.shape[-1]:
        x = layers.Add(name=f'{name}_bypass')([x, input_tensor])

    return x


def create_squeezenet_v11_64x300(num_classes, input_shape, dropout=0.2):
    """
    SqueezeNet v1.1 adapted for 64×300 spectrogram input (NATIVE, NO RESIZE).

    Key adaptations:
    1. First conv stride=1 instead of 2 (preserve 64×300 resolution)
    2. Adjusted pooling for non-square aspect ratio
    3. Bypass connections in Fire modules (v1.1 feature)

    Architecture based on SqueezeNet v1.1:
    "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"
    (Iandola et al., 2016)

    SqueezeNet v1.1 improvements:
    - Bypass connections for better gradient flow
    - ~50% fewer parameters than v1.0
    - More aggressive pooling schedule

    Expected: ~1.2M params → ~300 KB INT8
    """
    inputs = layers.Input(shape=input_shape)

    # Initial conv - CRITICAL: stride=1 to preserve 64×300 resolution
    # Standard SqueezeNet uses stride=2, but that's for 224×224
    x = layers.Conv2D(64, 3, strides=1, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_act')(x)

    # Max pooling - adjusted for 64×300 aspect ratio
    # Standard: 3×3 pool with stride 2 → 32×150
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool1')(x)

    # Fire modules 2-3 (bypass on fire3)
    x = fire_module(x, squeeze_filters=16, expand_filters=64,
                   name='fire2', use_bypass=False)
    x = fire_module(x, squeeze_filters=16, expand_filters=64,
                   name='fire3', use_bypass=True)  # Bypass connection

    # Max pooling → 16×75
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool3')(x)

    # Fire modules 4-5 (bypass on fire5)
    x = fire_module(x, squeeze_filters=32, expand_filters=128,
                   name='fire4', use_bypass=False)
    x = fire_module(x, squeeze_filters=32, expand_filters=128,
                   name='fire5', use_bypass=True)  # Bypass connection

    # Max pooling → 8×37
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool5')(x)

    # Fire modules 6-9 (bypass on fire7 and fire9)
    x = fire_module(x, squeeze_filters=48, expand_filters=192,
                   name='fire6', use_bypass=False)
    x = fire_module(x, squeeze_filters=48, expand_filters=192,
                   name='fire7', use_bypass=True)  # Bypass connection
    x = fire_module(x, squeeze_filters=64, expand_filters=256,
                   name='fire8', use_bypass=False)
    x = fire_module(x, squeeze_filters=64, expand_filters=256,
                   name='fire9', use_bypass=True)  # Bypass connection

    # Dropout before final conv
    x = layers.Dropout(dropout, name='dropout_fire')(x)

    # Final conv: 1x1 conv to num_classes
    x = layers.Conv2D(num_classes, 1, activation='relu',
                     padding='same', name='conv10')(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Softmax classifier
    outputs = layers.Activation('softmax', name='softmax')(x)

    return keras.Model(inputs, outputs, name="SqueezeNet_v1.1_64x300")


# --------------------------------------------------------------
# TFLITE CONVERSION & EVALUATION (from 1a)
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
    logger = TrainingLogger(config['output_dir'])

    print(f"\nModel 5a: SqueezeNet v1.1 @ 64×300 (NATIVE)")
    print(f"  Hypothesis: 224×224 resize is SUBOPTIMAL")
    print(f"  Random seed: {config['random_seed']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Augmentation: {config['augmentation_mode']}")

    logger.log_hyperparameters(config)

    # Load data (no global stats needed — per-sample normalisation)
    print("Loading dataset from CSV splits...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels, failed_count = load_data_from_csv(
        config['splits_csv'], config['flat_dir'],
        gmin=None, gmax=None,
        augmentation_mode=config['augmentation_mode'],
        time_shift_ms=config['time_shift_ms'],
        pitch_shift_steps=config['pitch_shift_steps'],
        mixup_alpha=config['mixup_alpha']
    )

    class_names = list(class_labels.keys())
    num_classes = len(class_names)

    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    logger.log_dataset_info(X_all, y_all, class_labels, X_train, X_val, X_test, failed_count)

    # Calibration set
    X_calib = X_val[:config['calib_samples']]
    print(f"\n✓ Calibration set: {len(X_calib)} samples")

    # Create model
    print(f"\n{'=' * 70}")
    print("CREATING SQUEEZENET V1.1 MODEL (64×300 NATIVE)")
    print(f"{'=' * 70}")
    model = create_squeezenet_v11_64x300(num_classes, config['input_shape'], config['dropout'])
    model.summary()

    logger.log_model_info(model)

    # Prepare training data
    if config['augmentation_mode'] == 'mixup':
        train_generator = MixupDataGenerator(
            X_train, y_train, config['batch_size'],
            alpha=config['mixup_alpha'], num_classes=num_classes
        )
        val_data = (X_val, keras.utils.to_categorical(y_val, num_classes))
        loss_function = 'categorical_crossentropy'
    else:
        train_generator = None
        val_data = (X_val, y_val)
        loss_function = 'sparse_categorical_crossentropy'

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=config['warmup_lr']),
        loss=loss_function,
        metrics=['accuracy']
    )

    # Warmup training
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

    if config['lr_schedule'] in ['cosine', 'both']:
        warmup_callbacks.append(
            callbacks.LearningRateScheduler(
                lambda epoch: config['warmup_lr'] * 0.5 * (1 + np.cos(np.pi * epoch / config['warmup_epochs'])),
                verbose=0
            )
        )
    if config['lr_schedule'] in ['plateau', 'both']:
        warmup_callbacks.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                min_lr=1e-7, verbose=1
            )
        )

    if config['augmentation_mode'] == 'mixup':
        warmup_history = model.fit(
            train_generator, validation_data=val_data,
            epochs=config['warmup_epochs'],
            callbacks=warmup_callbacks, verbose=1
        )
    else:
        warmup_history = model.fit(
            X_train, y_train, validation_data=val_data,
            epochs=config['warmup_epochs'],
            batch_size=config['batch_size'],
            callbacks=warmup_callbacks, verbose=1
        )

    logger.end_stage("STAGE 1: WARMUP TRAINING", warmup_history)

    # Fine-tuning
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

    if config['augmentation_mode'] == 'mixup':
        finetune_history = model.fit(
            train_generator, validation_data=val_data,
            epochs=config['finetune_epochs'],
            callbacks=finetune_callbacks, verbose=1
        )
    else:
        finetune_history = model.fit(
            X_train, y_train, validation_data=val_data,
            epochs=config['finetune_epochs'],
            batch_size=config['batch_size'],
            callbacks=finetune_callbacks, verbose=1
        )

    logger.end_stage("STAGE 2: FINE-TUNING", finetune_history)

    # Plot and save
    plot_training_history(warmup_history, finetune_history, config['output_dir'])

    fp32_path = os.path.join(config['output_dir'], 'squeezenet_fp32.keras')
    model.save(fp32_path)
    print(f"✓ Saved FP32 model: {fp32_path}")

    # Evaluate FP32
    logger.start_stage("EVALUATION: FP32 (.keras)")
    print(f"\n{'=' * 70}")
    print("EVALUATING FP32 MODEL")
    print(f"{'=' * 70}")
    fp32_acc = evaluate_model(model, X_test, y_test, class_names,
                              config['output_dir'], "FP32")
    logger.log_evaluation("FP32 (.keras)", fp32_acc,
                          os.path.join(config['output_dir'], 'classification_report_fp32.txt'))

    # Convert to TFLite INT8
    logger.start_stage("TFLITE CONVERSION (PTQ)")
    print(f"\n{'=' * 70}")
    print("CONVERTING TO INT8 TFLITE")
    print(f"{'=' * 70}")
    int8_path = os.path.join(config['output_dir'], 'squeezenet_int8.tflite')
    convert_to_tflite_int8(model, X_calib, int8_path)

    # Evaluate INT8
    logger.start_stage("EVALUATION: INT8 TFLite")
    print(f"\n{'=' * 70}")
    print("EVALUATING INT8 TFLITE")
    print(f"{'=' * 70}")
    int8_acc = evaluate_tflite(int8_path, X_test, y_test, class_names,
                               config['output_dir'])
    logger.log_evaluation("INT8 TFLite", int8_acc,
                          os.path.join(config['output_dir'], 'classification_report_int8.txt'))

    # Collect sizes
    model_sizes = {
        "FP32 (.keras)": f"{os.path.getsize(fp32_path) / (1024 ** 2):.2f} MB",
        "INT8 (.tflite)": f"{os.path.getsize(int8_path) / 1024:.1f} KB"
    }

    logger.log_final_results(fp32_acc, int8_acc, model_sizes,
                             warmup_history, finetune_history, config)

    # Summary
    drop = fp32_acc - int8_acc
    total_time = time.time() - script_start

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Model:                   5a_squeezenet_v11 (64×300 NATIVE)")
    print(f"Augmentation:            {config['augmentation_mode']}")
    print(f"FP32 Accuracy:           {fp32_acc:6.2f}%")
    print(f"INT8 Accuracy:           {int8_acc:6.2f}%")
    print(f"Accuracy Drop:           {drop:6.2f}%")
    print(f"Total Time:              {format_time(total_time)}")
    print(f"\nHypothesis Test: Compare with 224×224 resized version")
    print(f"Expected: Native 64×300 should preserve resolution better")
    print(f"{'=' * 70}")

    print(f"\n✓ Complete training report: {logger.log_path}")
    print(f"✓ All results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()

    total_script_time = time.time() - script_start
    print(f"\n{'=' * 70}")
    print(f"Script completed in: {format_time(total_script_time)}")
    print(f"{'=' * 70}")
