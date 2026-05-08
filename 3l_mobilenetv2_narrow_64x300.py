#!/usr/bin/env python3
"""
Model 8a - Narrow MobileNetV2 for Mygardenbird Classification
Fixed spectrogram shape: 64x300 (10ms per frame) - NATIVE INPUT, NO RESIZE
Data Split: Fixed 75:10:15 (train/val/test) from dataset directories

Architecture: MobileNetV2 with narrow width multiplier
- Inverted residual blocks (expand -> depthwise -> project)
- Linear bottleneck (no activation after projection)
- Skip connections when stride=1 and channels match
- ReLU6 activation (quantization-friendly)
- Width multiplier 0.35-0.5 to fit <512KB target

Key differences from MobileNetV1:
- Inverted residual structure (V1 has no expansion)
- Linear bottleneck (no activation on projection)
- Skip connections within blocks

Target: ARM Cortex-M7, <512KB INT8 model size
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
# CONSTANTS
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

# Default paths
DEFAULT_DATASET_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
DEFAULT_SPECTROGRAM_DIR = "/Volumes/Evo/precompute/mygardenbird_spectrograms_64x300"

# SpecAugment settings
SPECAUGMENT_FREQ_MASK = 8
SPECAUGMENT_TIME_MASK = 20
SPECAUGMENT_NUM_MASKS = 2

# Global stats percentiles
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98

# Global stats sample size per class
GLOBAL_STATS_SAMPLES = 100


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
        from tf_keras.optimizers import AdamW
        Adam = AdamW
        print("Using AdamW optimizer (Linux)")
    except ImportError:
        from tf_keras.optimizers import Adam
        print("Using standard Adam (AdamW not available)")
else:
    from tf_keras.optimizers import Adam
    print(f"Using standard Adam ({system})")


# --------------------------------------------------------------
# CACHE MANAGEMENT
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
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--calib_samples", type=int, default=200)
    parser.add_argument("--width_mult", type=float, default=0.35,
                        help="Width multiplier for MobileNetV2 (0.25, 0.35, 0.5, 0.75, 1.0)")

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

    # Configurable paths
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Path to dataset directory")
    parser.add_argument("--spectrogram_dir", type=str, default=DEFAULT_SPECTROGRAM_DIR,
                        help="Path to spectrogram cache directory")

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
        f"results/8a_mobilenetv2_64x300_ptq_"
        f"width{int(args.width_mult * 100):02d}_"
        f"drop{int(args.dropout * 100):02d}_"
        f"rand{args.random_seed}_"
        f"warm{args.warmup_epochs}_"
        f"{aug_suffix}_"
        f"{platform.system().lower()}"
    )

    # Clean up double underscores if aug_suffix is empty
    output_dir_name = output_dir_name.replace("__", "_").rstrip("_")

    config = {
        'warmup_epochs': args.warmup_epochs,
        'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size,
        'warmup_lr': args.warmup_lr,
        'finetune_lr': args.finetune_lr,
        'dropout': args.dropout,
        'width_mult': args.width_mult,
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
        'dataset_dir': args.dataset_dir,
        'spectrogram_dir': args.spectrogram_dir,
        'lr_schedule': args.lr_schedule,
        'random_seed': args.random_seed,
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    return config


# --------------------------------------------------------------
# LOGGING UTILITIES
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
            f.write("MODEL 8A: NARROW MOBILENETV2 @ 64x300 NATIVE\n")
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
            f.write(f"  Model Type:             MobileNetV2 Narrow (Model 8a)\n")
            f.write(f"  Input Shape:            {config['input_shape']} (NATIVE, NO RESIZE)\n")
            f.write(f"  Width Multiplier:       {config['width_mult']}\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  Block Type:             Inverted Residual (expand->dw->project)\n")
            f.write(f"  Linear Bottleneck:      Yes (no activation after projection)\n")
            f.write(f"  Skip Connections:       Yes (when stride=1 and channels match)\n")
            f.write(f"  Activation:             ReLU6 (quantization-friendly)\n")
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
                f.write(f"  Time Shift:             +/-{config['time_shift_ms']} ms\n")
                f.write(f"  Pitch Shift:            +/-{config['pitch_shift_steps']} semitones\n")
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
            f.write(f"  Dataset:                {config['dataset_dir']}\n")
            f.write(f"  Spectrogram Cache:      {config['spectrogram_dir']}\n")
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
                f.write(f"\n  WARNING: Model may exceed 512 KB target for Cortex-M7\n")
            else:
                f.write(f"\n  Model size within 512 KB target\n")

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
            f.write(f"Model: 8a_mobilenetv2 | Width: {config['width_mult']} | Dropout: {config['dropout']} | "
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
                f.write("Excellent (no degradation)\n")
            elif drop > 0:
                f.write("INT8 better! (quantization as regularizer)\n")
            elif drop > -2:
                f.write("Good (<2% drop)\n")
            elif drop > -5:
                f.write("Acceptable (2-5% drop)\n")
            else:
                f.write("High degradation (>5% drop)\n")

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
                f.write(" No overfitting\n")
            elif overfitting_gap < 5:
                f.write(" Slight overfitting\n")
            else:
                f.write(" Overfitting detected\n")

            f.write(f"\nExecution Time:\n")
            f.write(f"  Total Duration:         {format_time(total_time)}\n")

            f.write(f"\nTraining completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # CSV format
            f.write("\n" + "=" * 80 + "\n")
            f.write("CSV FORMAT (for batch comparison)\n")
            f.write("=" * 80 + "\n")
            f.write("model,width_mult,dropout,augmentation,warmup_epochs,finetune_epochs,warmup_lr,finetune_lr,"
                    "lr_schedule,fp32_acc,int8_acc,drop,best_val_acc,train_val_gap,train_time_sec,model_size_kb\n")
            f.write(f"8a_mobilenetv2,{config['width_mult']},{config['dropout']},{config['augmentation_mode']},"
                    f"{config['warmup_epochs']},{config['finetune_epochs']},{config['warmup_lr']},{config['finetune_lr']},"
                    f"{config['lr_schedule']},{fp32_acc:.2f},{int8_acc:.2f},{drop:.2f},"
                    f"{max(finetune_history.history['val_accuracy']) * 100:.2f},"
                    f"{overfitting_gap:.2f},{int(total_time)},"
                    f"{os.path.getsize(os.path.join(config['output_dir'], 'mobilenetv2_int8.tflite')) / 1024:.1f}\n")


# --------------------------------------------------------------
# GLOBAL STATS COMPUTATION
# --------------------------------------------------------------
def compute_global_stats(data_dir):
    """Compute global normalization statistics from dataset."""
    all_mel = []
    total_sampled = 0

    print(f"Computing global stats (sampling up to {GLOBAL_STATS_SAMPLES} files per class)...")

    # Use train directory for computing global stats
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        train_dir = data_dir  # Fallback to flat structure

    for class_name in sorted(os.listdir(train_dir)):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue

        wavs = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
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
                    n_mels=N_MELS, fmax=FMAX, center=True,
                    power=2.0, window='hann'
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                all_mel.append(mel_db.flatten())
                total_sampled += 1
            except Exception as e:
                print(f"\nFailed to process {f} during stats computation: {e}")
                continue

    if len(all_mel) == 0:
        raise RuntimeError("No valid audio files found for computing global stats")

    all_mel = np.concatenate(all_mel)
    gmin, gmax = np.percentile(all_mel, PERCENTILE_LOW), np.percentile(all_mel, PERCENTILE_HIGH)
    print(f"Global stats computed from {total_sampled} files: {gmin:.2f} -> {gmax:.2f} dB")
    return float(gmin), float(gmax)


def compute_spec(audio, sr, gmin, gmax):
    """Compute and normalize mel spectrogram with shape validation."""
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

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, gmin, gmax)
    mel_norm = (mel_db - gmin) / (gmax - gmin + 1e-8)
    return mel_norm[..., np.newaxis].astype(np.float32)


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
# MOBILENETV2 NARROW - ADAPTED FOR 64x300 INPUT
# --------------------------------------------------------------
def inverted_residual_block(x, expansion, filters, stride, block_id):
    """
    MobileNetV2 inverted residual block.

    Pattern: Expand (1x1) -> Depthwise (3x3) -> Project (1x1, linear)

    Key features:
    - Expansion: Increase channels before depthwise conv
    - Linear bottleneck: No activation after final 1x1 conv
    - Residual: Skip connection when stride=1 and channels match

    Args:
        x: Input tensor
        expansion: Expansion factor for the block
        filters: Number of output filters
        stride: Stride for depthwise conv
        block_id: Block identifier for naming
    """
    prefix = f'block{block_id}_'
    input_filters = x.shape[-1]
    expanded_filters = input_filters * expansion

    # Check if we can use residual connection
    use_residual = (stride == 1) and (input_filters == filters)

    # Expansion phase (skip if expansion == 1)
    if expansion != 1:
        x_expanded = layers.Conv2D(
            expanded_filters, 1, padding='same',
            use_bias=False, name=prefix + 'expand'
        )(x)
        x_expanded = layers.BatchNormalization(name=prefix + 'expand_bn')(x_expanded)
        x_expanded = layers.ReLU(6., name=prefix + 'expand_relu')(x_expanded)
    else:
        x_expanded = x

    # Depthwise convolution
    x_dw = layers.DepthwiseConv2D(
        3, strides=stride, padding='same',
        use_bias=False, name=prefix + 'depthwise'
    )(x_expanded)
    x_dw = layers.BatchNormalization(name=prefix + 'depthwise_bn')(x_dw)
    x_dw = layers.ReLU(6., name=prefix + 'depthwise_relu')(x_dw)

    # Project (linear bottleneck - NO activation)
    x_proj = layers.Conv2D(
        filters, 1, padding='same',
        use_bias=False, name=prefix + 'project'
    )(x_dw)
    x_proj = layers.BatchNormalization(name=prefix + 'project_bn')(x_proj)
    # Note: No activation here (linear bottleneck)

    # Residual connection
    if use_residual:
        x_proj = layers.Add(name=prefix + 'add')([x, x_proj])

    return x_proj


def create_mobilenetv2_narrow_64x300(num_classes, input_shape, width_mult=0.35, dropout=0.3):
    """
    Create Narrow MobileNetV2 adapted for 64x300 spectrogram input.

    Architecture based on MobileNetV2 (Sandler et al., 2018):
    - Inverted residual blocks with linear bottleneck
    - ReLU6 activation (quantization-friendly)
    - Width multiplier to reduce model size

    Key adaptations for 64x300:
    - First conv stride=1 to preserve resolution
    - Adjusted channel counts with width multiplier
    - Fewer downsampling steps to handle non-square input

    Args:
        num_classes: Number of output classes
        input_shape: Input shape (64, 300, 1)
        width_mult: Width multiplier (0.25, 0.35, 0.5, 0.75, 1.0)
        dropout: Dropout rate

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)

    # Helper to apply width multiplier
    def _make_divisible(v, divisor=8, min_value=8):
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def ch(channels):
        return _make_divisible(channels * width_mult)

    # Initial conv - stride=1 to preserve 64x300 resolution
    x = layers.Conv2D(
        ch(32), 3, strides=1, padding='same',
        use_bias=False, name='conv_stem'
    )(inputs)
    x = layers.BatchNormalization(name='conv_stem_bn')(x)
    x = layers.ReLU(6., name='conv_stem_relu')(x)

    # MobileNetV2 blocks
    # Format: (expansion, filters, stride, num_repeats)
    # Adapted for 64x300: adjusted downsampling
    block_configs = [
        # t (expansion), c (filters), s (stride), n (repeats)
        (1,  16,  1, 1),   # 64x300
        (6,  24,  2, 2),   # 32x150
        (6,  32,  2, 3),   # 16x75
        (6,  64,  2, 4),   # 8x37
        (6,  96,  1, 3),   # 8x37
        (6, 160,  2, 3),   # 4x18
        (6, 320,  1, 1),   # 4x18
    ]

    block_id = 0
    for t, c, s, n in block_configs:
        for i in range(n):
            stride = s if i == 0 else 1
            x = inverted_residual_block(x, t, ch(c), stride, block_id)
            block_id += 1

    # Final conv
    x = layers.Conv2D(
        ch(1280), 1, padding='same',
        use_bias=False, name='conv_head'
    )(x)
    x = layers.BatchNormalization(name='conv_head_bn')(x)
    x = layers.ReLU(6., name='conv_head_relu')(x)

    # Global pooling and classification
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return keras.Model(inputs, outputs, name="MobileNetV2_Narrow_64x300")


# --------------------------------------------------------------
# TFLITE CONVERSION & EVALUATION
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
    print(f"Saved INT8 TFLite: {path} ({os.path.getsize(path) / 1024:.1f} KB)")


def _save_classification_report(y_test, y_pred, class_names, output_dir, model_type):
    """Save classification report to file and print to console."""
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, f'classification_report_{model_type.lower()}.txt')
    with open(report_path, 'w') as f:
        f.write(f"{model_type} Model Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print(f"Saved classification report: {report_path}")
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
    print("Saved training history plot")


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


def load_data(data_dir, gmin, gmax, augmentation_mode='none',
              time_shift_ms=100, pitch_shift_steps=2, mixup_alpha=0.2):
    """Load data with fixed 75/10/15 split per class."""
    if augmentation_mode == 'none':
        print("\nWARNING: No augmentation enabled")
        print("  For better results, try --augment, --mixup, or --specaugment\n")

    X_test, y_test, test_paths = [], [], []
    X_val, y_val, val_paths = [], [], []
    X_train, y_train, train_paths = [], [], []
    labels = {}
    idx = 0
    failed_files = []

    # Count total files for progress bar (from fixed directories)
    total_files = 0

    for split in ['test', 'val', 'train']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir) and not class_name.startswith('.'):
                    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                    total_files += len(files)

    # Add augmented samples to total if enabled (not for mixup)
    train_dir = os.path.join(data_dir, 'train')
    train_count = 0
    if os.path.exists(train_dir):
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.isdir(class_dir) and not class_name.startswith('.'):
                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                train_count += len(files)

    if augmentation_mode in ['baseline', 'specaugment']:
        total_files += train_count

    print(f"\nDataset Structure: Fixed 75:10:15 split from directories")
    print(f"  Dataset root: {data_dir}")
    print(f"  Test dir:  {os.path.join(data_dir, 'test')}")
    print(f"  Val dir:   {os.path.join(data_dir, 'val')}")
    print(f"  Train dir: {os.path.join(data_dir, 'train')}")

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

    with tqdm(total=total_files, desc="Loading from fixed directories", unit="file") as pbar:
        # Load TEST set (never augmented)
        test_dir = os.path.join(data_dir, 'test')
        if os.path.exists(test_dir):
            for class_name in sorted(os.listdir(test_dir)):
                class_dir = os.path.join(test_dir, class_name)
                if not os.path.isdir(class_dir) or class_name.startswith('.'):
                    continue

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                for f in files:
                    try:
                        audio, _ = librosa.load(os.path.join(class_dir, f), sr=TARGET_SR)
                        audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                            np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                        spec = compute_spec(audio, TARGET_SR, gmin, gmax)
                        X_test.append(spec)
                        y_test.append(labels[class_name])
                        test_paths.append(os.path.join(class_dir, f))
                        pbar.update(1)
                    except Exception as e:
                        failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                        pbar.update(1)
                        continue

        # Load VALIDATION set (never augmented)
        val_dir = os.path.join(data_dir, 'val')
        if os.path.exists(val_dir):
            for class_name in sorted(os.listdir(val_dir)):
                class_dir = os.path.join(val_dir, class_name)
                if not os.path.isdir(class_dir) or class_name.startswith('.'):
                    continue

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                for f in files:
                    try:
                        audio, _ = librosa.load(os.path.join(class_dir, f), sr=TARGET_SR)
                        audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                            np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                        spec = compute_spec(audio, TARGET_SR, gmin, gmax)
                        X_val.append(spec)
                        y_val.append(labels[class_name])
                        val_paths.append(os.path.join(class_dir, f))
                        pbar.update(1)
                    except Exception as e:
                        failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                        pbar.update(1)
                        continue

        # Load TRAINING set (with optional augmentation)
        train_dir = os.path.join(data_dir, 'train')
        if os.path.exists(train_dir):
            for class_name in sorted(os.listdir(train_dir)):
                class_dir = os.path.join(train_dir, class_name)
                if not os.path.isdir(class_dir) or class_name.startswith('.'):
                    continue

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                for f in files:
                    try:
                        audio, _ = librosa.load(os.path.join(class_dir, f), sr=TARGET_SR)
                        audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                            np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                        # Original sample
                        spec = compute_spec(audio, TARGET_SR, gmin, gmax)
                        X_train.append(spec)
                        y_train.append(labels[class_name])
                        train_paths.append(os.path.join(class_dir, f))
                        pbar.update(1)

                        # Augmented sample (mode-dependent)
                        if augmentation_mode == 'baseline':
                            aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                            aug_spec = compute_spec(aug_audio, TARGET_SR, gmin, gmax)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(os.path.join(class_dir, f) + "_aug")
                            pbar.update(1)
                        elif augmentation_mode == 'specaugment':
                            aug_spec = augment_specaugment(spec)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(os.path.join(class_dir, f) + "_specaug")
                            pbar.update(1)

                    except Exception as e:
                        failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                        if augmentation_mode in ['baseline', 'specaugment']:
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

    if len(failed_files) > 0:
        print(f"\nWarning: {len(failed_files)} files failed to load")
        with open('data_loading_errors.txt', 'w') as f:
            for error in failed_files:
                f.write(error + '\n')

    print(f"\nFixed Split Complete:")
    print(f"  Test (held-out):        {len(X_test):5d} samples ({len(X_test) // 10} per class)")
    print(f"  Val (held-out):         {len(X_val):5d} samples ({len(X_val) // 10} per class)")
    print(f"  Train (w/ augment):     {len(X_train):5d} samples")
    print(f"  Total:                  {len(X_test) + len(X_val) + len(X_train):5d} samples")
    print(f"\n  NO DATA LEAKAGE: Test and Val samples never augmented")

    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)


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

    print(f"\nModel 8a: Narrow MobileNetV2 @ 64x300 (NATIVE)")
    print(f"  Width multiplier: {config['width_mult']}")
    print(f"  Random seed: {config['random_seed']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Augmentation: {config['augmentation_mode']}")

    logger.log_hyperparameters(config)

    # Compute global stats
    print("\nComputing global normalization stats...")
    global_min, global_max = compute_global_stats(config['dataset_dir'])

    # Load data
    print("\nLoading dataset with FIXED 75/10/15 SPLIT...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels, failed_count = load_data(
        config['dataset_dir'], global_min, global_max,
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
    print(f"\nCalibration set: {len(X_calib)} samples")

    # Create model
    print(f"\n{'=' * 70}")
    print("CREATING NARROW MOBILENETV2 MODEL (64x300 NATIVE)")
    print(f"{'=' * 70}")
    model = create_mobilenetv2_narrow_64x300(
        num_classes, config['input_shape'],
        width_mult=config['width_mult'],
        dropout=config['dropout']
    )
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

    fp32_path = os.path.join(config['output_dir'], 'mobilenetv2_fp32.keras')
    model.save(fp32_path)
    print(f"Saved FP32 model: {fp32_path}")

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
    int8_path = os.path.join(config['output_dir'], 'mobilenetv2_int8.tflite')
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
    print(f"Model:                   8a_mobilenetv2 (64x300 NATIVE)")
    print(f"Width Multiplier:        {config['width_mult']}")
    print(f"Augmentation:            {config['augmentation_mode']}")
    print(f"FP32 Accuracy:           {fp32_acc:6.2f}%")
    print(f"INT8 Accuracy:           {int8_acc:6.2f}%")
    print(f"Accuracy Drop:           {drop:6.2f}%")
    print(f"INT8 Model Size:         {os.path.getsize(int8_path) / 1024:.1f} KB")
    print(f"Total Time:              {format_time(total_time)}")
    print(f"{'=' * 70}")

    print(f"\nComplete training report: {logger.log_path}")
    print(f"All results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()

    total_script_time = time.time() - script_start
    print(f"\n{'=' * 70}")
    print(f"Script completed in: {format_time(total_script_time)}")
    print(f"{'=' * 70}")
