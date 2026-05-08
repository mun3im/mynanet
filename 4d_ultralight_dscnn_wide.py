#!/usr/bin/env python3
"""
Ablation Study 6c: Custom DS-CNN Inspired by MobileNetV3 for Mygardenbird Classification
Fixed spectrogram shape: 64x300 (10ms per frame) - NATIVE INPUT, NO RESIZE
Data Split: Fixed 75:10:15 (train/val/test) from dataset directories

Hypothesis: Standard CNNs like MobileNetV3 are designed for 224×224 RGB images and
are suboptimal for native spectrograms. Instead, we use a lightweight, custom
depthwise-separable CNN (DS-CNN) inspired by MobileNet’s efficiency principles:
- Depthwise separable convolutions
- Minimal channel counts
- No squeeze-excitation or hard-swish (for INT8 stability)
- Direct support for 64×300 mel-spectrograms

Architecture: UltraLight DS-CNN Wide (not MobileNetV3)
- Input: (64, 300, 1) — no resize, no distortion
- Blocks: 7 depthwise-separable layers with strategic strides
- Activation: ReLU (robust under PTQ)
- Global Average Pooling + Dropout + Dense
- Target: ~1.15M params → ~460 KB INT8 (fits in 512 KB Cortex-M7 budget)

Pipeline: Based on 7d with identical preprocessing, split, and PTQ.
Expected outcome: Validate if a purpose-built DS-CNN outperforms generic backbones
on native spectrograms without costly resizing or architectural mismatch.
"""
print("\n")
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
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks

    print(f"✓ TensorFlow version: {tf.__version__}")
    print("✓ Keras: built-in (from TensorFlow)")

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
if system == "Darwin":  # macOS (including Apple Silicon)
    from tensorflow.keras.optimizers.legacy import Adam
    print("Using LEGACY Adam (macOS)")
elif system == "Linux":
    try:
        from tensorflow.keras.optimizers import AdamW
        Adam = AdamW
        print("Using AdamW optimizer (Linux)")
    except ImportError:
        from tensorflow.keras.optimizers import Adam
        print("Using standard Adam (AdamW not available on Linux)")
else:
    from tensorflow.keras.optimizers import Adam
    print(f"Using standard Adam ({system})")

# --------------------------------------------------------------
# CACHE MANAGEMENT
# --------------------------------------------------------------
def compute_cache_hash(config_params):
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
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--calib_samples", type=int, default=200)

    # Augmentation flags
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--specaugment", action='store_true')

    # Baseline augmentation parameters
    parser.add_argument("--time_shift_ms", type=int, default=100)
    parser.add_argument("--pitch_shift_steps", type=int, default=2)

    # GPU parameters
    parser.add_argument("--force_cpu", action='store_true')
    parser.add_argument("--gpu_memory_limit", type=int, default=None)

    # Configurable paths
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--spectrogram_dir", type=str, default=DEFAULT_SPECTROGRAM_DIR)

    # LR schedule
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "plateau", "both", "none"])

    # Random seed
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_STATE)

    args = parser.parse_args()

    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

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
        f"results/6c_ultralight_dscnn_wide_64x300_ptq_"
        f"drop{int(args.dropout * 100):02d}_"
        f"rand{args.random_seed}_"
        f"warm{args.warmup_epochs}_"
        f"{aug_suffix}_"
        f"{platform.system().lower()}"
    )
    output_dir_name = output_dir_name.replace("__", "_").rstrip("_")

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
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, 'training_report.txt')
        self.start_time = time.time()
        self.stage_times = {}
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION 6C: ULTRALIGHT DS-CNN WIDE @ 64×300 NATIVE\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Platform: {platform.system()} {platform.machine()}\n")
            f.write(f"Python: {sys.version.split()[0]}\n")
            f.write(f"TensorFlow: {tf.__version__}\n")
            f.write(f"Keras: built-in (TensorFlow {tf.__version__})\n")

    def log_section(self, title):
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n")

    def log_hyperparameters(self, config):
        self.log_section("HYPERPARAMETERS")
        with open(self.log_path, 'a') as f:
            f.write("\nSystem Configuration:\n")
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
            f.write(f"  Window Length:          400 samples (25ms at 16kHz)\n")
            f.write(f"  Hop Length:             {HOP_LENGTH} samples (10.0 ms)\n")
            f.write(f"  Mel Bins (N_MELS):      {N_MELS}\n")
            f.write(f"  Max Frequency (FMAX):   {FMAX} Hz\n")
            f.write(f"  Time Frames:            {TIME_FRAMES} (FIXED)\n")
            f.write(f"  Spectrogram Shape:      {N_MELS}x{TIME_FRAMES}\n")

            f.write("\nModel Architecture:\n")
            f.write(f"  Model Type:             Custom DS-CNN (MobileNet-inspired)\n")
            f.write(f"  Input Shape:            {config['input_shape']} (NATIVE, NO RESIZE)\n")
            f.write(f"  Hypothesis:             Generic CNNs (e.g., MobileNet) are suboptimal for spectrograms\n")
            f.write(f"  Design:                 Depthwise separable blocks, ReLU, no SE/hard-swish\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  Global Pooling:         GlobalAveragePooling2D\n")
            f.write(f"  Quantization Target:    INT8 optimized architecture\n")

            f.write("\nTraining Configuration:\n")
            f.write(f"  Random Seed:            {config['random_seed']}\n")
            f.write(f"  Warmup Epochs:          {config['warmup_epochs']}\n")
            f.write(f"  Fine-tune Epochs:       {config['finetune_epochs']}\n")
            f.write(f"  Batch Size:             {config['batch_size']}\n")
            f.write(f"  Optimizer:              Adam (Legacy on Apple Silicon)\n")
            f.write(f"  Loss Function:          Sparse Categorical Crossentropy\n")

            f.write("\nData Augmentation:\n")
            f.write(f"  Mode:                   {config['augmentation_mode']}\n")
            if config['augmentation_mode'] == 'baseline':
                f.write(f"  Type:                   Baseline (Time/Pitch Shift)\n")
                f.write(f"  Time Shift:             ±{config['time_shift_ms']} ms\n")
                f.write(f"  Pitch Shift:            ±{config['pitch_shift_steps']} semitones\n")
            elif config['augmentation_mode'] == 'mixup':
                f.write(f"  Type:                   Mixup\n")
                f.write(f"  Alpha:                  {config['mixup_alpha']}\n")
            elif config['augmentation_mode'] == 'specaugment':
                f.write(f"  Type:                   SpecAugment\n")
                f.write(f"  Frequency Mask:         {SPECAUGMENT_FREQ_MASK} bins\n")
                f.write(f"  Time Mask:              {SPECAUGMENT_TIME_MASK} frames\n")

            f.write("\nQuantization:\n")
            f.write(f"  Method:                 Post-Training Quantization (PTQ)\n")
            f.write(f"  Target Format:          INT8 TFLite\n")
            f.write(f"  Calibration Samples:    {config['calib_samples']}\n")

            f.write("\nDeployment Target:\n")
            f.write(f"  Platform:               ARM Cortex-M7\n")
            f.write(f"  Memory Target:          <512 KB\n")

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
        with open(self.log_path, 'a') as f:
            import io
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_str = stream.getvalue()
            f.write("\n" + summary_str)
            total_params = model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            fp32_size_mb = total_params * 4 / (1024 ** 2)
            int8_size_kb = total_params / 1024
            f.write(f"\nEstimated Model Sizes:\n")
            f.write(f"  FP32 (4 bytes/param):   {fp32_size_mb:.2f} MB\n")
            f.write(f"  INT8 (1 byte/param):    {int8_size_kb:.1f} KB\n")
            if int8_size_kb > 512:
                f.write(f"\n⚠ WARNING: Model may exceed 512 KB target for Cortex-M7\n")
            else:
                f.write(f"\n✓ Model size within 512 KB target\n")

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
                final_epoch = len(history.history['loss'])
                f.write(f"\nTraining History:\n")
                f.write(f"  Epochs Completed:       {final_epoch}\n")
                f.write(f"  Final Train Loss:       {history.history['loss'][-1]:.4f}\n")
                f.write(f"  Final Train Accuracy:   {history.history['accuracy'][-1]:.4f}\n")
                f.write(f"  Final Val Loss:         {history.history['val_loss'][-1]:.4f}\n")
                f.write(f"  Final Val Accuracy:     {history.history['val_accuracy'][-1]:.4f}\n")

    def log_evaluation(self, model_name, accuracy, report_path):
        with open(self.log_path, 'a') as f:
            f.write(f"\n{model_name} Evaluation:\n")
            f.write(f"  Test Accuracy:          {accuracy:.2f}%\n")
            f.write(f"  Classification Report:  {report_path}\n")

    def log_final_results(self, fp32_acc, int8_acc, model_sizes,
                          warmup_history, finetune_history, config):
        self.log_section("FINAL RESULTS SUMMARY")
        drop = fp32_acc - int8_acc
        total_time = time.time() - script_start
        with open(self.log_path, 'a') as f:
            f.write(f"\nAccuracy Results:\n")
            f.write(f"  FP32 (.keras):          {fp32_acc:6.2f}%\n")
            f.write(f"  INT8 (TFLite):          {int8_acc:6.2f}%\n")
            f.write(f"\nAccuracy Change (INT8 vs FP32): {drop:+6.2f}%\n")
            f.write(f"\nModel Sizes:\n")
            for model_type, size_info in model_sizes.items():
                f.write(f"  {model_type:20s}: {size_info}\n")
            f.write(f"\nExecution Time: {format_time(total_time)}\n")

# --------------------------------------------------------------
# DATA PROCESSING (identical to 7d)
# --------------------------------------------------------------
def compute_global_stats(data_dir):
    all_mel = []
    total_sampled = 0
    print(f"Computing global stats (sampling up to {GLOBAL_STATS_SAMPLES} files per class)...")
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        train_dir = data_dir
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
                continue
    if len(all_mel) == 0:
        raise RuntimeError("No valid audio files found for computing global stats")
    all_mel = np.concatenate(all_mel)
    gmin, gmax = np.percentile(all_mel, PERCENTILE_LOW), np.percentile(all_mel, PERCENTILE_HIGH)
    print(f"✓ Global stats computed from {total_sampled} files: {gmin:.2f} → {gmax:.2f} dB")
    return float(gmin), float(gmax)

def compute_spec(audio, sr, gmin, gmax):
    WIN_LENGTH = 400
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT,
        win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX, center=True,
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

def augment_baseline(audio, sr, time_shift_ms=100, pitch_steps=2):
    if np.random.rand() > 0.5:
        shift_samples = int(np.random.uniform(-time_shift_ms, time_shift_ms) * sr / 1000)
        audio = np.roll(audio, shift_samples)
    if np.random.rand() > 0.5:
        n_steps = np.random.uniform(-pitch_steps, pitch_steps)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return audio

def augment_specaugment(spec):
    spec_aug = spec.copy()
    freq_bins, time_bins, _ = spec_aug.shape
    for _ in range(SPECAUGMENT_NUM_MASKS):
        f_mask = np.random.randint(0, SPECAUGMENT_FREQ_MASK)
        f0 = np.random.randint(0, max(1, freq_bins - f_mask))
        spec_aug[f0:f0 + f_mask, :, 0] = 0
    for _ in range(SPECAUGMENT_NUM_MASKS):
        t_mask = np.random.randint(0, SPECAUGMENT_TIME_MASK)
        t0 = np.random.randint(0, max(1, time_bins - t_mask))
        spec_aug[:, t0:t0 + t_mask, 0] = 0
    return spec_aug

# --------------------------------------------------------------
# CUSTOM DS-CNN MODEL (MobileNet-inspired, not MobileNet)
# --------------------------------------------------------------
def ds_block(x, out_filters, kernel_size=(3, 3), strides=(1, 1), use_residual=True, block_name="ds_block"):
    shortcut = x
    x = layers.DepthwiseConv2D(
        kernel_size, strides=strides, padding='same', use_bias=False,
        name=f"{block_name}_dw"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_dw_bn")(x)
    x = layers.ReLU(name=f"{block_name}_dw_relu")(x)
    x = layers.Conv2D(
        out_filters, (1, 1), use_bias=False,
        name=f"{block_name}_pw"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_pw_bn")(x)
    x = layers.ReLU(name=f"{block_name}_pw_relu")(x)
    if use_residual and \
       shortcut.shape[1] == x.shape[1] and \
       shortcut.shape[2] == x.shape[2] and \
       shortcut.shape[3] == x.shape[3]:
        x = layers.Add(name=f"{block_name}_add")([x, shortcut])
    return x

def create_ultralight_ds_cnn_wide(num_classes, input_shape=(64, 300, 1), dropout=0.2):
    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False, name='stem_dw')(inputs)
    x = layers.BatchNormalization(name='stem_dw_bn')(x)
    x = layers.ReLU(name='stem_dw_relu')(x)
    x = layers.Conv2D(32, 1, use_bias=False, name='stem_pw')(x)  # ↑ 20 → 32
    x = layers.BatchNormalization(name='stem_pw_bn')(x)
    x = layers.ReLU(name='stem_pw_relu')(x)

    # Block 1: time reduction
    x = ds_block(x, 48, kernel_size=(3, 5), strides=(1, 1), use_residual=False, block_name="block1")  # ↑ 28 → 48
    x = layers.MaxPooling2D(pool_size=(1, 2), name="block1_pool")(x)

    x = ds_block(x, 48, kernel_size=(3, 3), strides=(1, 1), use_residual=True, block_name="block2")

    # Block 3: freq reduction
    x = ds_block(x, 64, kernel_size=(3, 3), strides=(1, 1), use_residual=False, block_name="block3")  # ↑ 36 → 64
    x = layers.MaxPooling2D(pool_size=(2, 1), name="block3_pool")(x)

    x = ds_block(x, 64, kernel_size=(3, 3), strides=(1, 1), use_residual=True, block_name="block4")

    # Block 5: joint reduction
    x = ds_block(x, 96, kernel_size=(3, 3), strides=(2, 2), use_residual=False, block_name="block5")  # ↑ 56 → 96
    x = ds_block(x, 96, kernel_size=(3, 3), strides=(1, 1), use_residual=True, block_name="block6")

    # Block 7: time reduction
    x = ds_block(x, 128, kernel_size=(2, 3), strides=(1, 1), use_residual=False, block_name="block7")  # ↑ 80 → 128
    x = layers.MaxPooling2D(pool_size=(1, 2), name="block7_pool")(x)

    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)
    return keras.Model(inputs, outputs, name="UltraLight_DS_CNN_Wide")

# --------------------------------------------------------------
# TFLITE CONVERSION & EVALUATION
# --------------------------------------------------------------
def convert_to_tflite_int8(model, X_calib, path):
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
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, f'classification_report_{model_type.lower()}.txt')
    with open(report_path, 'w') as f:
        f.write(f"{model_type} Model Classification Report\n")
        f.write("=" * 70 + "\n")
        f.write(report)
    print(f"✓ Saved classification report: {report_path}")
    return report_path

def _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, accuracy):
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(warmup_hist.history['accuracy'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[0].plot(warmup_hist.history['val_accuracy'], label='Val (Warmup)', color='blue', linestyle='--')
    offset = len(warmup_hist.history['accuracy'])
    epochs_finetune = range(offset, offset + len(finetune_hist.history['accuracy']))
    axes[0].plot(epochs_finetune, finetune_hist.history['accuracy'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[0].plot(epochs_finetune, finetune_hist.history['val_accuracy'], label='Val (Finetune)', color='red', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(warmup_hist.history['loss'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[1].plot(warmup_hist.history['val_loss'], label='Val (Warmup)', color='blue', linestyle='--')
    axes[1].plot(epochs_finetune, finetune_hist.history['loss'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[1].plot(epochs_finetune, finetune_hist.history['val_loss'], label='Val (Finetune)', color='red', linestyle='--')
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
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred) * 100
    _save_classification_report(y_test, y_pred, class_names, output_dir, model_type)
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, acc)
    return acc

def evaluate_tflite(tflite_path, X_test, y_test, class_names, output_dir):
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
    X_test, y_test, test_paths = [], [], []
    X_val, y_val, val_paths = [], [], []
    X_train, y_train, train_paths = [], [], []
    labels = {}
    idx = 0
    failed_files = []

    total_files = 0
    for split in ['test', 'val', 'train']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir) and not class_name.startswith('.'):
                    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                    total_files += len(files)
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

    with tqdm(total=total_files, desc="Loading from fixed directories", unit="file") as pbar:
        for split, X_list, y_list, paths_list in [('test', X_test, y_test, test_paths),
                                                  ('val', X_val, y_val, val_paths),
                                                  ('train', X_train, y_train, train_paths)]:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
            for class_name in sorted(os.listdir(split_dir)):
                class_dir = os.path.join(split_dir, class_name)
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
                        X_list.append(spec)
                        y_list.append(labels[class_name])
                        paths_list.append(os.path.join(class_dir, f))
                        pbar.update(1)
                    except Exception as e:
                        failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                        pbar.update(1)
                        continue
                if split == 'train' and augmentation_mode in ['baseline', 'specaugment']:
                    for f in files:
                        try:
                            audio, _ = librosa.load(os.path.join(class_dir, f), sr=TARGET_SR)
                            audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                                    np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))
                            spec = compute_spec(audio, TARGET_SR, gmin, gmax)
                            if augmentation_mode == 'baseline':
                                aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                                aug_spec = compute_spec(aug_audio, TARGET_SR, gmin, gmax)
                                X_train.append(aug_spec)
                                y_train.append(labels[class_name])
                                pbar.update(1)
                            elif augmentation_mode == 'specaugment':
                                aug_spec = augment_specaugment(spec)
                                X_train.append(aug_spec)
                                y_train.append(labels[class_name])
                                pbar.update(1)
                        except Exception as e:
                            failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                            if augmentation_mode in ['baseline', 'specaugment']:
                                pbar.update(1)
                            continue

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    if len(failed_files) > 0:
        print(f"\n⚠ Warning: {len(failed_files)} files failed to load")
        with open('data_loading_errors.txt', 'w') as f:
            for error in failed_files:
                f.write(error + '\n')

    print(f"\n✓ Fixed Split Complete:")
    print(f"  Test (held-out):        {len(X_test):5d} samples")
    print(f"  Val (held-out):         {len(X_val):5d} samples")
    print(f"  Train (w/ augment):     {len(X_train):5d} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)

class MixupDataGenerator(keras.utils.Sequence):
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
    print(f"\nAblation 6c: Custom DS-CNN (MobileNet-inspired) @ 64×300 (NATIVE)")
    print(f"  Hypothesis: Generic CNNs are suboptimal for spectrograms")
    print(f"  Random seed: {config['random_seed']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Augmentation: {config['augmentation_mode']}")
    logger.log_hyperparameters(config)

    print("\nComputing global normalization stats...")
    global_min, global_max = compute_global_stats(config['dataset_dir'])

    print("\nLoading dataset with FIXED 90/60/450 SPLIT...")
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

    X_calib = X_val[:config['calib_samples']]
    print(f"\n✓ Calibration set: {len(X_calib)} samples")

    print(f"\n{'=' * 70}")
    print("CREATING CUSTOM DS-CNN MODEL (64×300 NATIVE)")
    print(f"{'=' * 70}")
    model = create_ultralight_ds_cnn_wide(num_classes, config['input_shape'], config['dropout'])
    model.summary()
    logger.log_model_info(model)

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

    model.compile(
        optimizer=Adam(learning_rate=config['warmup_lr']),
        loss=loss_function,
        metrics=['accuracy']
    )

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

    plot_training_history(warmup_history, finetune_history, config['output_dir'])
    fp32_path = os.path.join(config['output_dir'], 'mobilenetv3_fp32.keras')
    model.save(fp32_path)
    print(f"✓ Saved FP32 model: {fp32_path}")

    logger.start_stage("EVALUATION: FP32 (.keras)")
    print(f"\n{'=' * 70}")
    print("EVALUATING FP32 MODEL")
    print(f"{'=' * 70}")
    fp32_acc = evaluate_model(model, X_test, y_test, class_names,
                              config['output_dir'], "FP32")
    logger.log_evaluation("FP32 (.keras)", fp32_acc,
                          os.path.join(config['output_dir'], 'classification_report_fp32.txt'))

    logger.start_stage("TFLITE CONVERSION (PTQ)")
    print(f"\n{'=' * 70}")
    print("CONVERTING TO INT8 TFLITE")
    print(f"{'=' * 70}")
    int8_path = os.path.join(config['output_dir'], 'mobilenetv3_int8.tflite')
    convert_to_tflite_int8(model, X_calib, int8_path)

    logger.start_stage("EVALUATION: INT8 TFLite")
    print(f"\n{'=' * 70}")
    print("EVALUATING INT8 TFLite")
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
                             warmup_history, finetune_history, config)

    drop = fp32_acc - int8_acc
    total_time = time.time() - script_start
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Model:                   6c_ultralight_dscnn (64×300 NATIVE)")
    print(f"Augmentation:            {config['augmentation_mode']}")
    print(f"FP32 Accuracy:           {fp32_acc:6.2f}%")
    print(f"INT8 Accuracy:           {int8_acc:6.2f}%")
    print(f"Accuracy Drop:           {drop:6.2f}%")
    print(f"Total Time:              {format_time(total_time)}")
    print(f"\n✓ Complete training report: {logger.log_path}")
    print(f"✓ All results saved to: {config['output_dir']}/")

if __name__ == "__main__":
    main()
    total_script_time = time.time() - script_start
    print(f"\n{'=' * 70}")
    print(f"Script completed in: {format_time(total_script_time)}")
    print(f"{'=' * 70}")