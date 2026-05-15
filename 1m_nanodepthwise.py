#!/usr/bin/env python3
"""
Model 1m: NanoDepthwise-CNN for Mygardenbird Classification
DS-CNN lineage: ultra-compact depthwise-separable architecture, no residual/SE/attention.
Achieved ~87% accuracy at <50 KB INT8 — extreme efficiency on Portenta H7.
NanoDepthwise-CNN Key Features:
- 4-stage depthwise separable blocks, channel ramp [32→64→128→192]
- No skip connections (minimises activation memory)
- ~100K params → ~25–40 KB INT8
- MCU-compatible ops: TFLite Micro on Cortex-M7
Input: 64×300 mel-spectrogram (native, no resize)
Accuracy: ~87% FP32/INT8 @ <50 KB INT8 (observed)
Target: Ultra-low footprint for multi-model or FreeRTOS co-existence on H7
"""
print("\n" + " ❤️ " * 30 + "\n")
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
# CONSTANTS (Same as 4a/7d)
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
        'n_fft': N_FFT, 'hop_length': HOP_LENGTH, 'n_mels': N_MELS,
        'fmax': FMAX, 'target_sr': TARGET_SR, 'time_frames': TIME_FRAMES,
        'audio_length': FIXED_AUDIO_LENGTH, 'center': True,
        'window': 'hann', 'win_length': 400,
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
    # Configurable paths
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
        f"results_mygardenbird_1_{platform.system().lower()}/"
        f"1m_nanodepthwise_"
        f"drop{int(args.dropout * 100):02d}_"
        f"rand{args.random_seed}_"
        f"warm{args.warmup_epochs}_"
        f"{aug_suffix}"
    )
    output_dir_name = output_dir_name.replace("__", "_").rstrip("_")
    n_classes = len([d for d in os.listdir(args.flat_dir)
                     if os.path.isdir(os.path.join(args.flat_dir, d)) and not d.startswith('.')])
    config = {
        'warmup_epochs': args.warmup_epochs, 'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size, 'warmup_lr': args.warmup_lr, 'finetune_lr': args.finetune_lr,
        'dropout': args.dropout, 'time_frames': TIME_FRAMES,
        'input_shape': (N_MELS, TIME_FRAMES, 1), 'output_dir': output_dir_name,
        'calib_samples': args.calib_samples, 'augmentation_mode': augmentation_mode,
        'mixup_alpha': args.mixup, 'time_shift_ms': args.time_shift_ms,
        'pitch_shift_steps': args.pitch_shift_steps, 'force_cpu': args.force_cpu,
        'gpu_memory_limit': args.gpu_memory_limit, 'splits_csv': args.splits_csv,
        'flat_dir': args.flat_dir, 'n_classes': n_classes,
        'lr_schedule': args.lr_schedule, 'random_seed': args.random_seed,
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
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SERIES 1M: NANODEPTHWISE-CNN @ 64×300 NATIVE (Model 1m)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Platform: {platform.system()} {platform.machine()}\n")
            f.write(f"Python: {sys.version.split()[0]}\n")
            f.write(f"TensorFlow: {tf.__version__}\n")
            f.write(f"Keras: {keras.__version__}\n\n")

    def log_section(self, title):
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n")

    def log_hyperparameters(self, config):
        self.log_section("HYPERPARAMETERS")
        with open(self.log_path, 'a') as f:
            f.write("\nSystem Configuration:\n")
            f.write(f"  Platform: {platform.system()} {platform.machine()}\n")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and not config.get('force_cpu', False):
                f.write(f"  GPU: {len(gpus)} device(s) detected\n  GPU Memory: Dynamic growth enabled\n")
                if config.get('gpu_memory_limit'): f.write(f"  GPU Memory Limit: {config['gpu_memory_limit']} MB\n")
            else: f.write(f"  Compute: CPU only\n")
            f.write("\nAudio Processing:\n")
            f.write(f"  Target Sample Rate:     {TARGET_SR} Hz\n")
            f.write(f"  Audio Length:           {AUDIO_LENGTH_SEC} seconds ({FIXED_AUDIO_LENGTH} samples)\n")
            f.write(f"  FFT Size (N_FFT):       {N_FFT}\n  FFT Window:             Hann\n")
            f.write(f"  Window Length:          400 samples (25ms at 16kHz)\n")
            f.write(f"  Hop Length:             {HOP_LENGTH} samples (10.0 ms)\n")
            f.write(f"  Mel Bins (N_MELS):      {N_MELS}\n  Max Frequency (FMAX):   {FMAX} Hz\n")
            f.write(f"  Time Frames:            {TIME_FRAMES} (FIXED)\n  Spectrogram Shape:      {N_MELS}x{TIME_FRAMES}\n")
            f.write(f"  Center Padding:         Enabled\n")
            f.write("\nModel Architecture:\n")
            f.write(f"  Model Type:             NanoDepthwise-CNN (Model 1m)\n")
            f.write(f"  Input Shape:            {config['input_shape']} (NATIVE, NO RESIZE)\n")
            f.write(f"  Key Features:           Depthwise separable, No skip connections\n")
            f.write(f"  Channel Ramp:           32 -> 64 -> 128 -> 192\n")
            f.write(f"  Target INT8 Size:       <50 KB (Optimized for Multi-model M7)\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  Activation:             ReLU\n")
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
            f.write(f"  Optimizer:              Adam\n  Loss Function:          Sparse Categorical Crossentropy\n")
            f.write("\nData Augmentation:\n")
            f.write(f"  Mode:                   {config['augmentation_mode']}\n")
            if config['augmentation_mode'] == 'baseline':
                f.write(f"  Type:                   Baseline (Time/Pitch Shift)\n")
                f.write(f"  Time Shift:             ±{config['time_shift_ms']} ms\n")
                f.write(f"  Pitch Shift:            ±{config['pitch_shift_steps']} semitones\n")
            elif config['augmentation_mode'] == 'mixup':
                f.write(f"  Type:                   Mixup (alpha={config['mixup_alpha']})\n")
            elif config['augmentation_mode'] == 'specaugment':
                f.write(f"  Type:                   SpecAugment (freq={SPECAUGMENT_FREQ_MASK}, time={SPECAUGMENT_TIME_MASK})\n")
            else: f.write(f"  Enabled:                False\n")
            f.write("\nQuantization:\n")
            f.write(f"  Method:                 Post-Training Quantization (PTQ)\n")
            f.write(f"  Target Format:          INT8 TFLite\n  Calibration Samples:    {config['calib_samples']}\n")
            f.write("\nDeployment Target:\n  Platform:               ARM Cortex-M7 (Portenta H7)\n  Memory Target:          <150 KB Peak SRAM\n")
            f.write("\nData Paths:\n")
            f.write(f"  Flat Dir:               {config['flat_dir']}\n  Splits CSV:             {config['splits_csv']}\n")
            f.write(f"  Output Directory:       {config['output_dir']}\n")

    def log_dataset_info(self, X, y, class_labels, X_train, X_val, X_test, failed_files=0):
        self.log_section("DATASET INFORMATION")
        total = len(X)
        with open(self.log_path, 'a') as f:
            f.write(f"\nTotal Samples:          {total}\nNumber of Classes:      {len(class_labels)}\n")
            if failed_files > 0: f.write(f"Failed Files:           {failed_files}\n")
            if total > 0:
                f.write(f"\nData Split (Fixed per-class):\n")
                f.write(f"  Training:             {len(X_train)} ({len(X_train) / total * 100:.1f}%)\n")
                f.write(f"  Validation:           {len(X_val)} ({len(X_val) / total * 100:.1f}%)\n")
                f.write(f"  Test:                 {len(X_test)} ({len(X_test) / total * 100:.1f}%)\n")
                f.write(f"\nClass Distribution:\n")
                for name, idx in sorted(class_labels.items(), key=lambda x: x[1]):
                    count = np.sum(y == idx)
                    f.write(f"  {name:30s}: {count:5d} samples ({count / total * 100:5.2f}%)\n")
            else:
                f.write("\n⚠ WARNING: Dataset is EMPTY. Check CSV filenames and paths.\n")

    def log_model_info(self, model):
        self.log_section("MODEL ARCHITECTURE")
        with open(self.log_path, 'a') as f:
            import io
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            f.write("\n" + stream.getvalue())
            total_params = model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            f.write(f"\nParameter Summary:\n  Total Parameters:       {total_params:,}\n  Trainable Parameters:   {trainable_params:,}\n")
            fp32_size_mb = total_params * 4 / (1024 ** 2)
            int8_size_kb = total_params / 1024
            f.write(f"\nEstimated Model Sizes:\n  FP32 (4 bytes/param):   {fp32_size_mb:.2f} MB\n  INT8 (1 byte/param):    {int8_size_kb:.1f} KB\n")
            if int8_size_kb > 150: f.write(f"\n⚠ WARNING: Model may exceed 150 KB target for multi-model concurrency\n")
            else: f.write(f"\n✓ Model size within 150 KB target (Ultra-efficient)\n")

    def start_stage(self, stage_name):
        self.stage_times[stage_name] = {'start': time.time()}
        self.log_section(stage_name)

    def end_stage(self, stage_name, history=None):
        if stage_name not in self.stage_times: return
        self.stage_times[stage_name]['end'] = time.time()
        elapsed = self.stage_times[stage_name]['end'] - self.stage_times[stage_name]['start']
        with open(self.log_path, 'a') as f:
            f.write(f"\nStage Duration: {format_time(elapsed)}\n")
            if history:
                f.write(f"\nTraining History:\n  Epochs Completed:       {len(history.history['loss'])}\n")
                f.write(f"  Final Train Loss:       {history.history['loss'][-1]:.4f}\n")
                f.write(f"  Final Train Accuracy:   {history.history['accuracy'][-1]:.4f}\n")
                f.write(f"  Final Val Loss:         {history.history['val_loss'][-1]:.4f}\n")
                f.write(f"  Final Val Accuracy:     {history.history['val_accuracy'][-1]:.4f}\n")
                f.write(f"  Best Val Loss:          {min(history.history['val_loss']):.4f}\n")
                f.write(f"  Best Val Accuracy:      {max(history.history['val_accuracy']):.4f}\n")

    def log_evaluation(self, model_name, accuracy, report_path):
        with open(self.log_path, 'a') as f:
            f.write(f"\n{model_name} Evaluation:\n  Test Accuracy:          {accuracy:.2f}%\n  Classification Report:  {report_path}\n")

    def log_final_results(self, fp32_acc, int8_acc, model_sizes,
                          warmup_history, finetune_history, config):
        self.log_section("FINAL RESULTS SUMMARY")
        drop = fp32_acc - int8_acc
        total_time = time.time() - script_start
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\nQUICK REFERENCE (Copy to spreadsheet)\n" + "=" * 80 + "\n")
            f.write(f"Model: 1m_nanodepthwise | Dropout: {config['dropout']} | Aug: {config['augmentation_mode']} | Seed: {config['random_seed']}\n")
            f.write(f"FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}% | Drop: {drop:+.2f}% | Time: {format_time(total_time)}\n")
            f.write("\n" + "-" * 80 + "\nDETAILED RESULTS\n" + "-" * 80 + "\n")
            f.write(f"\nAccuracy Results:\n  FP32 (.keras):          {fp32_acc:6.2f}%\n  INT8 (TFLite):          {int8_acc:6.2f}%\n")
            f.write(f"\nAccuracy Change (INT8 vs FP32):\n  Drop:                   {drop:+6.2f}% ")
            if abs(drop) < 0.5: f.write("✓ Excellent\n")
            elif drop > 0: f.write("✓✓ INT8 better!\n")
            elif drop > -2: f.write("✓ Good (<2% drop)\n")
            elif drop > -5: f.write("⚠ Acceptable (2-5% drop)\n")
            else: f.write("✗ High degradation (>5% drop)\n")
            f.write(f"\nModel Sizes:\n")
            for model_type, size_info in model_sizes.items(): f.write(f"  {model_type:20s}: {size_info}\n")
            f.write(f"\nTraining Metrics:\n  Best Warmup Val Acc:    {max(warmup_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Best Finetune Val Acc:  {max(finetune_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Final Train Acc:        {finetune_history.history['accuracy'][-1] * 100:6.2f}%\n")
            f.write(f"  Final Val Acc:          {finetune_history.history['val_accuracy'][-1] * 100:6.2f}%\n")
            overfitting_gap = finetune_history.history['accuracy'][-1] * 100 - finetune_history.history['val_accuracy'][-1] * 100
            f.write(f"  Train-Val Gap:          {overfitting_gap:+6.2f}%")
            if overfitting_gap < 2: f.write(" ✓ No overfitting\n")
            elif overfitting_gap < 5: f.write(" ⚠ Slight overfitting\n")
            else: f.write(" ✗ Overfitting detected\n")
            f.write(f"\nExecution Time:\n  Total Duration:         {format_time(total_time)}\n")
            f.write(f"\nTraining completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "=" * 80 + "\nCSV FORMAT (for batch comparison)\n" + "=" * 80 + "\n")
            f.write("model,dropout,augmentation,warmup_epochs,finetune_epochs,warmup_lr,finetune_lr,"
                    "lr_schedule,fp32_acc,int8_acc,drop,best_val_acc,train_val_gap,train_time_sec,model_size_kb\n")
            f.write(f"1m_nanodepthwise,{config['dropout']},{config['augmentation_mode']},"
                    f"{config['warmup_epochs']},{config['finetune_epochs']},{config['warmup_lr']},{config['finetune_lr']},"
                    f"{config['lr_schedule']},{fp32_acc:.2f},{int8_acc:.2f},{drop:.2f},"
                    f"{max(finetune_history.history['val_accuracy']) * 100:.2f},{overfitting_gap:.2f},{int(total_time)},"
                    f"{os.path.getsize(os.path.join(config['output_dir'], 'nanodepthwise_int8.tflite')) / 1024:.1f}\n")

# --------------------------------------------------------------
# DATA LOADING & AUGMENTATION
# --------------------------------------------------------------
def compute_spec(audio, sr, gmin=None, gmax=None):
    """Compute mel spectrogram with per-sample percentile normalisation → [0, 1]."""
    WIN_LENGTH = 400
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX, center=True, power=2.0, window='hann'
    )
    if mel.shape[1] > TIME_FRAMES: mel = mel[:, :TIME_FRAMES]
    if mel.shape[1] < TIME_FRAMES: mel = np.pad(mel, ((0, 0), (0, TIME_FRAMES - mel.shape[1])))
    mel_db = librosa.power_to_db(mel, top_db=None)
    p2, p98 = np.percentile(mel_db, (2, 98))
    if p98 > p2 + 1e-8:
        mel_norm = (np.clip(mel_db, p2, p98) - p2) / (p98 - p2)
    else:
        mel_norm = np.full_like(mel_db, 0.5)
    return mel_norm[..., np.newaxis].astype(np.float32)

def parse_splits_csv(csv_path):
    """Read splits CSV into a {filename: split} dict."""
    splits = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',', 1)
            if len(parts) != 2: continue
            key, split = parts[0].strip(), parts[1].strip()
            if key in ('filename', 'file_id'): continue
            # FIX: Ensure .wav extension and lowercase to match file_lookup
            key = key.lower()
            if not key.endswith('.wav'):
                key += '.wav'
            splits[key] = split
    return splits

def load_data_from_csv(csv_path, flat_dir, gmin=None, gmax=None, augmentation_mode='none',
                       time_shift_ms=100, pitch_shift_steps=2, mixup_alpha=0.2):
    """Load data using a flat directory + splits CSV."""
    if augmentation_mode == 'none':
        print("\n⚠ WARNING: No augmentation enabled. Consider --augment, --mixup, or --specaugment\n")
    splits = parse_splits_csv(csv_path)
    X_test, y_test = [], []
    X_val, y_val = [], []
    X_train, y_train = [], []
    labels = {}
    idx = 0
    failed_files = []
    csv_misses = 0
    file_lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'): continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'): file_lookup[f] = (class_name, os.path.join(class_dir, f))
            
    total_files = sum(1 for fn in splits if fn in file_lookup)
    train_count = sum(1 for fn, sp in splits.items() if fn in file_lookup and sp == 'train')
    if augmentation_mode in ['baseline', 'specaugment']: total_files += train_count
    
    print(f"\nDataset Structure: CSV-based split from {csv_path}\n  Flat dir: {flat_dir}\n  CSV entries: {len(splits)} | Files found: {len(file_lookup)}")
    with tqdm(total=total_files, desc="Loading from CSV splits", unit="file") as pbar:
        for target_split in ['test', 'val', 'train']:
            for fn, split in sorted(splits.items()):
                if split != target_split: continue
                if fn not in file_lookup:
                    csv_misses += 1; continue
                class_name, full_path = file_lookup[fn]
                if class_name not in labels:
                    labels[class_name] = idx; idx += 1
                try:
                    audio, _ = librosa.load(full_path, sr=TARGET_SR)
                    audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                        np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))
                    spec = compute_spec(audio, TARGET_SR)
                    if split == 'test':
                        X_test.append(spec); y_test.append(labels[class_name]); pbar.update(1)
                    elif split == 'val':
                        X_val.append(spec); y_val.append(labels[class_name]); pbar.update(1)
                    elif split == 'train':
                        X_train.append(spec); y_train.append(labels[class_name]); pbar.update(1)
                        if augmentation_mode == 'baseline':
                            aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                            X_train.append(compute_spec(aug_audio, TARGET_SR))
                            y_train.append(labels[class_name]); pbar.update(1)
                        elif augmentation_mode == 'specaugment':
                            X_train.append(augment_specaugment(spec))
                            y_train.append(labels[class_name]); pbar.update(1)
                except Exception as e:
                    failed_files.append(f"{full_path}: {str(e)}")
                    pbar.update(2 if split == 'train' and augmentation_mode in ['baseline', 'specaugment'] else 1)
                    continue
    X_test = np.array(X_test, dtype=np.float32); y_test = np.array(y_test, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32); y_val = np.array(y_val, dtype=np.int32)
    X_train = np.array(X_train, dtype=np.float32); y_train = np.array(y_train, dtype=np.int32)
    if csv_misses > 0: print(f"\n⚠ Warning: {csv_misses} CSV entries had no matching file")
    if failed_files: print(f"\n⚠ Warning: {len(failed_files)} files failed to load")
    print(f"\n✓ CSV Split: test={len(X_test)} val={len(X_val)} train={len(X_train)} classes={len(labels)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)

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
        f0 = np.random.randint(0, freq_bins - f_mask) if f_mask < freq_bins else 0
        spec_aug[f0:f0 + f_mask, :, 0] = 0
    for _ in range(SPECAUGMENT_NUM_MASKS):
        t_mask = np.random.randint(0, SPECAUGMENT_TIME_MASK)
        t0 = np.random.randint(0, time_bins - t_mask) if t_mask < time_bins else 0
        spec_aug[:, t0:t0 + t_mask, 0] = 0
    return spec_aug

# --------------------------------------------------------------
# NANO DEPTHWISE-CNN ARCHITECTURE
# --------------------------------------------------------------
def create_nanodepthwise_cnn(num_classes, input_shape, dropout=0.2):
    """
    NanoDepthwise-CNN adapted for 64x300 input.
    Designed for ultra-low memory footprint on Cortex-M7.
    Architecture: 4 stages of depthwise separable blocks with no skip connections.
    Channel Ramp: 32 -> 64 -> 128 -> 192
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv: 32 filters
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name='init_conv')(inputs)
    x = layers.BatchNormalization(name='init_bn')(x)
    x = layers.Activation('relu', name='init_relu')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)  # 16 x 75
    
    # Stage 1: 32 -> 64
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)  # 8 x 37
    
    # Stage 2: 64 -> 128
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)  # 4 x 18
    
    # Stage 3: 128 -> 192
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(192, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Global Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier Head
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='softmax')(x)
    
    return keras.Model(inputs, outputs, name="NanoDepthwise_CNN")

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
    with open(path, 'wb') as f: f.write(tflite_model)
    print(f"✓ Saved INT8 TFLite: {path} ({os.path.getsize(path) / 1024:.1f} KB)")

def _save_classification_report(y_test, y_pred, class_names, output_dir, model_type):
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, f'classification_report_{model_type.lower()}.txt')
    with open(report_path, 'w') as f:
        f.write(f"{model_type} Model Classification Report\n" + "=" * 70 + "\n" + report)
    print(f"✓ Saved classification report: {report_path}\n{model_type} Classification Report:\n{report}")
    return report_path

def _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    cmap = 'Blues' if 'FP32' in model_type else 'Greens'
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.title(f'{model_type} Confusion Matrix - Accuracy: {accuracy:.2f}%', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12); plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_history(warmup_hist, finetune_hist, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(warmup_hist.history['accuracy'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[0].plot(warmup_hist.history['val_accuracy'], label='Val (Warmup)', color='blue', linestyle='--')
    offset = len(warmup_hist.history['accuracy'])
    epochs_finetune = range(offset, offset + len(finetune_hist.history['accuracy']))
    axes[0].plot(epochs_finetune, finetune_hist.history['accuracy'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[0].plot(epochs_finetune, finetune_hist.history['val_accuracy'], label='Val (Finetune)', color='red', linestyle='--')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].set_title('Training Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(warmup_hist.history['loss'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[1].plot(warmup_hist.history['val_loss'], label='Val (Warmup)', color='blue', linestyle='--')
    axes[1].plot(epochs_finetune, finetune_hist.history['loss'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[1].plot(epochs_finetune, finetune_hist.history['val_loss'], label='Val (Finetune)', color='red', linestyle='--')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].set_title('Training Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
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

class MixupDataGenerator(keras.utils.Sequence):
    """Custom data generator for mixup augmentation"""
    def __init__(self, X, y, batch_size, alpha=0.2, num_classes=10):
        self.X, self.y, self.batch_size, self.alpha, self.num_classes = X, y, batch_size, alpha, num_classes
        self.indices = np.arange(len(X))
    def __len__(self): return int(np.ceil(len(self.X) / self.batch_size))
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices]; y_batch = self.y[batch_indices]
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
    def on_epoch_end(self): np.random.shuffle(self.indices)

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    config = get_config()
    logger = TrainingLogger(config['output_dir'])
    print(f"\nModel 1m: NanoDepthwise-CNN @ 64×300 (NATIVE)")
    print(f"  Hypothesis: Depthwise Separable > Fire/Mobile blocks for Multi-model SRAM efficiency")
    print(f"  Random seed: {config['random_seed']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Augmentation: {config['augmentation_mode']}")
    logger.log_hyperparameters(config)
    print("Loading dataset from CSV splits...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels, failed_count = load_data_from_csv(
        config['splits_csv'], config['flat_dir'], gmin=None, gmax=None,
        augmentation_mode=config['augmentation_mode'], time_shift_ms=config['time_shift_ms'],
        pitch_shift_steps=config['pitch_shift_steps'], mixup_alpha=config['mixup_alpha']
    )
    class_names = list(class_labels.keys())
    num_classes = len(class_names)
    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    logger.log_dataset_info(X_all, y_all, class_labels, X_train, X_val, X_test, failed_count)
    if len(X_all) == 0:
        print("\n✗ CRITICAL: Dataset is empty. Aborting to prevent training crashes.")
        print("  Please verify CSV filenames match the .wav files in your flat directory.")
        sys.exit(1)
    X_calib = X_val[:config['calib_samples']]
    print(f"\n✓ Calibration set: {len(X_calib)} samples")
    print(f"\n{'=' * 70}\nCREATING NANO DEPTHWISE-CNN (64×300 NATIVE)\n{'=' * 70}")
    model = create_nanodepthwise_cnn(num_classes, config['input_shape'], config['dropout'])
    model.summary()
    logger.log_model_info(model)
    if config['augmentation_mode'] == 'mixup':
        train_generator = MixupDataGenerator(X_train, y_train, config['batch_size'],
                                             alpha=config['mixup_alpha'], num_classes=num_classes)
        val_data = (X_val, keras.utils.to_categorical(y_val, num_classes))
        loss_function = 'categorical_crossentropy'
    else:
        train_generator = None; val_data = (X_val, y_val); loss_function = 'sparse_categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=config['warmup_lr']), loss=loss_function, metrics=['accuracy'])
    logger.start_stage("STAGE 1: WARMUP TRAINING")
    print(f"\n{'=' * 70}\nSTAGE 1: WARMUP TRAINING ({config['warmup_epochs']} epochs)\n{'=' * 70}")
    warmup_checkpoint = os.path.join(config['output_dir'], 'warmup_best.weights.h5')
    warmup_callbacks = [
        callbacks.ModelCheckpoint(warmup_checkpoint, monitor='val_accuracy', save_best_only=True,
                                  save_weights_only=True, mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    ]
    if config['lr_schedule'] in ['cosine', 'both']:
        warmup_callbacks.append(callbacks.LearningRateScheduler(
            lambda epoch: config['warmup_lr'] * 0.5 * (1 + np.cos(np.pi * epoch / config['warmup_epochs'])), verbose=0))
    if config['lr_schedule'] in ['plateau', 'both']:
        warmup_callbacks.append(callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1))
    if config['augmentation_mode'] == 'mixup':
        warmup_history = model.fit(train_generator, validation_data=val_data, epochs=config['warmup_epochs'],
                                   callbacks=warmup_callbacks, verbose=1)
    else:
        warmup_history = model.fit(X_train, y_train, validation_data=val_data, epochs=config['warmup_epochs'],
                                   batch_size=config['batch_size'], callbacks=warmup_callbacks, verbose=1)
    logger.end_stage("STAGE 1: WARMUP TRAINING", warmup_history)
    logger.start_stage("STAGE 2: FINE-TUNING")
    print(f"\n{'=' * 70}\nSTAGE 2: FINE-TUNING ({config['finetune_epochs']} epochs)\n{'=' * 70}")
    model.compile(optimizer=Adam(learning_rate=config['finetune_lr']), loss=loss_function, metrics=['accuracy'])
    finetune_checkpoint = os.path.join(config['output_dir'], 'finetune_best.weights.h5')
    finetune_callbacks = [
        callbacks.ModelCheckpoint(finetune_checkpoint, monitor='val_accuracy', save_best_only=True,
                                  save_weights_only=True, mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    ]
    if config['lr_schedule'] in ['plateau', 'both']:
        finetune_callbacks.append(callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1))
    if config['augmentation_mode'] == 'mixup':
        finetune_history = model.fit(train_generator, validation_data=val_data, epochs=config['finetune_epochs'],
                                     callbacks=finetune_callbacks, verbose=1)
    else:
        finetune_history = model.fit(X_train, y_train, validation_data=val_data, epochs=config['finetune_epochs'],
                                     batch_size=config['batch_size'], callbacks=finetune_callbacks, verbose=1)
    logger.end_stage("STAGE 2: FINE-TUNING", finetune_history)
    plot_training_history(warmup_history, finetune_history, config['output_dir'])
    fp32_path = os.path.join(config['output_dir'], 'nanodepthwise_fp32.keras')
    model.save(fp32_path); print(f"✓ Saved FP32 model: {fp32_path}")
    logger.start_stage("EVALUATION: FP32 (.keras)")
    print(f"\n{'=' * 70}\nEVALUATING FP32 MODEL\n{'=' * 70}")
    fp32_acc = evaluate_model(model, X_test, y_test, class_names, config['output_dir'], "FP32")
    logger.log_evaluation("FP32 (.keras)", fp32_acc, os.path.join(config['output_dir'], 'classification_report_fp32.txt'))
    logger.start_stage("TFLITE CONVERSION (PTQ)")
    print(f"\n{'=' * 70}\nCONVERTING TO INT8 TFLITE\n{'=' * 70}")
    int8_path = os.path.join(config['output_dir'], 'nanodepthwise_int8.tflite')
    convert_to_tflite_int8(model, X_calib, int8_path)
    logger.start_stage("EVALUATION: INT8 TFLite")
    print(f"\n{'=' * 70}\nEVALUATING INT8 TFLITE\n{'=' * 70}")
    int8_acc = evaluate_tflite(int8_path, X_test, y_test, class_names, config['output_dir'])
    logger.log_evaluation("INT8 TFLite", int8_acc, os.path.join(config['output_dir'], 'classification_report_int8.txt'))
    model_sizes = {"FP32 (.keras)": f"{os.path.getsize(fp32_path) / (1024 ** 2):.2f} MB",
                   "INT8 (.tflite)": f"{os.path.getsize(int8_path) / 1024:.1f} KB"}
    logger.log_final_results(fp32_acc, int8_acc, model_sizes, warmup_history, finetune_history, config)
    drop = fp32_acc - int8_acc
    total_time = time.time() - script_start
    print(f"\n{'=' * 70}\nFINAL RESULTS\n{'=' * 70}")
    print(f"Model:                   1m_nanodepthwise (64×300 NATIVE)")
    print(f"Augmentation:            {config['augmentation_mode']}")
    print(f"FP32 Accuracy:           {fp32_acc:6.2f}%")
    print(f"INT8 Accuracy:           {int8_acc:6.2f}%")
    print(f"Accuracy Drop:           {drop:6.2f}%")
    print(f"Total Time:              {format_time(total_time)}")
    print(f"\n{'=' * 70}")
    print(f"✓ Complete training report: {logger.log_path}")
    print(f"✓ All results saved to: {config['output_dir']}/")

if __name__ == "__main__":
    main()
    total_script_time = time.time() - script_start
    print(f"\n{'=' * 70}\nScript completed in: {format_time(total_script_time)}\n{'=' * 70}")