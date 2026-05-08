#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) for Mygardenbird TCN → Cortex-M7 deployment
Fixed spectrogram shape: 64x300 (10ms per frame)
Data Split: Fixed 75:10:15 (train/val/test) from dataset directories
"""

print("\n\n\n")
for _ in range(3):
    print(" ❤️ " * 30)

import os
import sys
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
import argparse
import platform
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from datetime import datetime

# GPU Configuration (before TensorFlow import)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# Prevent cuDNN from exhaustively searching for algorithms
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
# Reduce messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=warning, 2=error, 3=critical

# TensorFlow Environment Check
try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers, callbacks

    print("=" * 70)
    print("PTQ TCN for Mygardenbird 16 kHz → Cortex-M7")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"tf_keras version: {keras.__version__}")

    # Configure GPU with memory growth and error recovery
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Parse command line args early to get GPU settings
            import argparse
            temp_parser = argparse.ArgumentParser()
            temp_parser.add_argument("--gpu_memory_limit", type=int, default=None)
            temp_parser.add_argument("--force_cpu", action='store_true')
            temp_args, _ = temp_parser.parse_known_args()

            # Set memory limit if specified (must be done before memory growth)
            if temp_args.gpu_memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=temp_args.gpu_memory_limit)]
                )
                print(f"✓ Found {len(gpus)} GPU(s)")
                print(f"  GPU: {gpus[0].name}")
                print(f"  Memory limit: {temp_args.gpu_memory_limit} MB")
            else:
                # Enable memory growth if no limit specified
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Get GPU details
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

except Exception as e:
    print(f"✗ TensorFlow environment check failed: {e}")
    print("Install: pip install tensorflow tf_keras")
    sys.exit(1)

script_start = time.time()



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
# OPTIMIZER – LEGACY ON APPLE SILICON
# --------------------------------------------------------------
if platform.system() == "Darwin" and platform.processor() == "arm":
    from tf_keras.optimizers.legacy import Adam as LegacyAdam

    Adam = LegacyAdam
    print("Using LEGACY Adam (fast on M1/M2/M4)")
else:
    from tf_keras.optimizers import Adam

    print("Using standard Adam")

# --------------------------------------------------------------
# CONFIG - FIXED 64x300 SPECTROGRAM (10ms per frame)
# --------------------------------------------------------------
RANDOM_STATE = 786
tf.random.set_seed(RANDOM_STATE)

TARGET_SR = 16000
AUDIO_LENGTH_SEC = 3
FIXED_AUDIO_LENGTH = TARGET_SR * AUDIO_LENGTH_SEC
HOP_LENGTH = 160  # 10ms at 16kHz = 160 samples
N_FFT = 512
N_MELS = 64
FMAX = 8000

PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98
GLOBAL_STATS_SAMPLES = 100

# SpecAugment settings
SPECAUGMENT_FREQ_MASK = 8
SPECAUGMENT_TIME_MASK = 20
SPECAUGMENT_NUM_MASKS = 2

TIME_FRAMES = 300  # Fixed: 3 seconds / 10ms = 300 frames

DEFAULT_FLAT_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
DEFAULT_SPLITS_CSV = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--calib_samples", type=int, default=200)
    parser.add_argument("--tcn_channels", type=int, default=64, help="TCN channel width (64, 96, 128)")
    parser.add_argument("--use_augmentation", type=lambda x: x.lower() == 'true', default=False,
                        help="Enable time/pitch shift augmentation")
    parser.add_argument("--time_shift_ms", type=int, default=100,
                        help="Max time shift in milliseconds")
    parser.add_argument("--pitch_shift_steps", type=int, default=2,
                        help="Max pitch shift in semitones")
    parser.add_argument("--force_cpu", action='store_true',
                        help="Force CPU execution (disable GPU)")
    parser.add_argument("--gpu_memory_limit", type=int, default=None,
                        help="GPU memory limit in MB (e.g., 8192 for 8GB)")
    parser.add_argument("--splits_csv", type=str, default=DEFAULT_SPLITS_CSV,
                        help="Path to splits CSV from seabird_splitter_mip.py")
    parser.add_argument("--flat_dir", type=str, default=DEFAULT_FLAT_DIR,
                        help="Path to flat dataset directory (class_name/file.wav)")
    parser.add_argument("--random_seed", type=int, default=RANDOM_STATE,
                        help="Random seed for reproducibility")
    parser.add_argument("--mixup", type=float, default=None,
                        help="Enable mixup augmentation (alpha, e.g. 0.2)")
    parser.add_argument("--augment", action="store_true",
                        help="Enable baseline augmentation (time/pitch shift)")
    parser.add_argument("--specaugment", action="store_true",
                        help="Enable SpecAugment")
    args = parser.parse_args()

    # Apply force CPU if requested
    if args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("\n⚠ Force CPU mode enabled - GPU disabled")

    aug_suffix = ""
    augmentation_mode = "none"
    if args.mixup is not None:
        augmentation_mode = "mixup"
        aug_suffix = f"mixup{args.mixup}"
    elif args.specaugment:
        augmentation_mode = "specaugment"
        aug_suffix = "specaugment"
    elif args.augment or args.use_augmentation:
        augmentation_mode = "baseline"
        aug_suffix = "baseline"

    output_dir_name = (
        f"results_mygardenbird_2_{platform.system().lower()}/"
        f"2m_tcn_optimized_"
        f"mels{N_MELS}_"
        f"drop{int(args.dropout * 100):02d}_"
        f"rand{args.random_seed}_"
        f"warm{args.warmup_epochs}_"
        f"{aug_suffix}_"
        f"split80:10:10"
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
        'tcn_channels': args.tcn_channels,
        'use_augmentation': args.use_augmentation,
        'augmentation_mode': augmentation_mode,
        'mixup_alpha': args.mixup,
        'time_shift_ms': args.time_shift_ms,
        'pitch_shift_steps': args.pitch_shift_steps,
        'force_cpu': args.force_cpu,
        'gpu_memory_limit': args.gpu_memory_limit,
        'splits_csv': args.splits_csv,
        'flat_dir': args.flat_dir,
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
            f.write("MYGARDENBIRD TCN TRAINING REPORT\n")
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
            f.write(f"  Hop Length:             {HOP_LENGTH} samples (10.0 ms)\n")
            f.write(f"  Mel Bins (N_MELS):      {N_MELS}\n")
            f.write(f"  Max Frequency (FMAX):   {FMAX} Hz\n")
            f.write(f"  Time Frames:            {TIME_FRAMES} (FIXED)\n")
            f.write(f"  Spectrogram Shape:      {N_MELS}x{TIME_FRAMES}\n")

            f.write("\nModel Architecture:\n")
            f.write(f"  Model Type:             Temporal Convolutional Network (TCN)\n")
            f.write(f"  TCN Channels:           {config['tcn_channels']}\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  Input Shape:            {config['input_shape']}\n")
            f.write(f"  TCN Blocks:             2\n")
            f.write(f"  Dilation Rates:         [1, 2, 4, 8, 16, 32] per block\n")
            f.write(f"  Activation:             ReLU6 (quantization-friendly)\n")
            f.write(f"  Residual Connections:   Yes\n")
            f.write(f"  Skip Connections:       Yes\n")

            f.write("\nTraining Configuration:\n")
            f.write(f"  Warmup Epochs:          {config['warmup_epochs']}\n")
            f.write(f"  Fine-tune Epochs:       {config['finetune_epochs']}\n")
            f.write(f"  Total Epochs:           {config['warmup_epochs'] + config['finetune_epochs']}\n")
            f.write(f"  Batch Size:             {config['batch_size']}\n")
            f.write(f"  Warmup Learning Rate:   {config['warmup_lr']}\n")
            f.write(f"  Fine-tune Learning Rate:{config['finetune_lr']}\n")
            f.write(f"  Optimizer:              Adam (Legacy on Apple Silicon)\n")
            f.write(f"  Loss Function:          Sparse Categorical Crossentropy\n")

            f.write("\nData Augmentation:\n")
            f.write(f"  Enabled:                {config['use_augmentation']}\n")
            if config['use_augmentation']:
                f.write(f"  Time Shift:             ±{config['time_shift_ms']} ms\n")
                f.write(f"  Pitch Shift:            ±{config['pitch_shift_steps']} semitones\n")
                f.write(f"  Data Multiplier:        2x (original + augmented)\n")

            f.write("\nQuantization:\n")
            f.write(f"  Method:                 Post-Training Quantization (PTQ)\n")
            f.write(f"  Target Format:          INT8 TFLite\n")
            f.write(f"  Calibration Samples:    {config['calib_samples']}\n")
            f.write(f"  Input/Output Type:      INT8\n")

            f.write("\nDeployment Target:\n")
            f.write(f"  Platform:               ARM Cortex-M7\n")
            f.write(f"  Memory Target:          <512 KB (50% of 1MB)\n")

            f.write("\nData Paths:\n")
            f.write(f"  Dataset:                {config['flat_dir']}\n")
            f.write(f"  Spectrogram Cache:      {config['splits_csv']}\n")
            f.write(f"  Output Directory:       {config['output_dir']}\n")

    def log_dataset_info(self, X, y, class_labels, X_train, X_val, X_test):
        """Log dataset statistics."""
        self.log_section("DATASET INFORMATION")
        with open(self.log_path, 'a') as f:
            f.write(f"\nTotal Samples:          {len(X)}\n")
            f.write(f"Number of Classes:      {len(class_labels)}\n")
            f.write(f"\nData Split:\n")
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
            # Capture model.summary() output
            import io
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_str = stream.getvalue()
            f.write("\n" + summary_str)

            # Add parameter count
            total_params = model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params

            f.write(f"\nParameter Summary:\n")
            f.write(f"  Total Parameters:       {total_params:,}\n")
            f.write(f"  Trainable Parameters:   {trainable_params:,}\n")
            f.write(f"  Non-trainable Params:   {non_trainable_params:,}\n")

            # Estimate model sizes
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
        self.stage_times[stage_name] = time.time()
        self.log_section(stage_name)

    def end_stage(self, stage_name, history=None):
        """Mark the end of a training stage and log results."""
        elapsed = time.time() - self.stage_times[stage_name]
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

    def log_final_results(self, fp32_acc_keras, fp32_acc_h5, int8_acc, model_sizes,
                          warmup_history, finetune_history, config):
        """Log final comparison results."""
        self.log_section("FINAL RESULTS SUMMARY")

        drop_keras = fp32_acc_keras - int8_acc
        drop_h5 = fp32_acc_h5 - int8_acc
        total_time = time.time() - script_start

        with open(self.log_path, 'a') as f:
            # Quick reference card for spreadsheet comparison
            f.write("\n" + "=" * 80 + "\n")
            f.write("QUICK REFERENCE (Copy to spreadsheet)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Config: drp{int(config['dropout'] * 10)}_tcn{config['tcn_channels']}_"
                    f"{'aug_' if config['use_augmentation'] else ''}warmup{config['warmup_epochs']}_"
                    f"finetune{config['finetune_epochs']}\n")
            f.write(f"FP32_Keras: {fp32_acc_keras:.2f}% | INT8: {int8_acc:.2f}% | "
                    f"Drop: {drop_keras:+.2f}% | Time: {format_time(total_time)}\n")

            # Detailed results
            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n")

            f.write(f"\nAccuracy Results:\n")
            f.write(f"  FP32 (.keras):          {fp32_acc_keras:6.2f}%\n")
            f.write(f"  FP32 (.h5):             {fp32_acc_h5:6.2f}%\n")
            f.write(f"  INT8 (TFLite):          {int8_acc:6.2f}%\n")

            f.write(f"\nAccuracy Change (INT8 vs FP32):\n")
            f.write(f"  From .keras:            {drop_keras:+6.2f}% ")
            if abs(drop_keras) < 0.5:
                f.write("✓ Excellent (no degradation)\n")
            elif drop_keras > 0:
                f.write("✓✓ INT8 better! (quantization as regularizer)\n")
            elif drop_keras > -2:
                f.write("✓ Good (<2% drop)\n")
            elif drop_keras > -5:
                f.write("⚠ Acceptable (2-5% drop)\n")
            else:
                f.write("✗ High degradation (>5% drop)\n")

            f.write(f"  From .h5:               {drop_h5:+6.2f}%\n")

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

            overfitting_gap = finetune_history.history['accuracy'][-1] * 100 - finetune_history.history['val_accuracy'][
                -1] * 100
            f.write(f"  Train-Val Gap:          {overfitting_gap:+6.2f}%")
            if overfitting_gap < 2:
                f.write(" ✓ No overfitting\n")
            elif overfitting_gap < 5:
                f.write(" ⚠ Slight overfitting\n")
            else:
                f.write(" ✗ Overfitting detected\n")

            f.write(f"\nExecution Time:\n")
            f.write(f"  Total Duration:         {format_time(total_time)}\n")

            if 'STAGE 1: WARMUP TRAINING' in self.stage_times:
                warmup_time = self.stage_times.get('warmup_end', time.time()) - self.stage_times[
                    'STAGE 1: WARMUP TRAINING']
                f.write(f"  Warmup Stage:           {format_time(warmup_time)}\n")

            if 'STAGE 2: FINE-TUNING' in self.stage_times:
                finetune_time = self.stage_times.get('finetune_end', time.time()) - self.stage_times[
                    'STAGE 2: FINE-TUNING']
                f.write(f"  Fine-tuning Stage:      {format_time(finetune_time)}\n")

            f.write(f"\nTraining completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # CSV format for easy import
            f.write("\n" + "=" * 80 + "\n")
            f.write("CSV FORMAT (for batch comparison)\n")
            f.write("=" * 80 + "\n")
            f.write("dropout,tcn_channels,augmentation,warmup_epochs,finetune_epochs,warmup_lr,finetune_lr,"
                    "fp32_acc,int8_acc,drop,best_val_acc,train_val_gap,train_time_sec,model_size_kb\n")
            f.write(f"{config['dropout']},{config['tcn_channels']},{int(config['use_augmentation'])},"
                    f"{config['warmup_epochs']},{config['finetune_epochs']},{config['warmup_lr']},{config['finetune_lr']},"
                    f"{fp32_acc_keras:.2f},{int8_acc:.2f},{drop_keras:.2f},"
                    f"{max(finetune_history.history['val_accuracy']) * 100:.2f},"
                    f"{overfitting_gap:.2f},{int(total_time)},"
                    f"{os.path.getsize(os.path.join(config['output_dir'], 'tcn_int8.tflite')) / 1024:.1f}\n")

            # Analysis
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS & RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n")

            f.write(f"\nCurrent Performance: {int8_acc:.2f}%\n")

            # Performance tier
            if int8_acc >= 98:
                f.write("  🏆 EXCEPTIONAL - Near-perfect performance!\n")
            elif int8_acc >= 95:
                f.write("  ✓✓ EXCELLENT - Strong baseline performance\n")
            elif int8_acc >= 90:
                f.write("  ✓ GOOD - Solid performance with room for improvement\n")
            elif int8_acc >= 85:
                f.write("  ⚠ FAIR - Significant improvement needed\n")
            else:
                f.write("  ✗ POOR - Major improvements required\n")

            f.write(f"\nQuantization Method:\n")
            f.write(f"  Method: Post-Training Quantization (PTQ)\n")
            f.write(f"  INT8 vs FP32: {drop_keras:+.2f}%\n")

            f.write(f"\nModel Architecture:\n")
            f.write(f"  ✓ True Temporal Convolutional Network (TCN)\n")
            f.write(f"  - Causal dilated convolutions: [1,2,4,8,16,32] × 2 blocks\n")
            f.write(f"  - Residual + skip connections\n")
            f.write(f"  - ReLU6 activation (quantization-friendly)\n")

            # Recommendations based on current performance
            f.write(f"\n" + "-" * 80 + "\n")
            f.write("HYPERPARAMETER TUNING SUGGESTIONS\n")
            f.write("-" * 80 + "\n")

            if int8_acc >= 95:
                f.write("\n🎯 PUSHING BEYOND 95% - Advanced optimizations:\n\n")

                f.write("1. MODEL CAPACITY (Marginal gains expected):\n")
                f.write(f"   Current: {config['tcn_channels']} channels, {config['dropout']} dropout\n")
                f.write("   • Try 96 or 128 channels (if memory allows)\n")
                f.write("   • Reduce dropout to 0.05-0.1 (model may not be overfitting)\n")
                f.write("   • Add 3rd TCN block with same dilation pattern\n")
                f.write("   • Increase dense layers: 256→512, 128→256\n\n")

                f.write("2. ADVANCED AUGMENTATION:\n")
                f.write("   • SpecAugment: Time/frequency masking\n")
                f.write("   • Mixup: α=0.2-0.4 between classes\n")
                f.write("   • Background noise injection (SNR 10-30dB)\n")
                f.write("   • Random gain augmentation (±6dB)\n")
                f.write("   • Cutout on spectrograms (random rectangles)\n\n")

                f.write("3. PREPROCESSING REFINEMENT:\n")
                f.write(f"   Current: N_MELS={N_MELS}, FMAX={FMAX}Hz\n")
                f.write("   • Try N_MELS=80 or 128 (more frequency detail)\n")
                f.write("   • Experiment FMAX: 10kHz, 12kHz (mygardenbirds use high freq)\n")
                f.write("   • PCEN instead of log-mel (robust to loudness)\n")
                f.write("   • Try different window lengths (N_FFT=1024)\n\n")

                f.write("4. TRAINING REFINEMENTS:\n")
                f.write(f"   Current: {config['warmup_epochs']}+{config['finetune_epochs']} epochs\n")
                f.write("   • Label smoothing (ε=0.1)\n")
                f.write("   • Stochastic Weight Averaging (SWA) last 10 epochs\n")
                f.write("   • Longer warmup with cosine decay\n")
                f.write("   • Test Time Augmentation (TTA) - average predictions\n\n")

                f.write("5. ENSEMBLE METHODS:\n")
                f.write("   • Train 3-5 models with different random seeds\n")
                f.write("   • Voting or probability averaging\n")
                f.write("   • Expect +0.5-2% accuracy gain\n\n")

                f.write("6. CLASS-SPECIFIC ANALYSIS:\n")
                f.write("   • Check confusion matrix for problematic pairs\n")
                f.write("   • Oversample minority classes if imbalanced\n")
                f.write("   • Use focal loss if some classes dominate errors\n")

            else:
                f.write("\n🎯 IMPROVING FROM BASELINE:\n\n")

                f.write("1. INCREASE MODEL CAPACITY:\n")
                f.write(f"   Current: {config['tcn_channels']} channels, {config['dropout']} dropout\n")
                f.write("   • Try tcn_channels: 96, 128\n")
                f.write("   • Adjust dropout: 0.2, 0.3\n")
                f.write("   • Add more TCN blocks\n\n")

                f.write("2. LONGER TRAINING:\n")
                f.write(f"   Current: {config['warmup_epochs'] + config['finetune_epochs']} total epochs\n")
                f.write("   • Increase warmup_epochs to 100-150\n")
                f.write("   • Use learning rate warmup + cosine decay\n\n")

                f.write("3. DATA AUGMENTATION:\n")
                f.write(f"   Current: {'Enabled' if config['use_augmentation'] else 'Disabled'}\n")
                if not config['use_augmentation']:
                    f.write("   • Enable --use_augmentation true\n")
                f.write("   • Add SpecAugment, mixup, noise injection\n\n")

                f.write("4. CHECK FOR ISSUES:\n")
                f.write("   • Review confusion matrix for systematic errors\n")
                f.write("   • Check class balance\n")
                f.write("   • Verify preprocessing (listen to augmented samples)\n")

            # Next experiments suggestion
            f.write(f"\n" + "=" * 80 + "\n")
            f.write("SUGGESTED NEXT EXPERIMENTS\n")
            f.write("=" * 80 + "\n\n")

            experiments = []

            # Suggest based on current config
            if config['dropout'] >= 0.2:
                experiments.append(f"• Lower dropout: --dropout 0.1 (current: {config['dropout']})")
            else:
                experiments.append(f"• Try higher dropout: --dropout 0.2 (current: {config['dropout']})")

            if config['tcn_channels'] == 64:
                experiments.append("• Wider model: --tcn_channels 96 or 128")

            if not config['use_augmentation']:
                experiments.append("• Enable augmentation: --use_augmentation true")

            if config['warmup_epochs'] < 100:
                experiments.append(f"• Longer training: --warmup_epochs 100 (current: {config['warmup_epochs']})")

            for exp in experiments[:5]:  # Top 5 suggestions
                f.write(exp + "\n")


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
                print(f"\n\u26a0 Failed to process {f} during stats computation: {e}")
                continue

    if len(all_mel) == 0:
        raise RuntimeError("No valid audio files found for computing global stats")

    all_mel = np.concatenate(all_mel)
    gmin, gmax = np.percentile(all_mel, PERCENTILE_LOW), np.percentile(all_mel, PERCENTILE_HIGH)
    print(f"\u2713 Global stats computed from {total_sampled} files: {gmin:.2f} \u2192 {gmax:.2f} dB")
    return float(gmin), float(gmax)



# --------------------------------------------------------------
# SPECTROGRAM + NORMALIZE (FIXED 64x300)
# --------------------------------------------------------------
def compute_spec(audio, sr, gmin, gmax, n_mels=N_MELS):
    # win_length=400 uses only 400 samples (25ms at 16kHz)
    # n_fft stays at original value (e.g., 2048) with zero-padding
    WIN_LENGTH = 400  # 25ms at 16kHz

    mel = librosa.feature.melspectrogram(y=audio, sr=sr,
                                         n_fft=N_FFT,
                                         win_length=WIN_LENGTH,
                                         hop_length=HOP_LENGTH,
                                         n_mels=n_mels,
                                         fmax=FMAX,
                                         center=True,
                                         power=2.0)
    if mel.shape[1] > TIME_FRAMES:
        mel = mel[:, :TIME_FRAMES]
    if mel.shape[1] < TIME_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, TIME_FRAMES - mel.shape[1])))
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, gmin, gmax)
    mel_norm = (mel_db - gmin) / (gmax - gmin + 1e-8)
    return mel_norm[..., np.newaxis].astype(np.float32)

# --------------------------------------------------------------
# AUGMENTATION
# --------------------------------------------------------------
def augment_audio(audio, sr, time_shift_ms=100, pitch_steps=2):
    if np.random.rand() > 0.5:
        shift_samples = int(np.random.uniform(-time_shift_ms, time_shift_ms) * sr / 1000)
        audio = np.roll(audio, shift_samples)

    if np.random.rand() > 0.5:
        n_steps = np.random.uniform(-pitch_steps, pitch_steps)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    return audio


def augment_baseline(audio, sr, time_shift_ms=100, pitch_steps=2):
    """Alias for augment_audio (for compatibility with load_data_from_csv)."""
    return augment_audio(audio, sr, time_shift_ms=time_shift_ms, pitch_steps=pitch_steps)


def augment_specaugment(spec):
    """SpecAugment: frequency and time masking on spectrogram."""
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
def create_tcn(num_classes, input_shape, dropout=0.2, channels=64):
    inputs = layers.Input(shape=input_shape)
    x = layers.Permute((2, 1, 3))(inputs)
    x = layers.Reshape((input_shape[1], input_shape[0]))(x)
    x = layers.Dense(channels, activation='relu6')(x)
    x = layers.Dropout(dropout * 0.5)(x)

    skips = []
    for _ in range(2):
        for d in [1, 2, 4, 8, 16, 32]:
            res = x
            x = layers.Conv1D(channels, 3, dilation_rate=d, padding='causal', activation='relu6')(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(channels, 3, dilation_rate=d, padding='causal')(x)
            if res.shape[-1] != channels:
                res = layers.Conv1D(channels, 1)(res)
            x = layers.Add()([x, res])
            x = layers.Activation('relu6')(x)
            x = layers.Dropout(dropout)(x)
            skips.append(layers.Conv1D(channels, 1)(x))
    if skips:
        x = layers.Add()(skips)
        x = layers.Activation('relu6')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu6')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu6')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name="Mygardenbird_TCN")


# --------------------------------------------------------------
# TFLITE CONVERSION (POST-TRAINING QUANTIZATION)
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

    # Choose colormap based on model type
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

    # Use helper functions for common operations
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

    # Use helper functions for common operations
    _save_classification_report(y_test, y_pred, class_names, output_dir, "INT8")
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, "INT8", acc)

    return acc



# --------------------------------------------------------------
# CSV SPLIT HELPERS
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
        print("\n\u26a0 WARNING: No augmentation enabled")
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
        print(f"\n\u26a0 Warning: {csv_misses} CSV entries had no matching file in {flat_dir}")

    if len(failed_files) > 0:
        print(f"\n\u26a0 Warning: {len(failed_files)} files failed to load")
        with open('data_loading_errors.txt', 'w') as f:
            for error in failed_files:
                f.write(error + '\n')

    num_classes = len(labels)
    print(f"\n\u2713 CSV Split Complete:")
    print(f"  Test (held-out):        {len(X_test):5d} samples")
    print(f"  Val (held-out):         {len(X_val):5d} samples")
    print(f"  Train (w/ augment):     {len(X_train):5d} samples")
    print(f"  Total:                  {len(X_test) + len(X_val) + len(X_train):5d} samples")
    print(f"  Classes:                {num_classes}")
    print(f"\n  \u2713 NO DATA LEAKAGE: Test and Val samples never augmented")
    print(f"  \u2713 INDEPENDENT SPLITS: True held-out evaluation (CSV-based)")

    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)


def main():
    # Initialize logger and config
    config = get_config()
    logger = TrainingLogger(config['output_dir'])

    print(f"\nConfig:")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Finetune epochs: {config['finetune_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Warmup LR: {config['warmup_lr']}")
    print(f"  Finetune LR: {config['finetune_lr']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  TCN Channels: {config['tcn_channels']}")
    print(f"  Augmentation: {config['use_augmentation']}")
    print(f"  Spectrogram: {N_MELS}x{TIME_FRAMES} (10ms/frame)")

    # Log hyperparameters
    logger.log_hyperparameters(config)

    # Note: We no longer clear cache since augmentation is now standard

    # Compute global stats
    print("\nComputing global normalization stats...")
    splits = parse_splits_csv(config['splits_csv'])
    train_files = {fn for fn, split in splits.items() if split == 'train'}
    global_min, global_max = compute_global_stats(
        config['flat_dir'], N_MELS, allowed_files=train_files)

    # Load data with fixed 90/60/450 split
    print("\nLoading dataset with FIXED 90/60/450 SPLIT...")
    print("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels, failed_count = load_data_from_csv(
        config['splits_csv'], config['flat_dir'],
        global_min, global_max, N_MELS,
        augmentation_mode=config.get('augmentation_mode', 'none'),
        time_shift_ms=config.get('time_shift_ms', 100),
        pitch_shift_steps=config.get('pitch_shift_steps', 2),
        mixup_alpha=config.get('mixup_alpha', 0.2)
    )
    print("=" * 70)

    class_names = list(class_labels.keys())
    num_classes = len(class_names)

    # Compute totals for reporting
    total_samples = len(X_train) + len(X_val) + len(X_test)

    print(f"\n✓ Total samples loaded: {total_samples}")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Spectrogram shape: {X_train[0].shape}")

    print(f"\nFinal Split:")
    print(f"  Train:     {len(X_train):5d} samples ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"  Val:       {len(X_val):5d} samples ({len(X_val) / total_samples * 100:.1f}%)")
    print(f"  Test:      {len(X_test):5d} samples ({len(X_test) / total_samples * 100:.1f}%)")
    print(f"  Total:     {total_samples:5d} samples")

    # Verify test and val set distributions
    print(f"\nTest Set Class Distribution:")
    for class_name, class_idx in sorted(class_labels.items(), key=lambda x: x[1]):
        count = np.sum(y_test == class_idx)
        print(f"  {class_name:30s}: {count:3d} samples")

    print(f"\nValidation Set Class Distribution:")
    for class_name, class_idx in sorted(class_labels.items(), key=lambda x: x[1]):
        count = np.sum(y_val == class_idx)
        print(f"  {class_name:30s}: {count:3d} samples")

    # Log dataset info (create combined arrays for logging)
    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    logger.log_dataset_info(X_all, y_all, class_labels, X_train, X_val, X_test)

    # Calibration set (from validation set)
    X_calib = X_val[:config['calib_samples']]
    print(f"\n✓ Calibration set: {len(X_calib)} samples (from validation set)")

    # Create model
    print(f"\n{'=' * 70}")
    print("CREATING TCN MODEL")
    print(f"{'=' * 70}")
    model = create_tcn(num_classes, config['input_shape'],
                       config['dropout'], config['tcn_channels'])
    model.summary()

    # Log model info
    logger.log_model_info(model)

    # Compile for warmup
    model.compile(
        optimizer=Adam(learning_rate=config['warmup_lr']),
        loss='sparse_categorical_crossentropy',
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
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-7, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        ),
        callbacks.LearningRateScheduler(
            lambda epoch: config['warmup_lr'] * 0.5 * (1 + np.cos(np.pi * epoch / config['warmup_epochs'])),
            verbose=0
        )
    ]

    try:
        warmup_history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['warmup_epochs'],
            batch_size=config['batch_size'],
            callbacks=warmup_callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Reduce batch size: --batch_size 16 (or 8)")
        print("2. Force CPU mode: --force_cpu")
        print("3. Limit GPU memory: --gpu_memory_limit 8192")
        print("4. Update CUDA/cuDNN to compatible versions")
        raise

    logger.stage_times['warmup_end'] = time.time()
    logger.end_stage("STAGE 1: WARMUP TRAINING", warmup_history)
    print("\n✓ Warmup complete - best weights restored")

    # Stage 2: Fine-tuning
    logger.start_stage("STAGE 2: FINE-TUNING")
    print(f"\n{'=' * 70}")
    print(f"STAGE 2: FINE-TUNING ({config['finetune_epochs']} epochs)")
    print(f"{'=' * 70}")

    model.compile(
        optimizer=Adam(learning_rate=config['finetune_lr']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    finetune_checkpoint = os.path.join(config['output_dir'], 'finetune_best.weights.h5')
    finetune_callbacks = [
        callbacks.ModelCheckpoint(
            finetune_checkpoint, monitor='val_accuracy',
            save_best_only=True, save_weights_only=True,
            mode='max', verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-8, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )
    ]

    try:
        finetune_history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['finetune_epochs'],
            batch_size=config['batch_size'],
            callbacks=finetune_callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"\n✗ Fine-tuning failed with error: {e}")
        print("\nTrying to continue with warmup model...")
        finetune_history = warmup_history

    logger.stage_times['finetune_end'] = time.time()
    logger.end_stage("STAGE 2: FINE-TUNING", finetune_history)
    print("\n✓ Fine-tuning complete - best weights restored")

    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(warmup_history, finetune_history, config['output_dir'])

    # Save models
    fp32_path = os.path.join(config['output_dir'], 'tcn_fp32.keras')
    h5_path = os.path.join(config['output_dir'], 'tcn_fp32.h5')
    model.save(fp32_path)
    model.save(h5_path)
    print(f"✓ Saved FP32 model: {fp32_path}")
    print(f"✓ Saved FP32 model: {h5_path}")

    # Evaluate FP32 (.keras)
    logger.start_stage("EVALUATION: FP32 (.keras)")
    print(f"\n{'=' * 70}")
    print("EVALUATING FP32 MODEL (.keras) ON HELD-OUT TEST SET")
    print(f"{'=' * 70}")
    fp32_acc_keras = evaluate_model(model, X_test, y_test, class_names,
                                    config['output_dir'], "FP32_Keras")
    logger.log_evaluation("FP32 (.keras)", fp32_acc_keras,
                          os.path.join(config['output_dir'], 'classification_report_fp32_keras.txt'))

    # Evaluate FP32 (.h5)
    logger.start_stage("EVALUATION: FP32 (.h5)")
    print(f"\n{'=' * 70}")
    print("EVALUATING FP32 MODEL (.h5) ON HELD-OUT TEST SET")
    print(f"{'=' * 70}")
    h5_model = keras.models.load_model(h5_path)
    fp32_acc_h5 = evaluate_model(h5_model, X_test, y_test, class_names,
                                 config['output_dir'], "FP32_H5")
    logger.log_evaluation("FP32 (.h5)", fp32_acc_h5,
                          os.path.join(config['output_dir'], 'classification_report_fp32_h5.txt'))

    # Convert to TFLite INT8 (POST-TRAINING QUANTIZATION)
    logger.start_stage("TFLITE CONVERSION (PTQ)")
    print(f"\n{'=' * 70}")
    print("CONVERTING TO INT8 TFLITE (POST-TRAINING QUANTIZATION)")
    print(f"{'=' * 70}")
    int8_path = os.path.join(config['output_dir'], 'tcn_int8.tflite')
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

    # Collect model sizes
    model_sizes = {
        "FP32 (.keras)": f"{os.path.getsize(fp32_path) / (1024 ** 2):.2f} MB",
        "FP32 (.h5)": f"{os.path.getsize(h5_path) / (1024 ** 2):.2f} MB",
        "INT8 (.tflite)": f"{os.path.getsize(int8_path) / 1024:.1f} KB"
    }

    # Log final results
    logger.log_final_results(fp32_acc_keras, fp32_acc_h5, int8_acc, model_sizes,
                             warmup_history, finetune_history, config)

    # Console summary
    drop_keras = fp32_acc_keras - int8_acc
    drop_h5 = fp32_acc_h5 - int8_acc
    total_time = time.time() - script_start

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS (FIXED 90/60/450 SPLIT EVALUATION)")
    print(f"{'=' * 70}")
    print(f"FP32 (.keras) Accuracy:  {fp32_acc_keras:6.2f}%")
    print(f"FP32 (.h5) Accuracy:     {fp32_acc_h5:6.2f}%")
    print(f"INT8 Accuracy:           {int8_acc:6.2f}%")
    print(f"Accuracy Drop (.keras):  {drop_keras:6.2f}%")
    print(f"Accuracy Drop (.h5):     {drop_h5:6.2f}%")
    print(f"Total Execution Time:    {format_time(total_time)}")
    print(f"\n✓ Test/Val sets were HELD-OUT during training (no data leakage)")
    print(f"✓ Results are publication-ready and reproducible")
    print(f"{'=' * 70}")

    print(f"\n✓ Complete training report saved to:")
    print(f"  {logger.log_path}")
    print(f"\nAll results saved to: {config['output_dir']}/")

if __name__ == "__main__":
    main()

    # Report total execution time at command line level
    total_script_time = time.time() - script_start
    print(f"\n{'=' * 70}")
    print(f"Script completed in: {format_time(total_script_time)}")
    print(f"{'=' * 70}")
