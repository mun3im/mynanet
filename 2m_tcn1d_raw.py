#!/usr/bin/env python3
"""
Model 2m - 1D TCN on raw waveform (ported from 4i_tcn1d_dual_stage)
- Input: raw audio (48000 samples, 1 channel) — no mel spectrogram
- Architecture: 6 causal dilated TCN blocks, dilations [1,2,4,8,16,32]
  filters 32 for blocks 0-3, 64 for blocks 4-5 (doubling after block 3)
- Peak-normalised waveform: audio / (max_abs + 1e-8)
- Note: RF = 2*(1+2+4+8+16+32)*3 = 378 samples = 23ms at 16kHz
  Very short for 3s clips — included as architectural baseline
"""

import os
import sys
import argparse
import platform
import warnings
import hashlib
import json

warnings.filterwarnings("ignore")

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_early_args():
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--force_cpu", action='store_true')
    temp_parser.add_argument("--gpu_memory_limit", type=int, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    if temp_args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return temp_args


early_args = parse_early_args()

print("\n" + "=" * 70)
print("ENVIRONMENT VALIDATION")
print("=" * 70)

try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers, callbacks

    print(f"✓ TensorFlow version: {tf.__version__}")
    print(f"✓ tf_keras version: {keras.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus and not early_args.force_cpu:
        try:
            if early_args.gpu_memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=early_args.gpu_memory_limit)]
                )
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"✓ Found {len(gpus)} GPU(s): {gpu_details.get('device_name', gpus[0].name)}")
        except RuntimeError as e:
            print(f"⚠ GPU config warning: {e}")
    else:
        print("✓ Running on CPU")
    print("=" * 70)
except Exception as e:
    print(f"\n✗ TensorFlow check failed: {e}")
    sys.exit(1)

import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

script_start = time.time()

# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------
DEFAULT_RANDOM_STATE = 786
TARGET_SR = 16000
AUDIO_LENGTH_SEC = 3
FIXED_AUDIO_LENGTH = TARGET_SR * AUDIO_LENGTH_SEC  # 48000

DEFAULT_FLAT_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
DEFAULT_SPLITS_CSV = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"


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


if platform.system() == "Darwin" and platform.processor() == "arm":
    from tf_keras.optimizers.legacy import Adam as LegacyAdam
    Adam = LegacyAdam
else:
    from tf_keras.optimizers import Adam


# --------------------------------------------------------------
# CONFIG
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
    parser.add_argument("--tcn_channels", type=int, default=32)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--specaugment", action='store_true')
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--force_cpu", action='store_true')
    parser.add_argument("--gpu_memory_limit", type=int, default=None)
    parser.add_argument("--splits_csv", type=str, default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--flat_dir", type=str, default=DEFAULT_FLAT_DIR)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "plateau", "both", "none"])
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--n_mels", type=int, default=64,
                        help="Ignored — 2m uses raw audio input")
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
        f"results_mygardenbird_2_{platform.system().lower()}/"
        f"2m_tcn1d_raw_"
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
        'input_shape': (FIXED_AUDIO_LENGTH, 1),
        'output_dir': output_dir_name,
        'calib_samples': args.calib_samples,
        'tcn_channels': args.tcn_channels,
        'augmentation_mode': augmentation_mode,
        'mixup_alpha': args.mixup,
        'force_cpu': args.force_cpu,
        'gpu_memory_limit': args.gpu_memory_limit,
        'splits_csv': args.splits_csv,
        'flat_dir': args.flat_dir,
        'lr_schedule': args.lr_schedule,
        'random_seed': args.random_seed,
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    return config


# --------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------
class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, 'training_report.txt')
        self.start_time = time.time()
        self.stage_times = {}
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MYGARDENBIRD TCN TRAINING REPORT (Model 2m - 1D TCN Raw Audio)\n")
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
                f.write(f"  GPU: {len(gpus)} device(s) detected\n")
                f.write(f"  GPU Memory: Dynamic growth enabled\n")
            else:
                f.write(f"  Compute: CPU only\n")

            f.write("\nAudio Processing:\n")
            f.write(f"  Target Sample Rate:     {TARGET_SR} Hz\n")
            f.write(f"  Audio Length:           {AUDIO_LENGTH_SEC} seconds ({FIXED_AUDIO_LENGTH} samples)\n")
            f.write(f"  Input Type:             Raw waveform (peak normalised)\n")
            f.write(f"  Input Shape:            {config['input_shape']}\n")
            f.write(f"  Normalisation:          peak / (|max| + 1e-8)\n")

            f.write("\nModel Architecture:\n")
            f.write(f"  Model Type:             1D TCN on raw audio (ported from 4i)\n")
            f.write(f"  TCN Channels:           {config['tcn_channels']} base (doubles to {config['tcn_channels']*2} at block 4)\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  TCN Blocks:             6, dilations [1,2,4,8,16,32] (causal)\n")
            f.write(f"  Receptive Field:        ~378 samples (~23ms at 16kHz)\n")
            f.write(f"  Normalization:          BatchNorm per block\n")

            f.write("\nTraining Configuration:\n")
            f.write(f"  Random Seed:            {config['random_seed']}\n")
            f.write(f"  Warmup Epochs:          {config['warmup_epochs']}\n")
            f.write(f"  Fine-tune Epochs:       {config['finetune_epochs']}\n")
            f.write(f"  Batch Size:             {config['batch_size']}\n")
            f.write(f"  Warmup LR:              {config['warmup_lr']}\n")
            f.write(f"  Fine-tune LR:           {config['finetune_lr']}\n")
            f.write(f"  LR Schedule:            {config['lr_schedule']}\n")

            f.write("\nData Augmentation:\n")
            f.write(f"  Mode:                   {config['augmentation_mode']}\n")
            if config['augmentation_mode'] == 'mixup':
                f.write(f"  Type:                   Mixup (alpha={config['mixup_alpha']})\n")

            f.write("\nDeployment Target:\n")
            f.write(f"  Platform:               ARM Cortex-M7\n")
            f.write(f"  Memory Target:          <512 KB\n")

            f.write("\nData Paths:\n")
            f.write(f"  Dataset:                {config['flat_dir']}\n")
            f.write(f"  Splits CSV:             {config['splits_csv']}\n")
            f.write(f"  Output Directory:       {config['output_dir']}\n")

    def log_dataset_info(self, X, y, class_labels, X_train, X_val, X_test, failed_files=0):
        self.log_section("DATASET INFORMATION")
        with open(self.log_path, 'a') as f:
            f.write(f"\nTotal Samples:          {len(X)}\n")
            f.write(f"Number of Classes:      {len(class_labels)}\n")
            if failed_files > 0:
                f.write(f"Failed Files:           {failed_files}\n")
            f.write(f"\nData Split:\n")
            f.write(f"  Training:             {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)\n")
            f.write(f"  Validation:           {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)\n")
            f.write(f"  Test:                 {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)\n")

    def log_model_info(self, model):
        self.log_section("MODEL ARCHITECTURE")
        with open(self.log_path, 'a') as f:
            import io
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            f.write("\n" + stream.getvalue())
            total_params = model.count_params()
            f.write(f"\nTotal Parameters:       {total_params:,}\n")
            f.write(f"  INT8 (1 byte/param):  {total_params/1024:.1f} KB\n")
            if total_params / 1024 > 512:
                f.write(f"  ⚠ WARNING: May exceed 512 KB\n")
            else:
                f.write(f"  ✓ Within 512 KB target\n")

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
                f.write(f"  Epochs: {len(history.history['loss'])}\n")
                f.write(f"  Best Val Acc: {max(history.history['val_accuracy'])*100:.2f}%\n")

    def log_evaluation(self, model_name, accuracy, report_path):
        with open(self.log_path, 'a') as f:
            f.write(f"\n{model_name}:\n")
            f.write(f"  Test Accuracy: {accuracy:.2f}%\n")
            f.write(f"  Report: {report_path}\n")

    def log_final_results(self, fp32_acc, int8_acc, model_sizes,
                          warmup_history, finetune_history, config):
        self.log_section("FINAL RESULTS SUMMARY")
        drop = fp32_acc - int8_acc
        total_time = time.time() - script_start
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("QUICK REFERENCE (Copy to spreadsheet)\n")
            f.write("=" * 80 + "\n")
            f.write(f"FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}% | "
                    f"Drop: {drop:+.2f}% | Time: {format_time(total_time)}\n")
            f.write(f"\nModel Sizes:\n")
            for k, v in model_sizes.items():
                f.write(f"  {k:20s}: {v}\n")
            f.write(f"\nModel Architecture:\n")
            f.write(f"  ✓ 1D TCN on raw audio (ported from 4i)\n")
            f.write(f"  - Causal dilated convolutions: [1,2,4,8,16,32]\n")
            f.write(f"  - Residual connections + BatchNorm\n")
            f.write(f"  - RF ≈ 378 samples (~23ms)\n")
            f.write(f"\nAugmentation: {config['augmentation_mode']}\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nCSV FORMAT\n")
            f.write("dropout,augmentation,warmup_epochs,fp32_acc,int8_acc,drop,train_time_sec\n")
            total_time_int = int(time.time() - script_start)
            f.write(f"{config['dropout']},{config['augmentation_mode']},"
                    f"{config['warmup_epochs']},{fp32_acc:.2f},{int8_acc:.2f},{drop:.2f},"
                    f"{total_time_int}\n")


# --------------------------------------------------------------
# RAW AUDIO LOADING
# --------------------------------------------------------------
def load_raw_audio(audio_path):
    """Load and peak-normalise raw audio → (48000, 1) float32."""
    audio, _ = librosa.load(audio_path, sr=TARGET_SR)
    if len(audio) > FIXED_AUDIO_LENGTH:
        audio = audio[:FIXED_AUDIO_LENGTH]
    else:
        audio = np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))
    peak = np.max(np.abs(audio)) + 1e-8
    audio = audio / peak
    return audio[:, np.newaxis].astype(np.float32)  # (48000, 1)


# --------------------------------------------------------------
# CSV SPLIT HELPERS
# --------------------------------------------------------------
def parse_splits_csv(csv_path):
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


def load_data_from_csv(csv_path, flat_dir, augmentation_mode='none', mixup_alpha=0.2):
    splits = parse_splits_csv(csv_path)

    X_test, y_test = [], []
    X_val, y_val = [], []
    X_train, y_train = [], []
    labels = {}
    idx = 0
    failed_files = []

    file_lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                file_lookup[f] = (class_name, os.path.join(class_dir, f))

    total_files = sum(1 for fn in splits if fn in file_lookup)
    print(f"\nDataset: CSV-based split, {len(splits)} entries, {len(file_lookup)} files found")
    print(f"Input: raw audio (48000 samples, peak-normalised)")

    with tqdm(total=total_files, desc="Loading raw audio", unit="file") as pbar:
        for target_split in ['test', 'val', 'train']:
            for fn, split in sorted(splits.items()):
                if split != target_split:
                    continue
                if fn not in file_lookup:
                    continue

                class_name, full_path = file_lookup[fn]
                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                try:
                    audio = load_raw_audio(full_path)
                    if split == 'test':
                        X_test.append(audio); y_test.append(labels[class_name])
                    elif split == 'val':
                        X_val.append(audio); y_val.append(labels[class_name])
                    elif split == 'train':
                        X_train.append(audio); y_train.append(labels[class_name])
                    pbar.update(1)
                except Exception as e:
                    failed_files.append(f"{full_path}: {e}")
                    pbar.update(1)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    print(f"\n✓ Loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    if failed_files:
        print(f"⚠ {len(failed_files)} files failed")

    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)


# --------------------------------------------------------------
# MIXUP GENERATOR
# --------------------------------------------------------------
class MixupDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2, num_classes=12):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.num_classes = num_classes
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx]
        y_batch = self.y[batch_idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(X_batch))
        X_mixed = lam * X_batch + (1 - lam) * X_batch[perm]
        y_a = keras.utils.to_categorical(y_batch, self.num_classes)
        y_b = keras.utils.to_categorical(y_batch[perm], self.num_classes)
        return X_mixed, lam * y_a + (1 - lam) * y_b

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# --------------------------------------------------------------
# MODEL — 1D TCN on raw audio (ported from 4i, functional API)
# --------------------------------------------------------------
def create_tcn(num_classes, input_shape, dropout=0.3, channels=32):
    """1D causal dilated TCN on raw waveform.

    Architecture (ported from 4i_tcn1d_dual_stage to functional API):
    - Initial Conv1D(channels, 7, causal) + BN
    - 6 residual TCN blocks, dilations [1,2,4,8,16,32]
      Blocks 0-3: channels filters; blocks 4-5: channels*2 filters
    - GlobalAveragePooling1D → Dense(128) → Dense(num_classes)
    - BatchNorm + SpatialDropout1D per block (quantization-friendly)

    Receptive field: 2 * (kernel-1) * sum(dilations) = 2*2*(1+2+4+8+16+32) = 252 samples
    At 16kHz: ~15.75ms — intentionally short; included as architectural baseline.
    """
    inputs = layers.Input(shape=input_shape, name='raw_audio')

    # Initial projection
    x = layers.Conv1D(channels, 7, padding='causal', name='init_conv')(inputs)
    x = layers.BatchNormalization(name='init_bn')(x)
    x = layers.Activation('relu6', name='init_act')(x)
    x = layers.SpatialDropout1D(dropout, name='init_drop')(x)

    # 6 dilated residual blocks
    filter_schedule = [channels] * 4 + [channels * 2] * 2
    dilations = [1, 2, 4, 8, 16, 32]

    for i, (ch, d) in enumerate(zip(filter_schedule, dilations)):
        res = x
        if res.shape[-1] != ch:
            res = layers.Conv1D(ch, 1, name=f'res_proj_{i}')(res)

        x = layers.Conv1D(ch, 3, dilation_rate=d, padding='causal',
                          name=f'dconv1_{i}')(x)
        x = layers.BatchNormalization(name=f'bn1_{i}')(x)
        x = layers.Activation('relu6', name=f'act1_{i}')(x)
        x = layers.SpatialDropout1D(dropout, name=f'drop1_{i}')(x)

        x = layers.Conv1D(ch, 3, dilation_rate=d, padding='causal',
                          name=f'dconv2_{i}')(x)
        x = layers.BatchNormalization(name=f'bn2_{i}')(x)
        x = layers.Activation('relu6', name=f'act2_{i}')(x)
        x = layers.SpatialDropout1D(dropout, name=f'drop2_{i}')(x)

        x = layers.Add(name=f'res_{i}')([x, res])
        x = layers.Activation('relu6', name=f'res_act_{i}')(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dropout(0.5, name='clf_drop1')(x)
    x = layers.Dense(128, activation='relu6', name='clf_dense')(x)
    x = layers.Dropout(0.5, name='clf_drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return keras.Model(inputs, outputs, name='MynaNet_TCN1D_Raw')


# --------------------------------------------------------------
# TFLITE CONVERSION
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
# EVALUATION
# --------------------------------------------------------------
def _save_classification_report(y_test, y_pred, class_names, output_dir, model_type):
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    path = os.path.join(output_dir, f'classification_report_{model_type.lower()}.txt')
    with open(path, 'w') as f:
        f.write(f"{model_type} Classification Report\n{'='*70}\n\n{report}")
    print(f"✓ Saved: {path}")
    return path


def _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    cmap = 'Blues' if 'FP32' in model_type else 'Greens'
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_type} — {accuracy:.2f}%', fontsize=14)
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type.lower()}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(wh, fh, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    offset = len(wh.history['accuracy'])
    fe = range(offset, offset + len(fh.history['accuracy']))
    for ax, key, title in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(wh.history[key], label='Train (warm)', color='blue', alpha=0.7)
        ax.plot(wh.history[f'val_{key}'], '--', color='blue')
        ax.plot(fe, fh.history[key], label='Train (fine)', color='red', alpha=0.7)
        ax.plot(fe, fh.history[f'val_{key}'], '--', color='red')
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()


def evaluate_model(model, X_test, y_test, class_names, output_dir, model_type="FP32"):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred) * 100
    _save_classification_report(y_test, y_pred, class_names, output_dir, model_type)
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, acc)
    return acc


def evaluate_tflite(tflite_path, X_test, y_test, class_names, output_dir):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']

    y_pred = []
    for i in tqdm(range(len(X_test)), desc="TFLite INT8 inference"):
        x = (X_test[i:i+1] / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        o = interp.get_tensor(out['index'])
        y_pred.append(np.argmax((o.astype(np.float32) - out_zp) * out_scale))

    y_pred = np.array(y_pred)
    acc = accuracy_score(y_test, y_pred) * 100
    _save_classification_report(y_test, y_pred, class_names, output_dir, "INT8")
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, "INT8", acc)
    return acc


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    config = get_config()
    logger = TrainingLogger(config['output_dir'])
    logger.log_hyperparameters(config)

    print(f"\nConfig: seed={config['random_seed']}, channels={config['tcn_channels']}, "
          f"dropout={config['dropout']}, aug={config['augmentation_mode']}")

    X_train, X_val, X_test, y_train, y_val, y_test, class_labels, failed = \
        load_data_from_csv(
            config['splits_csv'], config['flat_dir'],
            augmentation_mode=config['augmentation_mode'],
            mixup_alpha=config.get('mixup_alpha', 0.2)
        )

    class_names = list(class_labels.keys())
    num_classes = len(class_names)
    total = len(X_train) + len(X_val) + len(X_test)
    logger.log_dataset_info(
        np.concatenate([X_train, X_val, X_test]),
        np.concatenate([y_train, y_val, y_test]),
        class_labels, X_train, X_val, X_test, failed
    )

    X_calib = X_val[:config['calib_samples']]

    print(f"\n{'='*70}")
    print("CREATING 1D TCN RAW AUDIO MODEL")
    print(f"{'='*70}")
    model = create_tcn(num_classes, config['input_shape'],
                       config['dropout'], config['tcn_channels'])
    model.summary()
    logger.log_model_info(model)

    if config['augmentation_mode'] == 'mixup':
        train_gen = MixupDataGenerator(X_train, y_train, config['batch_size'],
                                       alpha=config['mixup_alpha'],
                                       num_classes=num_classes)
        val_data = (X_val, keras.utils.to_categorical(y_val, num_classes))
        loss_fn = 'categorical_crossentropy'
    else:
        train_gen = None
        val_data = (X_val, y_val)
        loss_fn = 'sparse_categorical_crossentropy'

    model.compile(optimizer=Adam(learning_rate=config['warmup_lr']),
                  loss=loss_fn, metrics=['accuracy'])

    # Stage 1: Warmup
    logger.start_stage("STAGE 1: WARMUP TRAINING")
    warmup_ckpt = os.path.join(config['output_dir'], 'warmup_best.weights.h5')
    wu_cbs = [
        callbacks.ModelCheckpoint(warmup_ckpt, monitor='val_accuracy',
                                  save_best_only=True, save_weights_only=True,
                                  mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                restore_best_weights=True, verbose=1),
    ]
    if config['lr_schedule'] in ['cosine', 'both']:
        wu_cbs.append(callbacks.LearningRateScheduler(
            lambda e: config['warmup_lr'] * 0.5 * (
                1 + np.cos(np.pi * e / config['warmup_epochs'])), verbose=0))
    if config['lr_schedule'] in ['plateau', 'both']:
        wu_cbs.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1))

    if train_gen:
        warmup_hist = model.fit(train_gen, validation_data=val_data,
                                epochs=config['warmup_epochs'], callbacks=wu_cbs, verbose=1)
    else:
        warmup_hist = model.fit(X_train, y_train, validation_data=val_data,
                                batch_size=config['batch_size'],
                                epochs=config['warmup_epochs'], callbacks=wu_cbs, verbose=1)
    logger.end_stage("STAGE 1: WARMUP TRAINING", warmup_hist)

    # Stage 2: Fine-tune
    logger.start_stage("STAGE 2: FINE-TUNING")
    model.compile(optimizer=Adam(learning_rate=config['finetune_lr']),
                  loss=loss_fn, metrics=['accuracy'])
    ft_ckpt = os.path.join(config['output_dir'], 'finetune_best.weights.h5')
    ft_cbs = [
        callbacks.ModelCheckpoint(ft_ckpt, monitor='val_accuracy',
                                  save_best_only=True, save_weights_only=True,
                                  mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                restore_best_weights=True, verbose=1),
    ]
    if config['lr_schedule'] in ['plateau', 'both']:
        ft_cbs.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1))

    if train_gen:
        finetune_hist = model.fit(train_gen, validation_data=val_data,
                                  epochs=config['finetune_epochs'], callbacks=ft_cbs, verbose=1)
    else:
        finetune_hist = model.fit(X_train, y_train, validation_data=val_data,
                                  batch_size=config['batch_size'],
                                  epochs=config['finetune_epochs'], callbacks=ft_cbs, verbose=1)
    logger.end_stage("STAGE 2: FINE-TUNING", finetune_hist)

    plot_training_history(warmup_hist, finetune_hist, config['output_dir'])

    fp32_path = os.path.join(config['output_dir'], 'tcn_fp32.keras')
    model.save(fp32_path)

    logger.start_stage("EVALUATION: FP32")
    fp32_acc = evaluate_model(model, X_test, y_test, class_names,
                              config['output_dir'], "FP32")
    logger.log_evaluation("FP32", fp32_acc,
                          os.path.join(config['output_dir'], 'classification_report_fp32.txt'))

    logger.start_stage("TFLITE CONVERSION (PTQ)")
    int8_path = os.path.join(config['output_dir'], 'tcn_int8.tflite')
    convert_to_tflite_int8(model, X_calib, int8_path)

    logger.start_stage("EVALUATION: INT8 TFLite")
    int8_acc = evaluate_tflite(int8_path, X_test, y_test, class_names, config['output_dir'])
    logger.log_evaluation("INT8", int8_acc,
                          os.path.join(config['output_dir'], 'classification_report_int8.txt'))

    model_sizes = {
        "FP32 (.keras)": f"{os.path.getsize(fp32_path)/(1024**2):.2f} MB",
        "INT8 (.tflite)": f"{os.path.getsize(int8_path)/1024:.1f} KB"
    }
    logger.log_final_results(fp32_acc, int8_acc, model_sizes,
                             warmup_hist, finetune_hist, config)

    drop = fp32_acc - int8_acc
    total_time = time.time() - script_start
    print(f"\n{'='*70}")
    print(f"FP32: {fp32_acc:.2f}%  INT8: {int8_acc:.2f}%  Drop: {drop:+.2f}%"
          f"  Time: {format_time(total_time)}")
    print(f"INT8 size: {os.path.getsize(int8_path)/1024:.1f} KB")
    print(f"{'='*70}")
    print(f"✓ Results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()
    total_script_time = time.time() - script_start
    print(f"\nScript completed in: {format_time(total_script_time)}")
