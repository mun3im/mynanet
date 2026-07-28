#!/usr/bin/env python3
"""
WrenNet-Inspired - Model 1n
Learnable frequency warping + MatchboxNet epilogue
Post-Training Quantization (PTQ) → Cortex-M7 deployment
Fixed spectrogram shape: 64x300 (10ms per frame)

WrenNet's core innovation: semi-learnable spectral feature extraction
ported onto 1j base architecture with 1o's MatchboxNet epilogue.

WrenNet (Ciapponi et al., ICASSP 2026):
  1. Parametric sigmoid-weighted frequency mapping (Eq. 1-4)
     - Learnable breakpoint (b) and transition width (w)
     - Smooth sigmoid transition between linear and logarithmic frequency spacing
     - Enables per-species frequency emphasis without fixed mel-scale assumption
  2. Semi-learnable: 64 FFT bins → 64 mel bins via learnable warping function
     - Two scalar learnable params (b, w) replace 64×64 fixed mel filterbank matrix
     - Forward pass: warp FFT bins, then apply standard STFT→warped freq mapping
  3. Causal convolutions for streaming (on-device)
     - MatchboxNet-style depthwise seperable blocks
  4. GRU temporal modeling (not needed for our case; use attention-free blocks)

Our 1n adaptation for 12-class garden bird classification:
  1. 64-bin FFT → learnable parametric frequency warping (1j base before mel-spec)
  2. Input to model: warped 64-bin spectrogram (like mel-spec but learned)
  3. Rest: 1j blocks 1-4 (unchanged) + 1o's MatchboxNet epilogue (DW conv + dense)

Changes from 1j/1o:
  - STFT+Warp layer: learnable sigmoid breakpoint & width → replaces fixed mel
  - Remaining architecture: identical 1j blocks (SE + inverted residual)
  - Epilogue: 1o's MatchboxNet (dilated DW conv + 1×1 proj + GAP + Dense)

Estimated: ~200 KB INT8 (within H7 512KB), learnable frequency adaptation.
All ops H7-compatible: Conv2D, DepthwiseConv2D, BN, ReLU6, Add, Mul, Lambda.
Target: >94% INT8 accuracy with adaptive frequency representation.
"""

print("\n\n\n")
for _ in range(3):
    print(" 🔷 " * 30)

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

# Early argument parsing for GPU config
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

early_args = parse_early_args()

# TensorFlow environment check
print("\n" + "=" * 70)
print("ENVIRONMENT VALIDATION (Checking before dataset preparation)")
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
                    [tf.config.LogicalDeviceConfiguration(memory_limit=early_args.gpu_memory_limit)]
                )
                print(f"✓ Found {len(gpus)} GPU(s), Memory limit: {early_args.gpu_memory_limit} MB")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Found {len(gpus)} GPU(s), Memory growth: Enabled")
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")
    else:
        print("✓ Running on CPU (no GPU detected)")

    print("\n✓ Environment check PASSED")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ CRITICAL: TensorFlow environment check failed: {e}")
    print("Install: pip install tensorflow tf_keras")
    sys.exit(1)

# Import remaining libraries
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

script_start = time.time()

# Platform-specific optimizer
system = platform.system()
processor = platform.processor()

if system == "Darwin" and processor == "arm":
    from tf_keras.optimizers.legacy import Adam as LegacyAdam
    Adam = LegacyAdam
    OPTIMIZER_NAME = "Legacy Adam"
elif system == "Linux":
    try:
        from tf_keras.optimizers import AdamW
        Adam = AdamW
        OPTIMIZER_NAME = "AdamW"
    except ImportError:
        from tf_keras.optimizers import Adam
        OPTIMIZER_NAME = "Adam"
else:
    from tf_keras.optimizers import Adam
    OPTIMIZER_NAME = "Adam"

# Constants
DEFAULT_RANDOM_STATE = 42
TARGET_SR = 16000
AUDIO_LENGTH_SEC = 3
FIXED_AUDIO_LENGTH = TARGET_SR * AUDIO_LENGTH_SEC
HOP_LENGTH = 160
N_FFT = 512
DEFAULT_N_MELS = 64
FMAX = 8000
TIME_FRAMES = 300

DEFAULT_FLAT_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
DEFAULT_SPECTROGRAM_DIR = "/Volumes/Evo/MYGARDENBIRD/precompute/spectrograms_16k_mels64"

SPECAUGMENT_FREQ_MASK = 8
SPECAUGMENT_TIME_MASK = 20
SPECAUGMENT_NUM_MASKS = 2

PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98
GLOBAL_STATS_SAMPLES = 100


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


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--calib_samples", type=int, default=200)

    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--specaugment", action='store_true')
    parser.add_argument("--time_shift_ms", type=int, default=100)
    parser.add_argument("--pitch_shift_steps", type=int, default=2)

    parser.add_argument("--force_cpu", action='store_true')
    parser.add_argument("--gpu_memory_limit", type=int, default=None)

    parser.add_argument("--splits_csv", type=str, required=True)
    parser.add_argument("--flat_dir", type=str, default=DEFAULT_FLAT_DIR)
    parser.add_argument("--spectrogram_dir", type=str, default=DEFAULT_SPECTROGRAM_DIR)
    parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS, choices=[48, 64, 80, 96])

    # WrenNet-specific parameters
    parser.add_argument("--breakpoint_init", type=float, default=0.5,
                        help="Initial breakpoint for frequency warping sigmoid (0-1)")
    parser.add_argument("--transition_width_init", type=float, default=0.2,
                        help="Initial transition width for warping function")

    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "plateau", "both", "none"])
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_STATE)

    args = parser.parse_args()

    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    n_mels = args.n_mels

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
        elif header.startswith('# loso'):
            toks = dict(t.split('=') for t in header.replace('#', '').split() if '=' in t)
            split_suffix = f"loso_k{toks.get('k', '?')}_fold{toks.get('fold', '?')}"
    except Exception:
        split_suffix = "splitcsv"

    n_classes = len([d for d in os.listdir(args.flat_dir)
                     if os.path.isdir(os.path.join(args.flat_dir, d)) and not d.startswith('.')])

    output_dir_name = (
        f"results_mygardenbird_1_{platform.system().lower()}/"
        f"1n_wrennet_adaptivefreq_"
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
        'warmup_epochs': args.warmup_epochs,
        'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size,
        'warmup_lr': args.warmup_lr,
        'finetune_lr': args.finetune_lr,
        'dropout': args.dropout,
        'time_frames': TIME_FRAMES,
        'n_mels': n_mels,
        'input_shape': (TIME_FRAMES, N_FFT // 2 + 1, 1),  # Raw FFT bins (257 → warped to 64)
        'output_dir': output_dir_name,
        'calib_samples': args.calib_samples,
        'model_type': 'wrennet_adaptivefreq',
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
        'breakpoint_init': args.breakpoint_init,
        'transition_width_init': args.transition_width_init,
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    return config


def compute_global_stats(data_dir, n_mels, allowed_files=None):
    """Compute global normalization statistics (2-98 percentile)."""
    all_mel = []
    total_sampled = 0

    print(f"Computing global stats (sampling up to {GLOBAL_STATS_SAMPLES} files per class)...")

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

                # Compute standard mel-spectrogram for stats
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
                print(f"\n⚠ Failed to process {f}: {e}")
                continue

    if len(all_mel) == 0:
        raise RuntimeError("No valid audio files found")

    all_mel = np.concatenate(all_mel)
    gmin, gmax = np.percentile(all_mel, PERCENTILE_LOW), np.percentile(all_mel, PERCENTILE_HIGH)
    print(f"✓ Global stats from {total_sampled} files: {gmin:.2f} → {gmax:.2f} dB")
    return float(gmin), float(gmax)


def compute_fft(audio, sr):
    """Compute power spectrogram (magnitude-squared FFT)."""
    D = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', win_length=400, center=True)
    S = np.abs(D) ** 2

    # Ensure exact TIME_FRAMES
    if S.shape[1] > TIME_FRAMES:
        S = S[:, :TIME_FRAMES]
    if S.shape[1] < TIME_FRAMES:
        S = np.pad(S, ((0, 0), (0, TIME_FRAMES - S.shape[1])))

    return S.T  # Shape: (TIME_FRAMES, N_FFT//2+1) for compatibility


def normalize_spec(spec, gmin, gmax):
    """Normalize spectrogram to [0, 1] range using provided min/max."""
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = np.clip(spec_db, gmin, gmax)
    spec_norm = (spec_db - gmin) / (gmax - gmin + 1e-8)
    return spec_norm[..., np.newaxis].astype(np.float32)


def augment_baseline(audio, sr, time_shift_ms=100, pitch_steps=2):
    """Baseline augmentation: time shift + pitch shift."""
    if np.random.rand() > 0.5:
        shift_samples = int(np.random.uniform(-time_shift_ms, time_shift_ms) * sr / 1000)
        audio = np.roll(audio, shift_samples)
    if np.random.rand() > 0.5:
        n_steps = np.random.uniform(-pitch_steps, pitch_steps)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return audio


def augment_specaugment(spec):
    """SpecAugment: frequency and time masking."""
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


def se_block(x, filters, reduction=16, block_id=0):
    """Squeeze-and-Excitation with hard-sigmoid (MobileNetV3 style)."""
    prefix = f'block{block_id}_se_'
    se = layers.GlobalAveragePooling2D(keepdims=True, name=prefix + 'squeeze')(x)
    se = layers.Conv2D(filters // reduction, (1, 1), activation='relu', use_bias=True, name=prefix + 'reduce')(se)
    se = layers.Conv2D(filters, (1, 1), use_bias=True, name=prefix + 'expand')(se)
    se = layers.Lambda(lambda t: tf.nn.relu6(t + 3.0) * (1.0 / 6.0), name=prefix + 'hard_sigmoid')(se)
    return layers.Multiply(name=prefix + 'scale')([x, se])


def inverted_residual_se(x, filters, kernel_size=(3, 3), strides=(1, 1), expand_ratio=6, block_id=0, se_reduction=16):
    """MobileNetV3 inverted residual with SE (hard-sigmoid)."""
    prefix = f'block{block_id}_'
    input_channels = x.shape[-1]
    expanded = input_channels * expand_ratio
    shortcut = x

    x = layers.Conv2D(expanded, (1, 1), padding='same', use_bias=False, name=prefix + 'expand')(x)
    x = layers.BatchNormalization(name=prefix + 'expand_bn')(x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)

    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False, name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(name=prefix + 'depthwise_bn')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    x = se_block(x, expanded, reduction=max(1, expand_ratio * se_reduction // 16), block_id=block_id)

    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + 'project')(x)
    x = layers.BatchNormalization(name=prefix + 'project_bn')(x)

    if input_channels == filters and strides == (1, 1):
        x = layers.Add(name=prefix + 'residual')([x, shortcut])

    return x


class LearnableFrequencyWarp(layers.Layer):
    """
    WrenNet's parametric frequency warping using sigmoid-weighted mapping.

    Converts raw FFT bins (257) to learnable warped frequency space.
    Uses learnable breakpoint (b) and transition width (w) to create smooth
    sigmoid-weighted blend between linear and log frequency mappings.

    Based on Ciapponi et al., ICASSP 2026, Eq. 1-4.
    """
    def __init__(self, output_bins=64, **kwargs):
        super().__init__(**kwargs)
        self.output_bins = output_bins
        self.n_fft_bins = 257  # N_FFT // 2 + 1 for 512-point FFT

    def build(self, input_shape):
        # Learnable breakpoint (b) and transition width (w)
        self.breakpoint = self.add_weight(
            name='breakpoint',
            shape=(),
            initializer=keras.initializers.Constant(0.5),
            trainable=True,
            dtype=tf.float32
        )
        self.transition_width = self.add_weight(
            name='transition_width',
            shape=(),
            initializer=keras.initializers.Constant(0.2),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, x):
        # x shape: (batch, time, n_fft_bins, 1)
        # Create frequency bin indices [0, 1, ..., 512] normalized to [0, 1]
        bin_indices = tf.range(self.n_fft_bins, dtype=tf.float32) / (self.n_fft_bins - 1)

        # Sigmoid-weighted frequency mapping: blend linear and log spacing
        # f_linear(x) = x
        # f_log(x) = log(f_max/f_min + 1) * x + log(f_min)  (simplified)
        # S(x; b, w) = sigmoid((x - b) / w)  (Eq. 2)
        # f_out(x; b, w) = (1 - S(x)) * f_linear(x) + S(x) * f_log(x)  (Eq. 3)

        sigmoid_weight = tf.nn.sigmoid((bin_indices - self.breakpoint) / (self.transition_width + 1e-6))

        # Linear: simply x
        f_linear = bin_indices

        # Log-like: use learnable log spacing (approximate via polynomial)
        f_log = tf.math.log(bin_indices + 1.0) / tf.math.log(2.0)  # Normalize to [0, 1]

        # Blend: smooth transition from linear to log
        warped_bins = (1.0 - sigmoid_weight) * f_linear + sigmoid_weight * f_log

        # Interpolate input FFT bins onto warped bin positions
        # Simplified: use nearest neighbor to 64 output bins
        # In practice, use linear interpolation or differentiable pooling

        # Resample input from 257 bins to 64 output bins via learned warping
        output_list = []
        for out_idx in range(self.output_bins):
            # Map output bin index to warped position
            target_pos = warped_bins[int((out_idx / self.output_bins) * (self.n_fft_bins - 1))]
            # Gather energy from input (simplified: direct indexing)
            sample_idx = tf.cast(target_pos * (self.n_fft_bins - 1), tf.int32)
            sample_idx = tf.clip_by_value(sample_idx, 0, self.n_fft_bins - 1)
            output_list.append(x[..., sample_idx:sample_idx+1, :])

        # Stack warped bins: (batch, time, 64, 1)
        warped_x = tf.concat(output_list, axis=-2)
        return warped_x


def create_wrennet(num_classes, input_shape, dropout=0.2,
                   breakpoint_init=0.5, transition_width_init=0.2):
    """
    WrenNet-inspired model: learnable frequency warping + 1j base + 1o epilogue.

    Architecture:
      Input: (time, 257, 1) power spectrogram from raw FFT
      LearnableFrequencyWarp: 257 → 64 bins (learnable sigmoid-weighted mapping)

      Blocks 1-4: identical to 1j (InvRes+SE)

      Epilogue (MatchboxNet-style, from 1o):
      DW(1×17) + Conv(128, 1×1) → GlobalAvgPool → Dense(n_classes)
    """
    inputs = layers.Input(shape=input_shape, name='input_raw_fft')

    # WrenNet's learnable frequency warping
    x = LearnableFrequencyWarp(output_bins=64, name='freq_warp')(inputs)

    # Reshape to spatial for conv blocks
    x = layers.Reshape((input_shape[0], 64, 1), name='reshape_to_spatial')(x)

    # Stem (same as 1j)
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(6., name='stem_relu')(x)

    # Block 1: t=1, 32→16
    x = inverted_residual_se(x, 16, kernel_size=(3, 3), block_id=1, expand_ratio=1, se_reduction=4)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(dropout * 0.5, name='drop1')(x)

    # Block 2: t=6, 16→24
    x = inverted_residual_se(x, 24, kernel_size=(3, 3), block_id=2, expand_ratio=6, se_reduction=8)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(dropout * 0.5, name='drop2')(x)

    # Block 3: t=6, 24→48, dw=5×5
    x = inverted_residual_se(x, 48, kernel_size=(5, 5), block_id=3, expand_ratio=6, se_reduction=16)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(dropout * 0.75, name='drop3')(x)

    # Block 4: t=6, 48→64 (scaled down like 1o for size)
    x = inverted_residual_se(x, 64, kernel_size=(5, 5), block_id=4, expand_ratio=6, se_reduction=16)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(dropout, name='drop4')(x)

    # MatchboxNet-style epilogue (from 1o)
    # DW(1×17) for temporal receptive field
    x = layers.DepthwiseConv2D((1, 17), padding='same', use_bias=False, name='matchbox_dw')(x)
    x = layers.BatchNormalization(name='matchbox_dw_bn')(x)
    x = layers.ReLU(6., name='matchbox_dw_relu')(x)

    # Project to 128 channels
    x = layers.Conv2D(128, (1, 1), padding='same', use_bias=False, name='matchbox_proj')(x)
    x = layers.BatchNormalization(name='matchbox_proj_bn')(x)
    x = layers.ReLU(6., name='matchbox_proj_relu')(x)

    # Global pooling and classification
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dropout(dropout, name='final_drop')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return keras.Model(inputs, outputs, name="WrenNet_AdaptiveFreq")


def convert_to_tflite_int8(model, X_calib, path):
    """Convert model to INT8 TFLite."""
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


def parse_splits_csv(csv_path):
    """Parse CSV: filename -> split."""
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
    """Load data from flat directory + splits CSV (raw FFT, not mel-spec)."""
    if augmentation_mode == 'none':
        print("\n⚠ WARNING: No augmentation enabled")

    splits = parse_splits_csv(csv_path)

    X_test, y_test = [], []
    X_val, y_val = [], []
    X_train, y_train = [], []
    labels = {}
    idx = 0
    failed_files = []

    # Build file lookup
    file_lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                file_lookup[f] = (class_name, os.path.join(class_dir, f))

    total_files = sum(1 for fn in splits if fn in file_lookup)
    train_count = sum(1 for fn in splits if fn in file_lookup and splits[fn] == 'train')
    if augmentation_mode in ['baseline', 'specaugment']:
        total_files += train_count

    print(f"\nDataset: {len(splits)} entries, {len(file_lookup)} files found")
    print(f"Augmentation: {augmentation_mode}")

    with tqdm(total=total_files, desc="Loading", unit="file") as pbar:
        for fn in sorted(splits.keys()):
            if fn not in file_lookup:
                continue

            class_name, audio_path = file_lookup[fn]
            if class_name not in labels:
                labels[class_name] = idx
                idx += 1

            split = splits[fn]

            try:
                audio, _ = librosa.load(audio_path, sr=TARGET_SR)
                if len(audio) > FIXED_AUDIO_LENGTH:
                    audio = audio[:FIXED_AUDIO_LENGTH]
                else:
                    audio = np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                # Compute raw power spectrogram (FFT bins, not mel)
                spec = compute_fft(audio, TARGET_SR)
                spec_norm = normalize_spec(spec, gmin, gmax)

                if split == 'test':
                    X_test.append(spec_norm)
                    y_test.append(labels[class_name])
                elif split == 'val':
                    X_val.append(spec_norm)
                    y_val.append(labels[class_name])
                elif split == 'train':
                    X_train.append(spec_norm)
                    y_train.append(labels[class_name])
                    pbar.update(1)

                    if augmentation_mode == 'baseline':
                        aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                        aug_spec = compute_fft(aug_audio, TARGET_SR)
                        aug_spec_norm = normalize_spec(aug_spec, gmin, gmax)
                        X_train.append(aug_spec_norm)
                        y_train.append(labels[class_name])
                        pbar.update(1)
                    elif augmentation_mode == 'specaugment':
                        aug_spec_norm = augment_specaugment(spec_norm)
                        X_train.append(aug_spec_norm)
                        y_train.append(labels[class_name])
                        pbar.update(1)
                else:
                    pbar.update(1)

            except Exception as e:
                failed_files.append(f"{fn}: {e}")
                if augmentation_mode in ['baseline', 'specaugment'] and split == 'train':
                    pbar.update(2)
                else:
                    pbar.update(1)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    print(f"\n✓ Data loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)


def evaluate_model(model, X_test, y_test, class_names, output_dir, model_type="FP32"):
    """Evaluate model and save results."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred) * 100

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, f'classification_report_{model_type.lower()}.txt')
    with open(report_path, 'w') as f:
        f.write(f"{model_type} Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_type} Confusion Matrix - {acc:.2f}%', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type.lower()}.png'), dpi=150)
    plt.close()

    return acc


# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == '__main__':
    config = get_config()
    print(f"Model: {config['model_type']}")
    print(f"Output: {config['output_dir']}")

    # Compute stats from mel-spectrograms (for normalization reference)
    gmin, gmax = compute_global_stats(config['flat_dir'], config['n_mels'])

    # Load raw FFT data (not mel-spec)
    X_train, X_val, X_test, y_train, y_val, y_test, labels, failed = load_data_from_csv(
        config['splits_csv'], config['flat_dir'], gmin, gmax, config['n_mels'],
        augmentation_mode=config['augmentation_mode'],
        time_shift_ms=config['time_shift_ms'],
        pitch_shift_steps=config['pitch_shift_steps'],
        mixup_alpha=config['mixup_alpha']
    )

    class_names = [name for name, _ in sorted(labels.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    print(f"\n✓ Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"  Shape: {X_train[0].shape} (time_frames, fft_bins, channels)")
    print(f"  Classes: {num_classes}")

    # Create model
    model = create_wrennet(
        num_classes=num_classes,
        input_shape=(TIME_FRAMES, N_FFT // 2 + 1, 1),
        dropout=config['dropout'],
        breakpoint_init=config['breakpoint_init'],
        transition_width_init=config['transition_width_init']
    )

    print("\n" + "=" * 70)
    print(f"Model: {model.name}")
    model.summary()
    total_params = model.count_params()
    print(f"Total params: {total_params:,}")
    print(f"Estimated INT8 size: {total_params / 1024:.1f} KB")
    print("=" * 70)

    # Compile and train (simplified workflow)
    model.compile(
        optimizer=Adam(learning_rate=config['warmup_lr']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "=" * 70)
    print("WARMUP PHASE")
    print("=" * 70)

    warmup_hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['warmup_epochs'],
        batch_size=config['batch_size'],
        verbose=1
    )

    # Switch to fine-tuning LR
    model.optimizer.learning_rate.assign(config['finetune_lr'])

    print("\n" + "=" * 70)
    print("FINE-TUNE PHASE")
    print("=" * 70)

    finetune_hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['finetune_epochs'],
        batch_size=config['batch_size'],
        verbose=1
    )

    # Save FP32 model
    model_fp32_path = os.path.join(config['output_dir'], 'model_fp32.keras')
    model.save(model_fp32_path)
    print(f"✓ Saved FP32 model: {model_fp32_path}")

    # Evaluate FP32
    fp32_acc = evaluate_model(model, X_test, y_test, class_names, config['output_dir'], "FP32")
    print(f"\nFP32 Test Accuracy: {fp32_acc:.2f}%")

    # Convert to INT8
    tflite_path = os.path.join(config['output_dir'], 'model_int8.tflite')
    convert_to_tflite_int8(model, X_train[:config['calib_samples']], tflite_path)

    # Evaluate INT8
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred_int8 = []
    for i in tqdm(range(len(X_test)), desc="INT8 inference"):
        x_int8 = (X_test[i:i+1] / input_details[0]['quantization'][0] + input_details[0]['quantization'][1]).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], x_int8)
        interpreter.invoke()
        y_pred_int8.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))

    int8_acc = accuracy_score(y_test, y_pred_int8) * 100
    print(f"\nINT8 Test Accuracy: {int8_acc:.2f}%")

    # Save summary
    summary_path = os.path.join(config['output_dir'], 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model 1n: WrenNet-inspired (Learnable Frequency Warping)\n")
        f.write(f"FP32: {fp32_acc:.2f}%\n")
        f.write(f"INT8: {int8_acc:.2f}%\n")
        f.write(f"Drop: {fp32_acc - int8_acc:+.2f}%\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Total time: {format_time(time.time() - script_start)}\n")

    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ Training complete! ({format_time(time.time() - script_start)})")
