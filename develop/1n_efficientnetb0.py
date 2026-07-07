#!/usr/bin/env python3
"""
Model 1n — EfficientNetB0 (pretrained ImageNet, accuracy ceiling)
MyGardenBird 12-class bird sound classifier.

NOT MCU-deployable (~5MB INT8, 10× over Portenta H7 512KB limit).
Serves as the transfer-learning accuracy ceiling for the series.

Feature extraction: mel spectrogram → 224×224×3, per-sample p2/p98 normalised,
                    efficientnet.preprocess_input (Stage9 pipeline).
Architecture: EfficientNetB0(include_top=False, pooling='avg') + Dropout(0.3) +
              Dense(256, relu) + Dropout(0.2) + Dense(num_classes, softmax).
Training: single phase, EarlyStopping(patience=10) + ReduceLROnPlateau(patience=5).
"""

print("\n\n\n")
for _ in range(3):
    print(" 🔷 " * 30)

import os
import sys
import argparse
import platform
import warnings

warnings.filterwarnings("ignore")

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_early_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--force_cpu", action='store_true')
    p.add_argument("--gpu_memory_limit", type=int, default=None)
    a, _ = p.parse_known_args()
    if a.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return a

early_args = parse_early_args()

print("\n" + "=" * 70)
print("ENVIRONMENT VALIDATION")
print("=" * 70)

try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers
    from tensorflow.keras.applications import efficientnet as effnet_preprocess

    print(f"✓ TensorFlow: {tf.__version__}  Keras: {keras.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus and not early_args.force_cpu:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ {len(gpus)} GPU(s): {gpus[0].name}  (memory growth enabled)")
    else:
        print("✓ CPU mode")
    print("=" * 70)
except Exception as e:
    print(f"✗ Environment check failed: {e}")
    sys.exit(1)

import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR          = 16000
AUDIO_LENGTH_SEC   = 3
FIXED_AUDIO_LEN    = TARGET_SR * AUDIO_LENGTH_SEC   # 48000 samples
TARGET_SIZE        = 224                             # EfficientNetB0 input side
HOP_LENGTH         = FIXED_AUDIO_LEN // TARGET_SIZE  # ≈214; gives ~224 time frames
DROPOUT1           = 0.3
DROPOUT2           = 0.2

DEFAULT_SPLITS_CSV = '/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv'
DEFAULT_FLAT_DIR   = '/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz'


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def get_config():
    parser = argparse.ArgumentParser(description='Model 1n — EfficientNetB0 pretrained')
    parser.add_argument('--splits_csv',    default=DEFAULT_SPLITS_CSV)
    parser.add_argument('--flat_dir',      default=DEFAULT_FLAT_DIR)
    parser.add_argument('--n_mels',        type=int,   default=224)
    parser.add_argument('--random_seed',   type=int,   default=42)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--num_epochs',    type=int,   default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--mixup',         type=float, default=None, metavar='ALPHA')
    parser.add_argument('--specaugment',   action='store_true')
    # Accepted but unused — keeps compatibility with run_seabird12_ablation.sh
    parser.add_argument('--warmup_epochs', type=int,   default=None)
    parser.add_argument('--dropout',       type=float, default=None)
    parser.add_argument('--force_cpu',     action='store_true')
    parser.add_argument('--gpu_memory_limit', type=int, default=None)
    args = parser.parse_args()

    if args.mixup is not None:
        aug_mode = f'mixup{args.mixup}'
    elif args.specaugment:
        aug_mode = 'specaugment'
    else:
        aug_mode = 'noaug'

    plat = platform.system().lower()
    out_dir = (
        f"results_mygardenbird_1_{plat}/"
        f"1n_efficientnetb0_mels{args.n_mels}_rand{args.random_seed}"
        f"_{aug_mode}_split80:10:10_{plat}"
    )
    return {
        'splits_csv':    args.splits_csv,
        'flat_dir':      args.flat_dir,
        'n_mels':        args.n_mels,
        'random_seed':   args.random_seed,
        'batch_size':    args.batch_size,
        'num_epochs':    args.num_epochs,
        'learning_rate': args.learning_rate,
        'aug_mode':      aug_mode,
        'mixup_alpha':   args.mixup,
        'specaugment':   args.specaugment,
        'output_dir':    out_dir,
        'platform':      plat,
    }


# ---------------------------------------------------------------------------
# Feature extraction (Stage9 pipeline)
# ---------------------------------------------------------------------------
def compute_spec(audio, n_mels, augment=False):
    """Audio → 224×224×3 float32 ready for EfficientNetB0."""
    audio = np.squeeze(audio).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_fft=2048, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmin=0.0, fmax=TARGET_SR // 2,
        center=True, pad_mode='constant'
    )
    log_mel = librosa.power_to_db(mel, top_db=None)

    if augment:
        if np.random.rand() < 0.5:
            w = np.random.randint(0, max(1, log_mel.shape[1] // 10))
            t = np.random.randint(0, max(1, log_mel.shape[1] - w))
            log_mel[:, t:t + w] = log_mel.min()
        if np.random.rand() < 0.5:
            w = np.random.randint(0, max(1, log_mel.shape[0] // 10))
            f = np.random.randint(0, max(1, log_mel.shape[0] - w))
            log_mel[f:f + w, :] = log_mel.min()

    # Crop / pad to (TARGET_SIZE, TARGET_SIZE)
    h, w = log_mel.shape
    if h > TARGET_SIZE:
        log_mel = log_mel[:TARGET_SIZE, :]
    elif h < TARGET_SIZE:
        log_mel = np.pad(log_mel, ((0, TARGET_SIZE - h), (0, 0)),
                         constant_values=log_mel.min())
    if w > TARGET_SIZE:
        log_mel = log_mel[:, :TARGET_SIZE]
    elif w < TARGET_SIZE:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_SIZE - w)),
                         constant_values=log_mel.min())

    # Stack to 3-channel
    rgb = np.stack([log_mel] * 3, axis=-1)  # (224, 224, 3)

    # Per-sample p2/p98 normalise → [0, 255]
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2 + 1e-8:
        rgb = np.clip(rgb, p2, p98)
        rgb = ((rgb - p2) / (p98 - p2) * 255.0).astype(np.float32)
    else:
        rgb = np.full_like(rgb, 128.0, dtype=np.float32)

    # EfficientNet preprocessing (ImageNet mean/std normalisation)
    rgb = effnet_preprocess.preprocess_input(rgb)
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# CSV-based flat-dir dataset loading
# ---------------------------------------------------------------------------
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


def build_file_lists(splits_csv, flat_dir):
    """Return {split: [(path, class_idx), ...]} and classes list."""
    splits = parse_splits_csv(splits_csv)

    # Build filename → (class_name, full_path)
    file_lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                file_lookup[f] = (class_name, os.path.join(class_dir, f))

    classes = sorted({file_lookup[fn][0] for fn in splits if fn in file_lookup})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    result = {'train': [], 'val': [], 'test': []}
    for fn, split in splits.items():
        if split not in result or fn not in file_lookup:
            continue
        class_name, path = file_lookup[fn]
        result[split].append((path, class_to_idx[class_name]))

    for s in result:
        result[s].sort()

    print(f"  train: {len(result['train'])}  val: {len(result['val'])}  "
          f"test: {len(result['test'])}  classes: {len(classes)}")
    return result, classes


# ---------------------------------------------------------------------------
# TF Dataset pipeline
# ---------------------------------------------------------------------------
def make_tf_dataset(file_list, n_mels, augment, batch_size, shuffle, seed, mixup_alpha=None, num_classes=None):
    paths  = [p for p, _ in file_list]
    labels = [l for _, l in file_list]

    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_and_process(path, label):
        def _fn(p, l):
            audio, _ = librosa.load(p.numpy().decode(), sr=TARGET_SR, duration=AUDIO_LENGTH_SEC)
            if len(audio) < FIXED_AUDIO_LEN:
                audio = np.pad(audio, (0, FIXED_AUDIO_LEN - len(audio)))
            else:
                audio = audio[:FIXED_AUDIO_LEN]
            spec = compute_spec(audio, n_mels, augment=augment)
            return spec.astype(np.float32), np.int32(l)

        spec, lbl = tf.py_function(_fn, [path, label], [tf.float32, tf.int32])
        spec.set_shape([TARGET_SIZE, TARGET_SIZE, 3])
        lbl.set_shape([])
        return spec, lbl

    ds = ds.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if mixup_alpha is not None and num_classes is not None:
        def to_one_hot(x, y):
            oh = tf.one_hot(tf.cast(y, tf.int32), num_classes)
            oh.set_shape([None, num_classes])
            return x, oh

        def mixup_batch(images, labels_oh):
            lam = tf.cast(
                tf.py_function(
                    lambda: np.float32(np.random.beta(mixup_alpha, mixup_alpha)),
                    [], tf.float32
                ), tf.float32
            )
            idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
            mixed_x = lam * images + (1.0 - lam) * tf.gather(images, idx)
            mixed_y = lam * labels_oh + (1.0 - lam) * tf.gather(labels_oh, idx)
            mixed_x.set_shape(images.shape)
            mixed_y.set_shape(labels_oh.shape)
            return mixed_x, mixed_y

        ds = ds.map(to_one_hot)
        ds = ds.map(mixup_batch)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def create_model(num_classes, input_shape=(TARGET_SIZE, TARGET_SIZE, 3)):
    base = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    model = keras.Sequential([
        base,
        layers.Dropout(DROPOUT1, name='drop1'),
        layers.Dense(256, activation='relu', name='dense_head'),
        layers.Dropout(DROPOUT2, name='drop2'),
        layers.Dense(num_classes, activation='softmax', name='output'),
    ], name='EfficientNetB0_1r')
    return model, base


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------
def export_tflite_int8(model, repr_ds, out_path):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        samples = []
        for batch_x, _ in repr_ds:
            for x in batch_x.numpy():
                samples.append(x[np.newaxis, ...])
            if len(samples) >= 200:
                break

        def repr_data():
            for s in samples[:200]:
                yield [s.astype(np.float32)]

        converter.representative_dataset = repr_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
        return len(tflite_model)
    except Exception as e:
        print(f"  [WARN] INT8 TFLite conversion failed: {e}")
        return None


def eval_tflite(tflite_path, ds):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    inp_scale, inp_zp = interpreter.get_input_details()[0]['quantization']
    out_scale, out_zp = interpreter.get_output_details()[0]['quantization']

    y_true, y_pred = [], []
    for batch_x, batch_y in tqdm(ds, desc='INT8 inference'):
        for i in range(len(batch_x)):
            x = batch_x[i].numpy()[np.newaxis, ...]
            x_q = np.round(x / inp_scale + inp_zp).astype(np.int8)
            interpreter.set_tensor(inp_idx, x_q)
            interpreter.invoke()
            out_q = interpreter.get_tensor(out_idx)
            out = (out_q.astype(np.float32) - out_zp) * out_scale
            y_pred.append(np.argmax(out))
            y_true.append(int(batch_y[i].numpy()))
    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = get_config()

    np.random.seed(cfg['random_seed'])
    tf.random.set_seed(cfg['random_seed'])

    import random; random.seed(cfg['random_seed'])

    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel 1n — EfficientNetB0 pretrained (accuracy ceiling)")
    print(f"  n_mels={cfg['n_mels']}  seed={cfg['random_seed']}  aug={cfg['aug_mode']}")
    print(f"  splits_csv: {cfg['splits_csv']}")
    print(f"  flat_dir:   {cfg['flat_dir']}")
    print(f"  output:     {cfg['output_dir']}")
    print()

    # ---- Dataset ----
    print("Building file lists...")
    file_lists, classes = build_file_lists(cfg['splits_csv'], cfg['flat_dir'])
    num_classes = len(classes)

    mixup_alpha = cfg['mixup_alpha'] if cfg['aug_mode'].startswith('mixup') else None

    train_ds = make_tf_dataset(
        file_lists['train'], cfg['n_mels'],
        augment=(cfg['aug_mode'] == 'specaugment'),
        batch_size=cfg['batch_size'], shuffle=True, seed=cfg['random_seed'],
        mixup_alpha=mixup_alpha, num_classes=num_classes
    )
    val_ds = make_tf_dataset(
        file_lists['val'], cfg['n_mels'],
        augment=False, batch_size=cfg['batch_size'], shuffle=False, seed=None
    )
    test_ds = make_tf_dataset(
        file_lists['test'], cfg['n_mels'],
        augment=False, batch_size=cfg['batch_size'], shuffle=False, seed=None
    )

    # val also needs one-hot when mixup is active (CategoricalCrossentropy)
    if mixup_alpha is not None:
        def to_one_hot_val(x, y):
            oh = tf.one_hot(tf.cast(y, tf.int32), num_classes)
            oh.set_shape([None, num_classes])
            return x, oh
        val_ds = val_ds.map(to_one_hot_val)

    # ---- Model ----
    print("Building model...")
    model, _ = create_model(num_classes)
    model.summary(line_length=80)

    plat = cfg['platform']
    if plat == 'linux':
        optimizer = keras.optimizers.AdamW(
            learning_rate=cfg['learning_rate'], weight_decay=1e-5, clipnorm=1.0)
    else:
        try:
            optimizer = keras.optimizers.legacy.Adam(
                learning_rate=cfg['learning_rate'], clipnorm=1.0)
        except AttributeError:
            optimizer = keras.optimizers.Adam(
                learning_rate=cfg['learning_rate'], clipnorm=1.0)

    if mixup_alpha is not None:
        loss_fn = keras.losses.CategoricalCrossentropy()
    else:
        loss_fn = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    cbs = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(out_dir / 'best.weights.h5'), monitor='val_accuracy',
            save_best_only=True, save_weights_only=True, verbose=0),
    ]

    print("\nTraining...")
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg['num_epochs'], callbacks=cbs, verbose=1
    )
    train_time_min = (time.time() - t0) / 60

    # Load best weights before evaluation
    best_w = out_dir / 'best.weights.h5'
    if best_w.exists():
        model.load_weights(str(best_w))

    # ---- Evaluate FP32 ----
    if mixup_alpha is not None:
        model.compile(
            optimizer=model.optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    print("\nEvaluating FP32...")
    _, fp32_acc = model.evaluate(test_ds, verbose=0)

    y_true, y_pred = [], []
    for bx, by in test_ds:
        preds = model.predict(bx, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(by.numpy())

    fp32_report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(out_dir / 'classification_report_fp32.txt', 'w') as f:
        f.write(fp32_report)

    cm_fp32 = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_fp32, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (FP32)')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix_fp32.png', dpi=150)
    plt.close()

    # ---- Save FP32 model ----
    keras_path = out_dir / 'model_fp32.keras'
    model.save(str(keras_path))
    keras_kb = keras_path.stat().st_size / 1024

    # ---- TFLite INT8 ----
    print("\nExporting INT8 TFLite...")
    tflite_path = out_dir / 'model_int8.tflite'
    tflite_bytes = export_tflite_int8(
        model,
        make_tf_dataset(file_lists['train'][:200], cfg['n_mels'],
                        augment=False, batch_size=32, shuffle=False, seed=None),
        tflite_path
    )

    int8_acc = None
    int8_report = None
    if tflite_bytes is not None:
        tflite_kb = tflite_bytes / 1024
        print(f"  INT8 TFLite: {tflite_kb:.1f} KB  (H7 limit 512 KB — NOT deployable)")
        print("\nEvaluating INT8...")
        yt, yp = eval_tflite(tflite_path, test_ds)
        int8_acc = np.mean(yt == yp)
        int8_report = classification_report(yt, yp, target_names=classes, digits=4)
        with open(out_dir / 'classification_report_int8.txt', 'w') as f:
            f.write(int8_report)
        cm_int8 = confusion_matrix(yt, yp)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_int8, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix (INT8)')
        plt.tight_layout()
        plt.savefig(out_dir / 'confusion_matrix_int8.png', dpi=150)
        plt.close()
    else:
        tflite_kb = 0.0

    # ---- Training history plot ----
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'training_history.png', dpi=150)
    plt.close()

    # ---- Training report ----
    epochs_ran = len(history.history['loss'])
    best_val_acc = max(history.history['val_accuracy'])
    hh = int(train_time_min * 60) // 3600
    mm = (int(train_time_min * 60) % 3600) // 60
    ss = int(train_time_min * 60) % 60

    report_lines = [
        "=" * 80,
        "MYGARDENBIRD EfficientNetB0 TRAINING REPORT (Model 1n)",
        "=" * 80,
        f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"TensorFlow: {tf.__version__}",
        f"Keras: {keras.__version__}",
        "",
        "=" * 80,
        "HYPERPARAMETERS",
        "=" * 80,
        "",
        "Model Architecture:",
        "  Model Type:             EfficientNetB0 (pretrained ImageNet) — Model 1n",
        "  Backbone:               EfficientNetB0 (include_top=False, pooling='avg')",
        f"  Feature extraction:     mel {cfg['n_mels']}→dB→crop/pad 224×224→3ch→p2/p98→EffNetPrep",
        f"  Head:                   Dropout({DROPOUT1}) → Dense(256,relu) → Dropout({DROPOUT2}) → Dense({num_classes})",
        "  MCU Deployable:         NO (EfficientNetB0 ~5MB INT8, H7 limit 512KB)",
        "",
        "Training Configuration:",
        f"  Random Seed:            {cfg['random_seed']}",
        f"  Max Epochs:             {cfg['num_epochs']}",
        f"  Epochs Completed:       {epochs_ran}",
        f"  Batch Size:             {cfg['batch_size']}",
        f"  Learning Rate:          {cfg['learning_rate']}",
        "  LR Schedule:            ReduceLROnPlateau (factor=0.5, patience=5)",
        "  Early Stopping:         patience=10 (val_loss)",
        "",
        "Data Augmentation:",
        f"  Mode:                   {cfg['aug_mode']}",
        "",
        "Data Paths:",
        f"  Splits CSV:             {cfg['splits_csv']}",
        f"  Flat Dir:               {cfg['flat_dir']}",
        f"  Output Directory:       {cfg['output_dir']}",
        "",
        "=" * 80,
        "DATASET INFORMATION",
        "=" * 80,
        "",
        f"Total Samples:          {sum(len(v) for v in file_lists.values())}",
        f"Number of Classes:      {num_classes}",
        f"Classes:                {classes}",
        "",
        "Data Split:",
        f"  Training:             {len(file_lists['train'])}",
        f"  Validation:           {len(file_lists['val'])}",
        f"  Test:                 {len(file_lists['test'])}",
        "",
        "=" * 80,
        "MODEL ARCHITECTURE",
        "=" * 80,
        "",
        f"Total Parameters:       {model.count_params():,}",
        f"  INT8 (1 byte/param):  {model.count_params() / 1024:.1f} KB",
        "",
        "=" * 80,
        "TRAINING",
        "=" * 80,
        "",
        f"Training duration: {hh}h {mm}m {ss}s",
        f"Best val accuracy: {best_val_acc * 100:.2f}%",
        f"Final val accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%",
        "",
        "=" * 80,
        "EVALUATION RESULTS",
        "=" * 80,
        "",
        f"EVALUATION: FP32",
        f"FP32 Accuracy: {fp32_acc * 100:.2f}%",
    ]
    if int8_acc is not None:
        report_lines += [
            f"EVALUATION: INT8 TFLite",
            f"INT8 TFLite Accuracy: {int8_acc * 100:.2f}%",
            "",
            f"FP32 Accuracy:           {fp32_acc * 100:.2f}%",
            f"INT8 Accuracy:           {int8_acc * 100:.2f}%",
        ]
    report_lines += [
        "",
        "Model Sizes:",
        f"  FP32 (.keras)            : {keras_kb / 1024:.2f} MB",
        f"  INT8 (.tflite)           : {tflite_kb:.1f} KB",
        "",
        "Note: EfficientNetB0 is NOT MCU-deployable (~5MB INT8, H7 limit 512KB)",
        "Note: INT8 accuracy may be degraded (swish activations, large dynamic range)",
    ]

    with open(out_dir / 'training_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE — Model 1n (EfficientNetB0 pretrained)")
    print(f"  FP32 Test Accuracy:  {fp32_acc * 100:.2f}%")
    if int8_acc is not None:
        print(f"  INT8 Test Accuracy:  {int8_acc * 100:.2f}%  ({tflite_kb:.1f} KB)")
    print(f"  Training Time:       {hh}h {mm}m {ss}s")
    print(f"  Results saved to:    {cfg['output_dir']}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
