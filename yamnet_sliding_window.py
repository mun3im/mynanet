#!/usr/bin/env python3
"""
YAMNet Sliding Window Benchmark on MyGardenBird
================================================

Approach:
- Load pre-trained YAMNet (Google, 64×96 input for ~1s clips)
- Extract 5 equidistant overlapping windows from 3s clips (64×300 mel specs)
- Run YAMNet embedding layer on each window
- Aggregate embeddings with weighted mean (edges down-weighted)
- Fine-tune 12-class head on aggregated embeddings
- Compare to MynaNet (1j): 94.91% INT8

Window positions (96-frame windows, 51-frame stride for equidistance):
  Window 1: frames 0-95
  Window 2: frames 51-146
  Window 3: frames 102-197
  Window 4: frames 153-248
  Window 5: frames 204-299
"""

import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import librosa.feature
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SPLITS_CSV = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
N_MELS = 64
SAMPLE_RATE = 16000
AUDIO_LENGTH_S = 3.0
TARGET_SHAPE = (300, N_MELS)  # 300 frames × 64 mels for 3s

# YAMNet window config
YAMNET_WINDOW_SIZE = 96
YAMNET_N_MELS = 64
WINDOW_STRIDE = 51  # Equidistant 5 windows: (204-0)/4 = 51

# Window positions (start indices)
WINDOW_STARTS = [0, 51, 102, 153, 204]

# Species mapping (12-class mygardenbird) - folder names
SPECIES_MAP = {
    'Asian Koel': 0,
    'Coppersmith Barbet': 1,
    'Common Iora': 2,
    'Common Tailorbird': 3,
    'Pied Fantail': 4,
    'Spotted Dove': 5,
    'White-breasted Waterhen': 6,
    'White-throated Kingfisher': 7,
    'Collared Kingfisher': 8,
    'Large-tailed Nightjar': 9,
    'Yellow-vented Bulbul': 10,
    'Olive-backed Sunbird': 11
}

def load_yamnet_model():
    """Load pre-trained YAMNet from TensorFlow Hub."""
    print("Loading YAMNet from TensorFlow Hub...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    return yamnet_model

def load_test_data(splits_csv, dataset_dir, max_samples=None):
    """Load test split from CSV. Search for file_id in all species folders."""
    test_files = []
    test_labels = []

    # Build a map of file_id -> (species_folder, label)
    file_map = {}
    for species_folder, label in SPECIES_MAP.items():
        species_dir = os.path.join(dataset_dir, species_folder)
        if os.path.isdir(species_dir):
            for fname in os.listdir(species_dir):
                if fname.endswith('.wav'):
                    file_id = fname.replace('.wav', '')  # Remove .wav extension
                    file_map[file_id] = (species_folder, label)

    with open(splits_csv, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header (includes comment lines starting with #)
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split(',')
            if len(parts) < 2:
                continue
            file_id = parts[0]
            split = parts[1]

            if split != 'test':
                continue

            # Look up file_id in our map
            if file_id not in file_map:
                continue

            species_folder, label = file_map[file_id]
            audio_path = os.path.join(dataset_dir, species_folder, f"{file_id}.wav")

            if os.path.exists(audio_path):
                test_files.append(audio_path)
                test_labels.append(label)

                if max_samples and len(test_files) >= max_samples:
                    break

    return test_files, np.array(test_labels)

def load_audio(path, sr=SAMPLE_RATE, duration=AUDIO_LENGTH_S, n_mels=N_MELS):
    """Load and process audio to mel spectrogram."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)

        # Pad or trim to fixed length
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        return mel_db
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_sliding_windows(mel_spec):
    """
    Extract 5 equidistant overlapping windows from 64×300 mel spec.

    Windows (96-frame size, 51-frame stride):
    - Window 1: frames 0-95
    - Window 2: frames 51-146
    - Window 3: frames 102-197
    - Window 4: frames 153-248
    - Window 5: frames 204-299

    Returns: list of 5 windows, each 64×96
    """
    windows = []
    for start in WINDOW_STARTS:
        end = start + YAMNET_WINDOW_SIZE
        window = mel_spec[:, start:end]  # (n_mels, window_size) = (64, 96)
        windows.append(window)

    return windows

def get_window_weights():
    """
    Compute weights for each window to down-weight edges.
    Edge frames (first/last 24 of each window) get lower weight.
    """
    # For 96-frame window: frames 0-23 and 72-95 are edges
    edge_size = 24
    weights = np.ones(YAMNET_WINDOW_SIZE)
    weights[:edge_size] = 0.5  # down-weight start edge
    weights[-edge_size:] = 0.5  # down-weight end edge
    return weights

def aggregate_embeddings(embeddings, overlap_strategy='weighted_mean'):
    """
    Aggregate 5 window embeddings into single embedding.

    Args:
        embeddings: list of 5 embeddings, each 1024-dim
        overlap_strategy: 'mean', 'weighted_mean', 'max'

    Returns:
        aggregated embedding (1024-dim)
    """
    embeddings = np.array(embeddings)  # (5, 1024)

    if overlap_strategy == 'mean':
        return np.mean(embeddings, axis=0)

    elif overlap_strategy == 'weighted_mean':
        # Weight by position: center windows get higher weight
        position_weights = np.array([0.9, 1.0, 1.1, 1.0, 0.9])  # Window 3 (center) highest
        position_weights /= position_weights.sum()
        return np.average(embeddings, axis=0, weights=position_weights)

    elif overlap_strategy == 'max':
        return np.max(embeddings, axis=0)

    else:
        return np.mean(embeddings, axis=0)

def run_yamnet_inference(yamnet_model, test_files, test_labels, batch_size=32):
    """
    Run YAMNet on test set with sliding windows + aggregation.

    Returns:
        aggregated_embeddings: (n_samples, embedding_dim)
        y_true: (n_samples,) true labels
    """
    all_embeddings = []
    valid_labels = []

    print(f"\nRunning YAMNet sliding window inference on {len(test_files)} samples...")

    for i, fpath in enumerate(test_files):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_files)}")

        # Load and process audio
        mel = load_audio(fpath, sr=SAMPLE_RATE, duration=AUDIO_LENGTH_S, n_mels=N_MELS)
        if mel is None or mel.shape != TARGET_SHAPE:
            continue

        # Extract sliding windows
        windows = extract_sliding_windows(mel)  # 5 × (64, 96)

        # Run YAMNet on each window, extract embeddings
        window_embeddings = []
        for window in windows:
            # YAMNet expects (batch, time, mels) or (batch, mels, time)
            # Input shape: (1, 96, 64) for (batch, time, mels)
            window_input = np.expand_dims(window.T, axis=0)  # (64, 96) -> (96, 64) -> (1, 96, 64)

            # YAMNet returns (scores, embeddings, spectrogram)
            try:
                scores, embedding, _ = yamnet_model(window_input)
                # embedding shape: (1, 128) - YAMNet embedding is 128-dim, not 1024
                embedding_flat = embedding.numpy()[0]  # (128,)
                window_embeddings.append(embedding_flat)
            except Exception as e:
                print(f"    Error processing window: {e}")
                continue

        if len(window_embeddings) < 5:
            continue  # Skip if not all windows processed

        # Aggregate embeddings
        agg_embedding = aggregate_embeddings(window_embeddings, overlap_strategy='weighted_mean')
        all_embeddings.append(agg_embedding)
        valid_labels.append(test_labels[i])

    embeddings_arr = np.array(all_embeddings)
    labels_arr = np.array(valid_labels)

    print(f"\n  Embeddings collected: {len(all_embeddings)} samples")
    print(f"  Embeddings shape: {embeddings_arr.shape}")
    print(f"  Labels shape: {labels_arr.shape}")

    return embeddings_arr, labels_arr

def build_classifier_head(embedding_dim=128, num_classes=12):
    """Build fine-tunable classifier head for aggregated embeddings."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_classifier(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train classifier head on aggregated embeddings."""
    # Debug: check shapes
    print(f"\n  X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Unique classes: {len(np.unique(y_train))}")

    if X_train.ndim != 2:
        print(f"ERROR: X_train must be 2D, got {X_train.ndim}D!")
        raise ValueError(f"X_train shape is {X_train.shape}, expected (n_samples, embedding_dim)")

    model = build_classifier_head(embedding_dim=X_train.shape[1], num_classes=len(np.unique(y_train)))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nTraining classifier on {len(X_train)} training samples...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
    )

    return model, history

def main():
    print("="*70)
    print("YAMNet Sliding Window Benchmark on MyGardenBird (12-class)")
    print("="*70)
    print(f"\nWindow configuration (equidistant, 5 windows):")
    for i, start in enumerate(WINDOW_STARTS, 1):
        end = start + YAMNET_WINDOW_SIZE
        time_start_ms = start * 10
        time_end_ms = end * 10
        print(f"  Window {i}: frames {start:3d}-{end-1:3d} ({time_start_ms:4d}-{time_end_ms:4d} ms)")

    # Load YAMNet
    try:
        yamnet_model = load_yamnet_model()
        print("\n✓ YAMNet loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load YAMNet: {e}")
        print("  Install tensorflow_hub: pip install tensorflow_hub")
        return

    # Load test data
    print(f"\nLoading test data from {SPLITS_CSV}...")
    test_files, test_labels = load_test_data(SPLITS_CSV, DATASET_DIR, max_samples=None)
    print(f"  Loaded {len(test_files)} test samples")

    if len(test_files) == 0:
        print("✗ No test samples loaded!")
        return

    # Run YAMNet inference with sliding windows
    embeddings_test, labels_test = run_yamnet_inference(yamnet_model, test_files, test_labels)

    # Load training data for fine-tuning
    print(f"\nLoading training/validation data...")
    train_files_all = []
    train_labels_all = []
    val_files = []
    val_labels = []

    # Build file map (reuse from test data loading)
    file_map = {}
    for species_folder, label in SPECIES_MAP.items():
        species_dir = os.path.join(DATASET_DIR, species_folder)
        if os.path.isdir(species_dir):
            for fname in os.listdir(species_dir):
                if fname.endswith('.wav'):
                    file_id = fname.replace('.wav', '')
                    file_map[file_id] = (species_folder, label)

    with open(SPLITS_CSV, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split(',')
            if len(parts) < 2:
                continue
            file_id = parts[0]
            split = parts[1]

            if file_id not in file_map:
                continue

            species_folder, label = file_map[file_id]
            audio_path = os.path.join(DATASET_DIR, species_folder, f"{file_id}.wav")

            if not os.path.exists(audio_path):
                continue

            if split == 'train':
                train_files_all.append(audio_path)
                train_labels_all.append(label)
            elif split == 'val':
                val_files.append(audio_path)
                val_labels.append(label)

    train_labels_all = np.array(train_labels_all)
    val_labels = np.array(val_labels)

    print(f"  Training: {len(train_files_all)} samples")
    print(f"  Validation: {len(val_files)} samples")
    print(f"  Test: {len(test_files)} samples")

    # Run YAMNet on training data
    embeddings_train, _ = run_yamnet_inference(yamnet_model, train_files_all, train_labels_all)
    embeddings_val, _ = run_yamnet_inference(yamnet_model, val_files, val_labels)

    # Train classifier
    classifier, history = train_classifier(
        embeddings_train, train_labels_all,
        embeddings_val, val_labels,
        epochs=50,
        batch_size=32
    )

    # Evaluate on test set
    print("\n" + "="*70)
    print("Results: YAMNet + Classifier Head (Fine-tuned)")
    print("="*70)

    y_pred_test = classifier.predict(embeddings_test, verbose=0)
    y_pred_test_labels = np.argmax(y_pred_test, axis=1)

    accuracy = accuracy_score(labels_test, y_pred_test_labels)
    f1_macro = f1_score(labels_test, y_pred_test_labels, average='macro', zero_division=0)

    print(f"\n✓ Test Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Test F1-Score:  {f1_macro:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(labels_test, y_pred_test_labels,
                                target_names=sorted(SPECIES_MAP.keys(), key=SPECIES_MAP.get),
                                digits=4))

    # Compare to MynaNet
    mynanet_acc = 0.9491  # 1j INT8 accuracy
    gap = (mynanet_acc - accuracy) * 100

    print(f"\n{'='*70}")
    print(f"Comparison to MynaNet (1j)")
    print(f"{'='*70}")
    print(f"YAMNet (fine-tuned):  {accuracy*100:.2f}%")
    print(f"MynaNet (1j, INT8):   {mynanet_acc*100:.2f}%")
    print(f"Gap:                  {gap:+.2f} pp")

    if gap > 0:
        print(f"↓ YAMNet is {gap:.2f}pp below MynaNet (domain-specific model advantage)")
    else:
        print(f"↑ YAMNet is {-gap:.2f}pp above MynaNet")

    # Save results
    results_file = "/home/muneim/Dropbox/Conda/mynanet/yamnet_sliding_window_results.csv"
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Training Dataset', 'Audio Duration', 'Task', 'Accuracy', 'F1-Score', 'Notes'])
        writer.writerow(['YAMNet (fine-tuned)', 'MyGardenBird 12-class', '3s', 'Bird classification', f'{accuracy:.4f}', f'{f1_macro:.4f}', 'Pre-trained on AudioSet, fine-tuned on mygardenbird'])
        writer.writerow(['MynaNet (1j, INT8)', 'MyGardenBird 12-class', '3s', 'Bird classification', '0.9491', '—', 'Trained from scratch, deployed model'])
        writer.writerow(['', '', '', '', '', '', ''])
        writer.writerow(['Summary', '', '', '', '', '', ''])
        writer.writerow(['Accuracy gap', '', '', '', f'{-gap:.2f} pp', '', 'MynaNet vs YAMNet'])

    print(f"\n✓ Results saved to {results_file}")

if __name__ == "__main__":
    main()
