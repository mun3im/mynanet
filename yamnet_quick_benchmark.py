#!/usr/bin/env python3
"""
YAMNet Quick Benchmark - Sliding Window Transfer Learning
===========================================================

Faster approach: Use YAMNet embeddings + simple KNN/SVM classifier
(No expensive fine-tuning, just transfer learning evaluation)

Window positions (equidistant, 5 windows):
  Window 1: frames 0-95, Window 2: frames 51-146, ..., Window 5: frames 204-299
"""

import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import librosa.feature
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SPLITS_CSV = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
N_MELS = 64
SAMPLE_RATE = 16000
AUDIO_LENGTH_S = 3.0
TARGET_SHAPE = (300, N_MELS)

YAMNET_WINDOW_SIZE = 96
WINDOW_STARTS = [0, 51, 102, 153, 204]

SPECIES_MAP = {
    'Asian Koel': 0, 'Coppersmith Barbet': 1, 'Common Iora': 2,
    'Common Tailorbird': 3, 'Pied Fantail': 4, 'Spotted Dove': 5,
    'White-breasted Waterhen': 6, 'White-throated Kingfisher': 7,
    'Collared Kingfisher': 8, 'Large-tailed Nightjar': 9,
    'Yellow-vented Bulbul': 10, 'Olive-backed Sunbird': 11
}

def load_yamnet_model():
    print("Loading YAMNet from TensorFlow Hub...")
    return hub.load("https://tfhub.dev/google/yamnet/1")

def load_data_split(splits_csv, dataset_dir, split_name):
    """Load files for a specific split."""
    files = []
    labels = []

    file_map = {}
    for species_folder, label in SPECIES_MAP.items():
        species_dir = os.path.join(dataset_dir, species_folder)
        if os.path.isdir(species_dir):
            for fname in os.listdir(species_dir):
                if fname.endswith('.wav'):
                    file_id = fname.replace('.wav', '')
                    file_map[file_id] = (species_folder, label)

    with open(splits_csv, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            file_id, split = parts[0], parts[1]
            if split != split_name or file_id not in file_map:
                continue
            species_folder, label = file_map[file_id]
            audio_path = os.path.join(dataset_dir, species_folder, f"{file_id}.wav")
            if os.path.exists(audio_path):
                files.append(audio_path)
                labels.append(label)

    return files, np.array(labels)

def load_audio_mel(path):
    """Load audio and convert to mel spectrogram."""
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, duration=AUDIO_LENGTH_S)
        target_len = int(SAMPLE_RATE * AUDIO_LENGTH_S)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
        return librosa.power_to_db(mel, ref=np.max)
    except Exception as e:
        print(f"  Error: {path}: {e}")
        return None

def extract_windows(mel_spec):
    """Extract 5 overlapping windows."""
    windows = []
    for start in WINDOW_STARTS:
        end = start + YAMNET_WINDOW_SIZE
        windows.append(mel_spec[:, start:end])
    return windows

def extract_embeddings(yamnet_model, mel_spec):
    """Extract aggregated YAMNet embedding from 5 windows."""
    windows = extract_windows(mel_spec)
    embeddings_5window = []

    for window in windows:
        window_input = np.expand_dims(window.T, axis=0)  # (64,96) -> (1,96,64)
        try:
            _, embedding, _ = yamnet_model(window_input)
            embedding_flat = embedding.numpy()[0]  # (128,)
            embeddings_5window.append(embedding_flat)
        except:
            continue

    if len(embeddings_5window) == 0:
        return None

    # Aggregate: mean of 5 windows
    return np.mean(embeddings_5window, axis=0)

def main():
    print("="*70)
    print("YAMNet Quick Benchmark (Sliding Window Transfer Learning)")
    print("="*70)

    yamnet_model = load_yamnet_model()
    print("✓ YAMNet loaded\n")

    # Load data
    print("Loading splits...")
    train_files, train_labels = load_data_split(SPLITS_CSV, DATASET_DIR, 'train')
    val_files, val_labels = load_data_split(SPLITS_CSV, DATASET_DIR, 'val')
    test_files, test_labels = load_data_split(SPLITS_CSV, DATASET_DIR, 'test')
    print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}\n")

    # Extract embeddings
    def extract_split_embeddings(files, labels, split_name):
        embeddings = []
        valid_labels = []
        for i, fpath in enumerate(files):
            if (i + 1) % max(1, len(files)//5) == 0:
                print(f"  {split_name}: {i+1}/{len(files)}")
            mel = load_audio_mel(fpath)
            if mel is None:
                continue
            emb = extract_embeddings(yamnet_model, mel)
            if emb is not None:
                embeddings.append(emb)
                valid_labels.append(labels[i])
        return np.array(embeddings), np.array(valid_labels)

    print("Extracting embeddings (5-window aggregation)...")
    X_train, y_train = extract_split_embeddings(train_files, train_labels, "Train")
    X_val, y_val = extract_split_embeddings(val_files, val_labels, "Val")
    X_test, y_test = extract_split_embeddings(test_files, test_labels, "Test")
    print(f"  Extracted: {len(X_train)}, {len(X_val)}, {len(X_test)}\n")

    # Train classifier
    print("Training Random Forest classifier on YAMNet embeddings...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_val = clf.predict(X_val_scaled)
    y_pred_test = clf.predict(X_test_scaled)

    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

    print("\n" + "="*70)
    print("Results: YAMNet Embeddings + Random Forest")
    print("="*70)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy:       {test_acc*100:.2f}%")
    print(f"Test F1-Score:       {test_f1:.4f}\n")

    # Compare to MynaNet
    mynanet_acc = 0.9491
    gap = (mynanet_acc - test_acc) * 100
    print(f"Comparison to MynaNet (1j, INT8 on same dataset):")
    print(f"  YAMNet (transfer):   {test_acc*100:.2f}%")
    print(f"  MynaNet (1j, INT8):  {mynanet_acc*100:.2f}%")
    print(f"  Gap:                 {gap:+.2f} pp\n")

    if gap > 0:
        print(f"→ YAMNet is {gap:.2f}pp below MynaNet")
        print("  (Domain-specific architecture beats general transfer learning)")
    else:
        print(f"→ YAMNet exceeds MynaNet by {-gap:.2f}pp")

    # Classification report
    print(f"\nPer-class accuracy (test set):")
    species_names = sorted(SPECIES_MAP.keys(), key=SPECIES_MAP.get)
    print(classification_report(y_test, y_pred_test, target_names=species_names, digits=3))

    # Save results
    results_file = "/home/muneim/Dropbox/Conda/mynanet/yamnet_quick_results.csv"
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Training Dataset', 'Audio Duration', 'Approach', 'Test Accuracy', 'Test F1', 'Notes'])
        writer.writerow(['YAMNet + RF', 'MyGardenBird 12-class', '3s', 'Transfer learning (AudioSet→birds)', f'{test_acc:.4f}', f'{test_f1:.4f}', 'Pre-trained embeddings + Random Forest'])
        writer.writerow(['MynaNet (1j, INT8)', 'MyGardenBird 12-class', '3s', 'Domain-specific, trained from scratch', '0.9491', '—', 'Deployed model'])
        writer.writerow(['', '', '', '', '', '', ''])
        writer.writerow(['Analysis', '', '', '', '', '', ''])
        writer.writerow(['Accuracy gap', '', '', '', f'{-gap:.2f} pp', '', 'MynaNet advantage over transfer'])

    print(f"\n✓ Results saved to {results_file}")

if __name__ == "__main__":
    main()
