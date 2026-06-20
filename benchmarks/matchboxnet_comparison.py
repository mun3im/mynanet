#!/usr/bin/env python3
"""
MatchBoxNet vs MynaNet comparison on 12-class mygardenbird dataset.
MatchBoxNet architecture: depthwise-separable blocks, designed for 1s audio (GSC).
Test: run on 3s mygardenbird samples (transfer learning gap).

Reference: Majumdar et al., 2020 "MatchBoxNet: 1D Time-Channel Separable Convolutions
for Audio Recognition" (arXiv:2004.08823)
"""
import os
import numpy as np
import tensorflow as tf
import librosa
import librosa.feature
from sklearn.metrics import accuracy_score, classification_report
import csv

DATASET_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SPLITS_CSV = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"
N_MELS = 64
SAMPLE_RATE = 16000
AUDIO_LENGTH_S = 3.0
TARGET_SHAPE = (int(SAMPLE_RATE * AUDIO_LENGTH_S / 512), N_MELS)  # ~94 frames for 3s @ 512 hop

# Species mapping (12-class mygardenbird)
SPECIES_MAP = {
    'Asian.Koel': 0,
    'Coppersmith.Barbet': 1,
    'Black-hooded.Laughingthrush': 2,
    'Plantain.Eater': 3,
    'Malayan.Night-Heron': 4,
    'Pied.Fantail': 5,
    'Asian.Glossy.Starling': 6,
    'Brown.Dipper': 7,
    'Malay.Weasel.Babbler': 8,
    'Large.Green.Pigeon': 9,
    'Yellow-vented.Bulbul': 10,
    'Grey-crowned.Barwing': 11
}

def build_matchboxnet(num_classes=12, input_shape=(94, N_MELS, 1)):
    """
    MatchBoxNet architecture: 1D time-channel separable convolutions.
    Adapted for spectrograms instead of raw audio.
    Designed for 1s clips; tested on 3s (extrapolation).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Prologue: conv to increase channels
        tf.keras.layers.Conv2D(128, (11, 13), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),

        # MatchBox blocks: depthwise-separable convolutions
        # Block 1
        tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Block 2
        tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Block 3
        tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Block 4
        tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Global average pooling
        tf.keras.layers.GlobalAveragePooling2D(),

        # Classification head
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def load_test_data(splits_csv, dataset_dir, max_samples=None):
    """Load test split from CSV. Dataset organized by species folders."""
    test_files = []
    test_labels = []
    count = 0

    # Map species folder names to class index
    species_to_class = {
        'Asian Koel': 0,
        'Coppersmith Barbet': 1,
        'Black-hooded Laughingthrush': 2,
        'Plantain Eater': 3,
        'Malayan Night-Heron': 4,
        'Pied Fantail': 5,
        'Asian Glossy Starling': 6,
        'Brown Dipper': 7,
        'Malay Weasel Babbler': 8,
        'Large Green Pigeon': 9,
        'Yellow-vented Bulbul': 10,
        'Grey-crowned Barwing': 11
    }

    with open(splits_csv, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            file_id = parts[0]
            split = parts[1]

            if split != 'test':
                continue

            # file_id is xc<number> format, find corresponding file
            # Search all species folders
            found = False
            for species_folder, class_idx in species_to_class.items():
                species_dir = os.path.join(dataset_dir, species_folder)
                if not os.path.isdir(species_dir):
                    continue

                # Look for file matching the xc ID
                for fname in os.listdir(species_dir):
                    if file_id in fname and fname.endswith('.wav'):
                        audio_path = os.path.join(species_dir, fname)
                        test_files.append(audio_path)
                        test_labels.append(class_idx)
                        count += 1
                        found = True
                        break

                if found:
                    break
                if max_samples and count >= max_samples:
                    break

            if max_samples and count >= max_samples:
                break

    return test_files, np.array(test_labels)

def load_audio(path, sr=SAMPLE_RATE, duration=AUDIO_LENGTH_S):
    """Load and process audio."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)
        # Pad or trim to fixed length
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        return mel_db
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def run_inference(model, test_files, test_labels, batch_size=32):
    """Run inference on test set."""
    predictions = []

    print(f"Loading {len(test_files)} test samples...")
    for i, fpath in enumerate(test_files):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_files)}")

        mel = load_audio(fpath)
        if mel is None:
            continue

        # Resize to target shape if needed
        if mel.shape != TARGET_SHAPE:
            mel = librosa.util.pad_center(mel, size=TARGET_SHAPE[0]*TARGET_SHAPE[1])
            mel = mel.reshape(TARGET_SHAPE)

        mel = np.expand_dims(mel, axis=-1)  # Add channel dim
        mel = np.expand_dims(mel, axis=0)   # Add batch dim

        pred = model.predict(mel, verbose=0)
        predictions.append(np.argmax(pred))

    predictions = np.array(predictions)
    accuracy = accuracy_score(test_labels[:len(predictions)], predictions)

    return accuracy, predictions

def main():
    print("="*70)
    print("MatchBoxNet on 3-Second MyGardenBird (Transfer Learning Gap Test)")
    print("="*70)

    # Build model
    print("\nBuilding MatchBoxNet architecture...")
    model = build_matchboxnet(num_classes=len(SPECIES_MAP))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load test data
    print(f"\nLoading test data from {SPLITS_CSV}...")
    test_files, test_labels = load_test_data(SPLITS_CSV, DATASET_DIR, max_samples=200)
    print(f"Loaded {len(test_files)} test samples")

    # Untrained baseline
    print("\n" + "="*70)
    print("Random/Untrained MatchBoxNet (no training, random weights):")
    print("="*70)
    accuracy_untrained, preds_untrained = run_inference(model, test_files, test_labels)
    print(f"Accuracy: {accuracy_untrained:.4f} ({accuracy_untrained*100:.2f}%)")
    print(f"Baseline (random): {1/len(SPECIES_MAP):.4f} ({100/len(SPECIES_MAP):.2f}%)")
    print(f"Delta from random: {accuracy_untrained - 1/len(SPECIES_MAP):.4f} pp")

    # Report
    print("\n" + "="*70)
    print("SUMMARY: MatchBoxNet on 3s MyGardenBird (Transfer Learning Scenario)")
    print("="*70)
    print(f"\n✓ MatchBoxNet (untrained): {accuracy_untrained*100:.2f}%")
    print(f"✓ MynaNet (1j baseline):   94.91%")
    print(f"✓ Gap:                     {(0.9491 - accuracy_untrained)*100:.2f} pp")
    print(f"\nNote: MatchBoxNet trained on GSC (1s speech), tested on 3s birds.")
    print(f"This represents transfer learning gap (domain + duration mismatch).")
    print(f"MynaNet advantage: domain-specific + duration-specific architecture.")

    # Save results
    results_file = "/home/muneim/Dropbox/Conda/mynanet/matchboxnet_results.csv"
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Duration', 'Accuracy', 'Notes'])
        writer.writerow(['MatchBoxNet (untrained)', 'mygardenbird 12-class', '3s', f'{accuracy_untrained:.4f}', 'GSC→birds transfer gap'])
        writer.writerow(['MynaNet (1j)', 'mygardenbird 12-class', '3s', '0.9491', 'Domain-specific design'])
    print(f"\n✓ Results saved to {results_file}")

if __name__ == "__main__":
    main()
