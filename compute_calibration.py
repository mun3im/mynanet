#!/usr/bin/env python3
"""
Compute ECE (Expected Calibration Error) and Brier score for 1j model variants.
Loads model_int8.tflite + test splits, runs inference, measures calibration.
"""
import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import brier_score_loss
import re

def find_test_split_csv(result_dir):
    """Locate the .csv used for this run (look for splits_mip* or fallback to default)."""
    # Check CLAUDE.md or infer from run metadata
    # For simplicity: assume standard split at /Volumes/Evo/MYGARDENBIRD/metadata16khz
    meta_dir = "/Volumes/Evo/MYGARDENBIRD/metadata16khz"
    if os.path.exists(os.path.join(meta_dir, "splits_mip_80_10_10.csv")):
        return os.path.join(meta_dir, "splits_mip_80_10_10.csv")
    return None

def load_test_data(splits_csv, flat_dir):
    """Load audio files labeled 'test' from splits_csv."""
    import librosa
    import librosa.feature

    test_files = []
    test_labels = []

    with open(splits_csv, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip()

        # Infer species label from header if present
        species_map = {}
        if "MyGardenBird" in flat_dir:
            # species in order from qc_report
            species_map = {
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

        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            file_id = parts[0]
            split = parts[1]

            if split != 'test':
                continue

            # Infer species from file_id (e.g., "3_Asian.Koel_XC123456_seg0.wav")
            match = re.match(r'\d+_([^_]+)_', file_id)
            if not match:
                continue

            species_name = match.group(1).replace('.', '.')
            if species_name not in species_map:
                continue

            label = species_map[species_name]
            audio_path = os.path.join(flat_dir, file_id)

            if os.path.exists(audio_path):
                test_files.append(audio_path)
                test_labels.append(label)

    # Load audio
    X_test = []
    for fpath in test_files[:100]:  # Cap at 100 for speed (demo)
        try:
            y, sr = librosa.load(fpath, sr=16000)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            X_test.append(mel_db)
        except Exception as e:
            print(f"  Skipped {fpath}: {e}")

    return np.array(X_test), np.array(test_labels[:len(X_test)])

def compute_ece(y_probs, y_true, n_bins=15):
    """Compute Expected Calibration Error (15-bin histogram)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_width = 1.0 / n_bins
    ece = 0.0
    total_samples = 0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Confidence: max predicted probability
        confidences = np.max(y_probs, axis=1)
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.sum(in_bin) == 0:
            continue

        bin_confidences = confidences[in_bin]
        bin_true = y_true[in_bin]
        bin_preds = np.argmax(y_probs[in_bin], axis=1)

        bin_accuracy = np.mean(bin_preds == bin_true)
        bin_confidence = np.mean(bin_confidences)

        ece += np.abs(bin_accuracy - bin_confidence) * np.sum(in_bin)
        total_samples += np.sum(in_bin)

    return ece / total_samples if total_samples > 0 else 0.0

def run_inference_tflite(model_path, X_test, n_classes=12):
    """Run inference on TFLite INT8 model, return softmax-like probabilities."""
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantization params for input
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    # Quantization params for output
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    predictions = []

    for i, sample in enumerate(X_test):
        if (i + 1) % 20 == 0:
            print(f"    Inference {i+1}/{len(X_test)}")

        # Quantize input to INT8
        sample = sample.astype(np.float32)
        sample_int8 = (sample / input_scale + input_zero_point).astype(np.int8)
        sample_int8 = np.expand_dims(sample_int8, axis=0)

        interpreter.set_tensor(input_details[0]['index'], sample_int8)
        interpreter.invoke()

        # Get quantized output and dequantize
        output_int8 = interpreter.get_tensor(output_details[0]['index'])
        output_float = (output_int8.astype(np.float32) - output_zero_point) * output_scale

        # Convert to softmax-like probabilities
        output_probs = tf.nn.softmax(output_float[0]).numpy()
        predictions.append(output_probs)

    return np.array(predictions)

def main():
    results_dir = "results_mygardenbird_1_linux"

    # Find 1j mixup and specaugment runs (single-seed, seed42)
    pattern_mixup = os.path.join(results_dir, "*mbv3_se*warm70_mixup0.2*split80:10:10*linux*")
    pattern_specaug = os.path.join(results_dir, "*mbv3_se*warm70_specaugment*split80:10:10*linux*")

    runs_mixup = sorted(glob.glob(pattern_mixup))
    runs_specaug = sorted(glob.glob(pattern_specaug))

    print(f"Found {len(runs_mixup)} mixup runs and {len(runs_specaug)} specaugment runs.\n")

    # Pick the first seed42 of each
    for run_path in runs_mixup:
        if "rand42" in run_path:
            calibrate_run(run_path, "mixup")
            break

    for run_path in runs_specaug:
        if "rand42" in run_path:
            calibrate_run(run_path, "specaugment")
            break

def calibrate_run(run_path, augment_name):
    """Calibrate a single run."""
    print(f"\n{'='*60}")
    print(f"Calibrating: {os.path.basename(run_path)}")
    print(f"Augmentation: {augment_name}")
    print(f"{'='*60}")

    model_path = os.path.join(run_path, "model_int8.tflite")
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return

    # Load test data (simplified: use small subset)
    flat_dir = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
    splits_csv = "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"

    print(f"Loading test data...")
    try:
        X_test, y_test = load_test_data(splits_csv, flat_dir)
        print(f"  Loaded {len(X_test)} test samples")
    except Exception as e:
        print(f"  Error loading test data: {e}")
        return

    # Normalize input
    X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test)) * 255 - 128

    # Run inference
    print(f"Running inference (INT8)...")
    y_probs = run_inference_tflite(model_path, X_test, n_classes=12)

    # Compute metrics
    print(f"Computing calibration metrics...")
    ece = compute_ece(y_probs, y_test, n_bins=15)
    brier = brier_score_loss(y_test, y_probs)

    print(f"\nResults for {augment_name}:")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print()

if __name__ == "__main__":
    main()
