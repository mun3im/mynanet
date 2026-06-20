#!/usr/bin/env python3
"""
Convert Keras model to INT8 TFLite v2 format for Arduino compatibility.

TFLite v3 (TensorFlow 2.13+) is not compatible with older Arduino TFLite Micro.
This script forces v2 schema output for maximum Arduino compatibility.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path

# Audio processing parameters (matching Stage9c)
TARGET_SR = 16000
FIXED_AUDIO_LENGTH = 48000  # 3 seconds
N_FFT = 512
HOP_LENGTH = 160
FMAX = 8000
N_CALIBRATION_SAMPLES = 200

def load_calibration_data(data_dir, splits_csv, n_mels, n_samples=200):
    """
    Load calibration samples from test set for quantization.

    Args:
        data_dir: Path to flat audio directory
        splits_csv: Path to CSV with train/val/test splits
        n_mels: Number of mel bins
        n_samples: Number of calibration samples to use
    """
    print(f"\n{'='*70}")
    print("LOADING CALIBRATION DATA")
    print(f"{'='*70}")

    # Parse splits CSV to get test files
    import csv
    test_files = set()
    with open(splits_csv, 'r') as f:
        # Skip comment lines
        lines = [line for line in f if not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            if row['split'].lower() == 'test':
                # Handle both formats: file_id (XC123) and filename (xc123.wav)
                if 'file_id' in row:
                    fid = row['file_id']
                    if fid.startswith('XC') or fid.startswith('xc'):
                        wav = fid.lower() + '.wav'
                    else:
                        wav = fid
                else:
                    wav = row['filename']
                test_files.add(wav)

    print(f"Found {len(test_files)} test files in splits CSV")

    # Compute global stats
    print(f"\nComputing global normalization stats (n_mels={n_mels})...")
    all_mel = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue

        wavs = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        for f in wavs[:10]:  # Sample 10 per class for stats
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

                # Ensure exactly 300 frames
                if mel_db.shape[1] > 300:
                    mel_db = mel_db[:, :300]
                elif mel_db.shape[1] < 300:
                    mel_db = np.pad(mel_db, ((0, 0), (0, 300 - mel_db.shape[1])), mode='constant')

                all_mel.append(mel_db.flatten())
            except Exception as e:
                continue

    if len(all_mel) == 0:
        raise RuntimeError("No valid audio files found for stats")

    all_mel = np.concatenate(all_mel)
    gmin, gmax = np.percentile(all_mel, 2), np.percentile(all_mel, 98)
    print(f"✓ Global stats: {gmin:.2f} → {gmax:.2f} dB")

    # Load calibration spectrograms from test set
    print(f"\nLoading {n_samples} calibration spectrograms from test set...")
    calibration_data = []
    count = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue

        wavs = [f for f in os.listdir(class_dir) if f.endswith('.wav') and f in test_files]

        for f in wavs:
            if count >= n_samples:
                break

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

                # Ensure exactly 300 frames
                if mel_db.shape[1] > 300:
                    mel_db = mel_db[:, :300]
                elif mel_db.shape[1] < 300:
                    mel_db = np.pad(mel_db, ((0, 0), (0, 300 - mel_db.shape[1])), mode='constant')

                # Normalize
                mel_norm = (mel_db - gmin) / (gmax - gmin + 1e-8)
                mel_norm = np.clip(mel_norm, 0, 1)

                # Add channel dimension
                mel_norm = mel_norm[..., np.newaxis]
                calibration_data.append(mel_norm)
                count += 1

            except Exception as e:
                print(f"⚠ Failed to load {f}: {e}")
                continue

        if count >= n_samples:
            break

    print(f"✓ Loaded {len(calibration_data)} calibration samples")
    print(f"  Shape: {calibration_data[0].shape}")

    return np.array(calibration_data, dtype=np.float32)


def convert_to_tflite_v2(keras_model_path, output_path, calibration_data):
    """
    Convert Keras model to INT8 TFLite with v2 schema for Arduino compatibility.

    Args:
        keras_model_path: Path to .keras model file
        output_path: Path to save .tflite file
        calibration_data: Numpy array of calibration samples
    """
    print(f"\n{'='*70}")
    print("CONVERTING TO INT8 TFLITE V2 (ARDUINO COMPATIBLE)")
    print(f"{'='*70}")

    # Load Keras model
    print(f"\nLoading Keras model: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    print(f"✓ Model loaded")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Create representative dataset for quantization
    def representative_dataset():
        for i in range(len(calibration_data)):
            yield [calibration_data[i:i+1].astype(np.float32)]

    # Convert with INT8 quantization
    print(f"\nConfiguring TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Force old converter for v2 schema compatibility
    # This ensures compatibility with Arduino TFLite Micro
    converter._experimental_new_converter = False

    # INT8 quantization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(f"  Optimization: INT8 post-training quantization")
    print(f"  Calibration samples: {len(calibration_data)}")
    print(f"  Using old converter (forces v2 schema)")

    # Convert
    print(f"\nConverting... (this may take a few minutes)")
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✓ Conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_kb:.1f} KB")

    # Check schema version
    print(f"\nChecking TFLite schema version...")
    try:
        # Try to read the flatbuffer to check version
        with open(output_path, 'rb') as f:
            content = f.read()
            # TFLite v3 has signature "TFL3" at specific offset
            # TFLite v2 typically doesn't have this marker
            if b'TFL3' in content[:1000]:
                print(f"  ⚠ WARNING: Schema appears to be v3 (may not work on Arduino)")
                print(f"  Recommendation: Use TensorFlow 2.4-2.12 for guaranteed v2 output")
            else:
                print(f"  ✓ Schema appears to be v2 (Arduino compatible)")
    except Exception as e:
        print(f"  Unable to verify schema version: {e}")

    return size_kb


def verify_tflite_model(tflite_path, test_sample):
    """Quick test to verify the TFLite model works."""
    print(f"\n{'='*70}")
    print("VERIFYING TFLITE MODEL")
    print(f"{'='*70}")

    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\n✓ Model loaded successfully")
        print(f"  Input details:")
        print(f"    Shape: {input_details[0]['shape']}")
        print(f"    Type: {input_details[0]['dtype']}")
        print(f"    Quantization: scale={input_details[0]['quantization'][0]:.6f}, "
              f"zero_point={input_details[0]['quantization'][1]}")

        print(f"  Output details:")
        print(f"    Shape: {output_details[0]['shape']}")
        print(f"    Type: {output_details[0]['dtype']}")
        print(f"    Quantization: scale={output_details[0]['quantization'][0]:.6f}, "
              f"zero_point={output_details[0]['quantization'][1]}")

        # Test inference
        print(f"\n  Testing inference with sample data...")
        input_scale, input_zero_point = input_details[0]['quantization']
        x_int8 = (test_sample / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], x_int8)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"  ✓ Inference successful")
        print(f"    Output shape: {output.shape}")
        print(f"    Output sample: {output[0][:5]}...")

        return True

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_tflite_v2.py <results_dir> [--n_mels 64]")
        print("\nExample:")
        print("  python convert_to_tflite_v2.py results_linux/v1_dscnn_se_res_att_wide_mels64_drop05_rand100_warm100_mixup0.25_split80:10:10_linux")
        print("  python convert_to_tflite_v2.py results_linux/v1_dscnn_se_res_att_wide_mels64_drop05_rand100_warm100_mixup0.25_split80:10:10_linux --n_mels 64")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Parse n_mels from command line or directory name
    n_mels = 80  # default
    if '--n_mels' in sys.argv:
        idx = sys.argv.index('--n_mels')
        n_mels = int(sys.argv[idx + 1])
    elif 'mels64' in results_dir:
        n_mels = 64
    elif 'mels80' in results_dir:
        n_mels = 80

    print(f"\n{'='*70}")
    print(f"KERAS → INT8 TFLITE V2 CONVERTER")
    print(f"{'='*70}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Results directory: {results_dir}")
    print(f"Mel bins: {n_mels}")

    # Paths
    keras_model_path = os.path.join(results_dir, 'model_fp32.keras')
    output_path = os.path.join(results_dir, 'model_int8_v2.tflite')

    # Check if model exists
    if not os.path.exists(keras_model_path):
        print(f"\n✗ Error: Model not found at {keras_model_path}")
        sys.exit(1)

    # Data paths (adjust these to your setup)
    data_dir = '/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz'
    splits_csv = '/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv'

    # Check if data paths exist
    if not os.path.exists(data_dir):
        print(f"\n✗ Error: Data directory not found: {data_dir}")
        print(f"Please update data_dir in the script to point to your audio files")
        sys.exit(1)

    if not os.path.exists(splits_csv):
        print(f"\n✗ Error: Splits CSV not found: {splits_csv}")
        print(f"Please update splits_csv in the script")
        sys.exit(1)

    # Load calibration data
    calibration_data = load_calibration_data(data_dir, splits_csv, n_mels, N_CALIBRATION_SAMPLES)

    # Convert
    size_kb = convert_to_tflite_v2(keras_model_path, output_path, calibration_data)

    # Verify
    verify_tflite_model(output_path, calibration_data[0:1])

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Conversion complete")
    print(f"  Input:  {keras_model_path}")
    print(f"  Output: {output_path}")
    print(f"  Size:   {size_kb:.1f} KB")
    print(f"\nReady for Arduino deployment!")
    print(f"\nNote: If Arduino TFLite Micro still reports version errors,")
    print(f"      try using TensorFlow 2.4-2.12 for guaranteed v2 output.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
