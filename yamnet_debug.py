#!/usr/bin/env python3
"""Debug YAMNet input/output shapes."""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import librosa.feature

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✓ YAMNet loaded\n")

# Create test mel spectrogram
print("Creating test 64×96 mel spectrogram (1 second)...")
mel_test = np.random.randn(64, 96).astype(np.float32)

# Try different input shapes
print("\nTesting different input shapes:")

# Shape 1: (1, 96, 64) - batch, time, mels
print("\nInput shape: (1, 96, 64) [batch=1, time=96, mels=64]")
try:
    input_1 = np.expand_dims(mel_test.T, axis=0)
    print(f"  Actual input shape: {input_1.shape}")
    scores, embedding, spectrogram = yamnet_model(input_1)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Spectrogram shape: {spectrogram.shape}")
    print(f"  ✓ Success!")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Shape 2: (1, 64, 96) - batch, mels, time
print("\nInput shape: (1, 64, 96) [batch=1, mels=64, time=96]")
try:
    input_2 = np.expand_dims(mel_test, axis=0)
    print(f"  Actual input shape: {input_2.shape}")
    scores, embedding, spectrogram = yamnet_model(input_2)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Spectrogram shape: {spectrogram.shape}")
    print(f"  ✓ Success!")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Shape 3: (96, 64) - time, mels
print("\nInput shape: (96, 64) [time=96, mels=64]")
try:
    input_3 = mel_test.T
    print(f"  Actual input shape: {input_3.shape}")
    scores, embedding, spectrogram = yamnet_model(input_3)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Spectrogram shape: {spectrogram.shape}")
    print(f"  ✓ Success!")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Try with actual audio
print("\n" + "="*60)
print("Testing with real audio file...")
audio_path = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz/Pied Fantail/xc1004365_10651.wav"
try:
    y, _ = librosa.load(audio_path, sr=16000, duration=3.0)
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(f"Mel spec shape: {mel_db.shape} (should be 64×300 for 3s)")

    # Extract 1st window (frames 0-95)
    window = mel_db[:, 0:96]
    print(f"Window shape: {window.shape}")

    # Try shape (1, 96, 64)
    window_input = np.expand_dims(window.T, axis=0)
    print(f"Input shape: {window_input.shape}")
    scores, embedding, spectrogram = yamnet_model(window_input)
    print(f"✓ YAMNet succeeded!")
    print(f"  Scores: {scores.shape} (class logits)")
    print(f"  Embedding: {embedding.shape} (feature vector)")
    print(f"  First 10 embedding values: {embedding.numpy().flatten()[:10]}")

except FileNotFoundError:
    print(f"✗ Audio file not found")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
