import os
# Suppress TensorFlow GPU/device info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=warning, 2=error, 3=critical

import time
import tensorflow as tf
from utils import format_time

start_time = time.time()

print("Loading MNIST dataset...")
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

print("Preparing data...")
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

print("Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

print("Training (1 epoch)…")
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)

print("Training complete.\n")

total_runtime_string = format_time(time.time() - start_time)
print(f"Total Runtime: {total_runtime_string}\n")