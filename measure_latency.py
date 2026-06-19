#!/usr/bin/env python3
"""
Measure single-sample INT8 inference latency of MynaNet on CPU.
Loads deploy/mynanet_int8.tflite, runs N repeated inferences with timer.
Reports mean/median/p95 latency (ms) for single sample.
"""
import os
import time
import numpy as np
import tensorflow as tf

MODEL_PATH = "deploy/mynanet_int8.tflite"
N_SAMPLES = 500

def load_model():
    """Load TFLite INT8 model."""
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        num_threads=1  # single-threaded to match MCU behavior
    )
    interpreter.allocate_tensors()
    return interpreter

def get_input_shape(interpreter):
    """Get input tensor shape."""
    input_details = interpreter.get_input_details()
    return input_details[0]['shape']

def run_inference(interpreter, dummy_input):
    """Run one inference, return runtime in ms."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], dummy_input)

    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    return latency_ms

def main():
    print(f"Loading model from {MODEL_PATH}...")
    interpreter = load_model()

    input_shape = get_input_shape(interpreter)
    print(f"Input shape: {input_shape}")

    # Create a dummy INT8 input (quantized 0-255 range, or actual quantization bounds)
    # For safety, use zeros (neutral input)
    dummy_input = np.zeros(input_shape, dtype=np.int8)

    print(f"\nWarming up (5 runs)...")
    for _ in range(5):
        run_inference(interpreter, dummy_input)

    print(f"Measuring latency ({N_SAMPLES} samples)...")
    latencies = []
    for i in range(N_SAMPLES):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_SAMPLES}")
        latency_ms = run_inference(interpreter, dummy_input)
        latencies.append(latency_ms)

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)

    print("\n" + "="*60)
    print(f"Single-sample INT8 latency (CPU, 1 thread):")
    print(f"  Mean:   {mean_latency:.2f} ms")
    print(f"  Median: {median_latency:.2f} ms")
    print(f"  P95:    {p95_latency:.2f} ms")
    print(f"  Min:    {np.min(latencies):.2f} ms")
    print(f"  Max:    {np.max(latencies):.2f} ms")
    print("="*60)

if __name__ == "__main__":
    main()
