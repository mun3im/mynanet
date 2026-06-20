#!/usr/bin/env python3
"""
Quick TFLite INT8 Conversion Test
Tests if your TF/Keras setup can convert a TCN-like model to INT8 TFLite
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tempfile
import os

print("=" * 70)
print("TFLite INT8 Conversion Test")
print("=" * 70)
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")
print("=" * 70)

# Create dummy data matching your spec shape (64x300x1)
print("\n1. Creating test data...")
X_test = np.random.randn(50, 64, 300, 1).astype(np.float32)
y_test = np.random.randint(0, 4, 50)
X_calib = X_test[:20]
print(f"   ✓ Test data shape: {X_test.shape}")

# Create a simplified TCN-like model
print("\n2. Building test TCN model...")
inputs = layers.Input(shape=(64, 300, 1))
x = layers.Permute((2, 1, 3))(inputs)  # (300, 64, 1)
x = layers.Reshape((300, 64))(x)  # (300, 64)
x = layers.Dense(32, use_bias=True)(x)
x = layers.Activation('relu6')(x)

# Simple TCN block
for d in [1, 2, 4]:
    res = x
    x = layers.Conv1D(32, 3, dilation_rate=d, padding='causal', use_bias=True)(x)
    x = layers.Activation('relu6')(x)
    if res.shape[-1] != 32:
        res = layers.Conv1D(32, 1, use_bias=True)(res)
    x = layers.Add()([x, res])

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, use_bias=True)(x)
x = layers.Activation('relu6')(x)
outputs = layers.Dense(4, activation='softmax', use_bias=True)(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"   ✓ Model created: {model.count_params()} parameters")

# Quick training to initialize weights properly
print("\n3. Training for 2 epochs (to initialize weights)...")
model.fit(X_test, y_test, epochs=2, batch_size=8, verbose=0)
print("   ✓ Training complete")

# Save as .keras and .h5
print("\n4. Saving models...")
with tempfile.TemporaryDirectory() as tmp_dir:
    keras_path = os.path.join(tmp_dir, 'test.keras')
    h5_path = os.path.join(tmp_dir, 'test.h5')
    tflite_path = os.path.join(tmp_dir, 'test.tflite')
    
    model.save(keras_path)
    model.save(h5_path)
    print(f"   ✓ Saved .keras: {os.path.getsize(keras_path)/1024:.1f} KB")
    print(f"   ✓ Saved .h5: {os.path.getsize(h5_path)/1024:.1f} KB")
    
    # Test INT8 conversion - Method 1: Direct from model
    print("\n5. Testing INT8 conversion...")
    print("   Method 1: Direct from Keras model...")
    
    def rep_dataset():
        for i in range(len(X_calib)):
            yield [X_calib[i:i+1].astype(np.float32)]
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        if hasattr(converter, 'experimental_new_converter'):
            converter.experimental_new_converter = False
        if hasattr(converter, 'experimental_new_quantizer'):
            converter.experimental_new_quantizer = False
        
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"   ✓ SUCCESS! INT8 TFLite: {os.path.getsize(tflite_path)/1024:.1f} KB")
        method1_success = True
        
    except Exception as e:
        print(f"   ✗ FAILED: {str(e)[:100]}...")
        method1_success = False
    
    # Test INT8 conversion - Method 2: Via SavedModel
    if not method1_success:
        print("\n   Method 2: Via SavedModel...")
        try:
            saved_model_path = os.path.join(tmp_dir, 'saved_model')
            
            # Try Keras 3.x method first
            try:
                model.export(saved_model_path)
            except AttributeError:
                # Fallback for older Keras
                tf.saved_model.save(model, saved_model_path)
            
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            if hasattr(converter, 'experimental_new_converter'):
                converter.experimental_new_converter = False
            if hasattr(converter, 'experimental_new_quantizer'):
                converter.experimental_new_quantizer = False
            
            tflite_model = converter.convert()
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"   ✓ SUCCESS! INT8 TFLite: {os.path.getsize(tflite_path)/1024:.1f} KB")
            method2_success = True
            
        except Exception as e:
            print(f"   ✗ FAILED: {str(e)[:100]}...")
            method2_success = False
    else:
        method2_success = True
    
    # Test INT8 conversion - Method 3: FP16 fallback
    if not method1_success and not method2_success:
        print("\n   Method 3: FP16 fallback...")
        try:
            saved_model_path = os.path.join(tmp_dir, 'saved_model_fp16')
            
            try:
                model.export(saved_model_path)
            except AttributeError:
                tf.saved_model.save(model, saved_model_path)
            
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            fp16_path = os.path.join(tmp_dir, 'test_fp16.tflite')
            with open(fp16_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"   ✓ FP16 works: {os.path.getsize(fp16_path)/1024:.1f} KB")
            print("   ⚠ Note: FP16, not INT8")
            method3_success = True
            
        except Exception as e:
            print(f"   ✗ FAILED: {str(e)[:100]}...")
            method3_success = False
    else:
        method3_success = True
    
    # Verify inference works
    if method1_success or method2_success:
        print("\n6. Testing INT8 inference...")
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"   Input:  {input_details[0]['dtype']} {input_details[0]['shape']}")
            print(f"   Output: {output_details[0]['dtype']} {output_details[0]['shape']}")
            
            # Run inference on one sample
            input_scale, input_zero_point = input_details[0]['quantization']
            x_fp32 = X_test[0:1]
            x_int8 = (x_fp32 / input_scale + input_zero_point).astype(np.int8)
            
            interpreter.set_tensor(input_details[0]['index'], x_int8)
            interpreter.invoke()
            
            output_int8 = interpreter.get_tensor(output_details[0]['index'])
            print(f"   ✓ Inference successful! Output shape: {output_int8.shape}")
            
        except Exception as e:
            print(f"   ✗ Inference failed: {e}")

# Final verdict
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

if method1_success or method2_success:
    print("✓ INT8 CONVERSION WORKS!")
    print("  Your TF/Keras setup can convert TCN models to INT8 TFLite")
    print("  You can proceed with your full training script")
elif method3_success:
    print("⚠ INT8 FAILED, but FP16 works")
    print("  Consider using FP16 quantization instead")
    print("  Or try a different TensorFlow version")
else:
    print("✗ ALL CONVERSIONS FAILED")
    print("  Try different TensorFlow versions:")
    print("    pip install tensorflow==2.15.0")
    print("    pip install tensorflow==2.20.0")

print("=" * 70)
