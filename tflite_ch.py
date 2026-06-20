import tensorflow as tf
import keras

print(f"TF: {tf.__version__}")
print(f"Keras: {keras.__version__}")
print(f"TFLite converter available: {hasattr(tf.lite, 'TFLiteConverter')}")

# Check if SavedModel export works
model = keras.Sequential([keras.layers.Dense(10, input_shape=(5,))])
try:
    model.export('/tmp/test_model')
    print("✓ SavedModel export works")
except Exception as e:
    print(f"✗ SavedModel export failed: {e}")
