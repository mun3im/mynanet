import tensorflow as tf
print("tf:", tf.__version__)
print("tf.keras has __version__?:", hasattr(tf.keras, "__version__") and tf.keras.__version__)
# create a tiny model to ensure Keras works
m = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
print("Model created OK")

