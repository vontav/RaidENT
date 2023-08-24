import tensorflow as tf

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    return model
