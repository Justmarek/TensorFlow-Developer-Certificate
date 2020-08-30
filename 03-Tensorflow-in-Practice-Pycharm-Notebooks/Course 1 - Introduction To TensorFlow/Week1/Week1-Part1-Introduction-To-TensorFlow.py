# Library imports
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Defining and compiling our first model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compiling the model
model.compile(optimizer="sgd", loss="mean_squared_error")

# Providing the initial data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype="float")
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype="float")

# Training the neural network
model.fit(xs, ys, epochs=100)

# Testing the model
print(model.predict([10.0]))
