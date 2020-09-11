import tensorflow as tf
import numpy as np
from tensorflow import keras

# Creating a pretrainedmodel
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compiling pretrainedmodel
model.compile(optimizer="sgd", loss="mean_squared_error")

# Create housing data
xs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10])
ys = np.array([50, 100, 150, 200, 250, 300, 350, 450, 550])

# Fit pretrainedmodel
model.fit(xs, ys, epochs=100)

# Predict 7 bedroom house
print(model.predict([7.0]))