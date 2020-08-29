# Imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tensorflow import keras

# The following code applied dropout regularization before every Dense layer, using a dropout rate of 0.2

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
]
)
