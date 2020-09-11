import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

# Import the data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print("Data imported!")

# Data shape
print("Data shape is: " + str(X_train_full.shape))
print("Data type is: " + str(X_train_full.dtype))

# Creating validation set and scaling the data
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]/ 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Create class_names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Creating the pretrainedmodel with the Sequential API
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())

print(model.layers)

hidden1 = model.layers[1]

print(hidden1.name)

print(model.get_layer('dense') is hidden1)

weights, biases = hidden1.get_weights()

weights.shape

biases

biases.shape