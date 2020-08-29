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

# Creating the model with the Sequential API
# This creates a Sequential model
model = keras.models.Sequential()

# This is a flatten layer whose role is to convert each image into a 1D array. The layer is here to do preprocessing
model.add(keras.layers.Flatten(input_shape=[28,28]))

# First dense layer = each dense layer manages it's own weight matrix
model.add(keras.layers.Dense(300, activation="relu"))

# Second dense layer
model.add(keras.layers.Dense(100, activation="relu"))

# Final dense layer output ;ayer with 10 neurons (one per class)
model.add(keras.layers.Dense(10, activation="softmax"))