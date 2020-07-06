# Using callbacks

# Saving Models

# Import libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import and transform data
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_valid = scalar.transform(X_valid)
X_test = scalar.transform(X_test)

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[: , 2:]
X_new_A, X_new_B = X_test_A[:3], X_test[:, 2:]

# Create checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint("models/my_keras_checkpoint_model.h5")
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data = ((X_valid_A, X_valid_B), y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

# Rollback to best model
model_rollback = keras.models.load_model("models/my_keras_checkpoint_model.h5")
