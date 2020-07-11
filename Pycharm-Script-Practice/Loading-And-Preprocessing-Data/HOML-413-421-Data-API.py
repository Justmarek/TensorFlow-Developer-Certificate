# Import libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# User TensorFlow 2.X
import tensorflow as tf
from tensorflow import keras

# Creating a dataset entirely in RAM
X = tf.range(10) # Any data Tensor
dataset = tf.data.Dataset.from_tensor_slices(X)

# Iterate over a datasets items
for item in dataset:
    print(item)

# One you have a dataset you can apply all sorts of transformations to it by calling its transformation methods
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)