# Import libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow import keras

# Importing Data


# Load sample images
china = load_sample_image("china.jpg")/255
flower = load_sample_image("flower.jpg")/255

# Define Convolutional neural networks with Conv2D and pooling layers.
# Build and train models to process real-world image datasets.
# Understand how to use convolutions to improve your neural network.
# Use real-world images in different shapes and sizes..
# Use image augmentation to prevent overfitting.
# Use ImageDataGenerator.
# Understand how ImageDataGenerator labels images based on the directory structure.