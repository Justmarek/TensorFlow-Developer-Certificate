# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# Load data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# Create directories for train and test
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# Create directories to split the train and test dog sets
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

# Environmental variables
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Data preparation - Create Image Generator class, it can read images from disk and prepr
train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training d
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our valid

# After creating the generators for training and validation images, the flow_from_directo
# Applies rescaling and resizes the images into the required dimensions
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_W
                                                                        class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IM
                                                                           class_mode='binary')

# Create the model
model = Sequential([
    Conv2D(16, 3, padding='sa
    MaxPooling2D(),
    Conv2D(32, 3, padding='sa
    MaxPooling2D(),
    Conv2D(64, 3, padding='sa
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='re
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam
loss = tf.keras.l
metrics = ['accur

           model.summary()