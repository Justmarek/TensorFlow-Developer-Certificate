#!/usr/bin/env python

"""All that is required for the Image Classification component of the TensorFlow Developer Certificate"""

__author__ = "Marek Biernacki"
__date__ = "3 August 2020"


# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import preprocessing

# Downloading data
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Preset parameters
batch_size = 32
img_height = 180
img_width = 180

# Data preprocessing
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)