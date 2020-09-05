# Library Imports
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory with our training horse pictures
train_horse_dir = os.path.join('data/horse_or_human/horses')
print(train_horse_dir)
# Directory with our training human pictures
train_human_dir = os.path.join('data/horse_or_human/humans')

# Building a Small Model from Scratch
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation="relu"),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for a class ("horses") and 1 for for the other ("humans")
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# Data preprocessing
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in bacthes of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'data/horse_or_human/', # TThis is the source directoy for training images
    target_size=(300, 300),
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode="binary")

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1
)
