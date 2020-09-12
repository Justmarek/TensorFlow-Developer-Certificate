# Import libraries
import tensorflow as tf
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
import numpy as np

def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter=',')
        imgs = []
        labels = []

        next(reader, None)

        for row in reader:
            # First row is the header
            label = row[0]
            # Data is rows below
            data = row[1:]
            img = np.array(data).reshape((28, 28))

            imgs.append(img)
            labels.append(label)

        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)

    return images, labels


path_sign_mnist_train = f"{getcwd()}/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

training_images = tf.expand_dims(training_images, axis=3)
testing_images = tf.expand_dims(testing_images, axis=3)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1.0/255)

# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D
model = tf.keras.models.Sequential([
    # First convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
                           ])

# Compile Model.
model.compile(loss = 'categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Train the Model
history = model.fit_generator(train_datagen,
                              epochs=10,
                              steps_per_epoch=20,
                              validation_data=validation_datagen,
                              verbose=1,
                              validation_steps=3)

model.evaluate(testing_images, testing_labels, verbose=2)