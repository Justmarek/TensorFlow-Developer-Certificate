# Import libraries
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create training directories
# Directory with our training horse pictures
train_horse_dir = os.path.join('data/horse_or_human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('data/horse_or_human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('data/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('data/validation-horse-or-human/humans')

# Build pretrainedmodel
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile pretrainedmodel
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# Data preproccessing
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'data/horse_or_human/',
    target_size=(300, 300),
    batch_size=129,
    class_mode="binary")

# Flow training in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    'data/validation-horse-or-human/',
    target_size=(300, 300),
    batch_size=32,
    class_mode="binary")

# Training pretrainedmodel
training = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8
)