# Import libraries
import os
import tensorflow as tf
import zipfile
from os import path, getcwd, chdir
from tensorflow import keras
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# User TensorFlow 2.X
print(tf.__version__)

# Import data
path= f"{getcwd()}/data/happy-or-sad.zip"
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

# Preprocess data to get it ready for use in a pretrainedmodel.
# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
     # Use callbacks to trigger the end of training cycles.
    DESIRED_ACCURACY = 0.999
    print(type(DESIRED_ACCURACY))
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epochs, logs={}):
            if (logs.get('accuracy') > DESIRED_ACCURACY):
                print('\nReached 99.9% accuracy so cancelling training!')
                self.model.stop_training = True

    # Callback class used here
    callbacks = myCallback()

    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        # Code block will assume 150 X 150 in this implementation
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convultion
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third concolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into the DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron, as this is a binary classifier
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    from tensorflow.keras.optimizers import RMSprop
    # Model is compiled here
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

    # This code block should create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call pretrainedmodel.fit_generator and train for
    # a number of epochs.
    # pretrainedmodel fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        callbacks=[callbacks])
    # pretrainedmodel fitting
    return history.history['accuracy'][-1]

train_happy_sad_model()


# Build, compile and train machine learning (ML) models using TensorFlow.

# Preprocess data to get it ready for use in a pretrainedmodel.

# Use models to predict results.

# Build sequential models with multiple layers.

# Build and train models for binary classification.

# Build and train models for multi-class categorization.

# Plot loss and accuracy of a trained pretrainedmodel.

#  Identify strategies to prevent overfitting, including augmentation and dropout.

# Extract features from pre-trained models.

# Ensure that inputs to a pretrainedmodel are in the correct shape.

# Ensure that you can match test data to the input shape of a neural network.

# Ensure you can match output data of a neural network to specified input shape for test data.