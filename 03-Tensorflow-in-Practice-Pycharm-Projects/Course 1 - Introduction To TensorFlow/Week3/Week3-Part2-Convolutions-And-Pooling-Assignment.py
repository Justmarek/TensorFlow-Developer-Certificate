# Import libraries
import tensorflow as tf
from os import path, getcwd, chdir

# Current working directoru
path = f"{getcwd()}/data/mnist.npz"

# Create function to do assignment
def train_mnist_conv():
    # Create callback class to stop at 99.8%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get("accuracy")>0.95):
                print("/n Reached 95% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()

    # Load mnist data
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)

    # Data normalisation
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images/255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images/255.0

    # Build a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Fit model
    history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
    return history.epoch, history.history["accuracy"][-1]

# Train model function
train_mnist_conv()