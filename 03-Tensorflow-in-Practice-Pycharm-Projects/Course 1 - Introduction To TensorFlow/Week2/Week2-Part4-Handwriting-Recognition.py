# import libraries
import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/data/mnist.npz"

def train_mnist():
    # Create callback class
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get("accuracy")>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    # Instantiate callback
    callbacks = myCallback()
    # Load data
    mnist = tf.keras.datasets.mnist
    # Create train test split
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path)
    # Normalise data
    x_train = x_train/255.0
    x_test = x_test/255.0
    # Create pretrainedmodel
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    # pretrainedmodel fitting
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    return history.epoch, history.history["accuracy"][-1]

# Run function
train_mnist()