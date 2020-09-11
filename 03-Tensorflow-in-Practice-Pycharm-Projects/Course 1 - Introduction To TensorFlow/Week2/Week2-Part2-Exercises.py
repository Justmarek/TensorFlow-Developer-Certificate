# Import libraries
import tensorflow as tf
print(tf.__version__)

# Import fashion MNISt dataset
mnist = tf.keras.datasets.mnist

# Create training and test set
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalising the data
training_images = training_images / 255.0
test_images = test_images / 255.0

# Designing the pretrainedmodel
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compiling a pretrainedmodel with an optimizer and a loss function
model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy")

# Creating a callback class to stop training at 60%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print("nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()


# We then train the pretrainedmodel by calling pretrainedmodel.dit asking it to fit training data to training labels
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Model evaluation
model_evaluation = model.evaluate(test_images, test_labels)
print(model_evaluation)

# Creating a set of classification for each of the test images
classifications = model.predict(test_images)
print(classifications[0])

# Printing label
print(test_labels[0])