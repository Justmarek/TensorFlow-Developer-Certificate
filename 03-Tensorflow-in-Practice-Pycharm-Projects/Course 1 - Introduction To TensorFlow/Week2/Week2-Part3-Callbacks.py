# Import libraries
import tensorflow as tf

# Create callbacks class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("accuracy")>0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

# Import data
mnist = tf.keras.datasets.fashion_mnist

# Creat test train split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Instantiate callback
callbacks = myCallback()

# Creating the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile model
model.compile(optimizer=tf.optimizers.Adam(),
             loss='sparse_categorical_crossentropy',
             metrics=["accuracy"])

# Fit model
model.fit(x_train, y_train, epochs=1, callbacks=[callbacks])
