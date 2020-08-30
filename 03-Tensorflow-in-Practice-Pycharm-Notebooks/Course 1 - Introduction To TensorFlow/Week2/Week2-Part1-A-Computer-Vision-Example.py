# Import libraries
import tensorflow as tf
print(tf.__version__)

# Import fashion MNISt dataset
mnist = tf.keras.datasets.fashion_mnist

# Create training and test set
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalising the data
training_images = training_images / 255.0
test_images = test_images / 255.0

# Designing the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compiling a model with an optimizer and a loss function
model.compile(optimizer = tf.optimizers.Adam(),
              loss = "sparse_categorical_crossentropy",
              metrics=["accuracy"])

# We then train the model by calling model.dit asking it to fit training data to training labels
model.fit(training_images, training_labels, epochs=5)

# Model evaluation
model_evaluation = model.evaluate(test_images, test_labels)
print(model_evaluation)