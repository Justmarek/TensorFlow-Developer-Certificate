# Function for fine tuning

def build_model(n_hidden=1, n_neurons=3-, learning_rare=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model