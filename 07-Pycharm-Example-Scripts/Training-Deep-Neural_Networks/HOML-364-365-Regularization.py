# Imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tensorflow import keras

'''How to apply l2 Regularisation to a Keras Layers Weights, using a regularization factor of 0.01
The l2() function returns a regularizer that will be called at each step during training to compute the regularization loss
This is added to your final loss
'''
layerl2 = keras.layers.Dense(100, activation="elu",
                             kernel_initializer="he_normal",
                             kernal_regularizer=keras.regularizers.l2(0.01))

'''Applying l1 is very similar'''
layerl1 = keras.layers.Dense(100, activation="elu",
                             kernel_initializer="he_normal",
                             kernal_regularizer=keras.regularizers.l2(0.01))

'''Applying both '''
layerl1l2 = keras.layers.Dense(100, activation="elu",
                             kernel_initializer="he_normal",
                             kernal_regularizer=keras.regularizers.l1_l2(0.01))

''' Typically we will want to apply the same regularizer to all layers in the network as well as using the same activation
function and the same initialization strategy in all hidden layers, this can mean repeating the same arguments. 
This makes the code ugly and error prone. To avoid this we use pythons functools.partial() function, which lets you create 
 a thin wrapper for any callable, with some default values'''

from functools import partial

RegularizedDense = partial(keras.layers.Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernal_regularizer=regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    RegularizedDense(300),
    RegularizedDense(300),
    RegularizedDense(10, activation="softmax",
                     kernal_initializer="glorot_uniform")
    ])