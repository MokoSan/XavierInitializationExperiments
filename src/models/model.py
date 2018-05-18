from tensorflow.python.keras.initializers import RandomUniform, glorot_uniform
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import numpy as np

def FullyConnectedNN(input_shape, classes, five_layers, activation, normalization):

    """
    Creates 4 or 5 Layer Fully Connected Neural Network

    Input:
        - input_shape: int, shape of features of examples
        - classes: int, number of classes
        - activation: activation function for layers - 'tanh', 'sigmoid', 'softsign'
        - normalization: boolean, True for Glorot Initialization, False for Random Uniform Initialization
        - five_layers: boolean, True for 5 layer network, False for 4 layer network

    Returns:
        - model: model class created with specified inputs

    """

    w_in = np.sqrt(0.001)
    initializer = RandomUniform(minval=-w_in, maxval=w_in)

    if normalization:
        initializer = glorot_uniform()

    x_input = Input(shape=(input_shape,))
    x = Dense(units=1000, activation=activation, kernel_initializer=initializer)(x_input)

    num_layers = 4

    if five_layers:
        num_layers = 5

    for _ in range(num_layers - 2):
        x = Dense(units=1000, activation=activation, kernel_initializer=initializer)(x)

    x_output = Dense(units=classes, activation='softmax', kernel_initializer=initializer)(x)

    model = Model(inputs=x_input, outputs=x_output)

    return model