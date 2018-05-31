from tensorflow.python.keras.initializers import RandomUniform, glorot_uniform
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import numpy as np

def fully_connected_neural_net(input_shape, classes, is_five_layers, activation, is_normalized):

    """
    Creates 4 or 5 Layer Fully Connected Neural Network

    Input:
        - input_shape: int, shape of features of examples
        - classes: int, number of classes
        - activation: activation function for layers - 'tanh', 'sigmoid', 'softsign'
        - is_normalized: boolean, True for Glorot Initialization, False for Random Uniform Initialization
        - is_five_layers: boolean, True for 5 layer network, False for 4 layer network

    Returns:
        - model: model class created with specified inputs

    """

    w_in = np.sqrt(0.001)
    initializer = RandomUniform(minval=-w_in, maxval=w_in)
    
    num_layers = 4

    if is_normalized:
        initializer = glorot_uniform()
    
    if is_five_layers:
        num_layers = 5

    x_input = Input(shape=(input_shape,))
    x = Dense(units=1000, activation=activation, kernel_initializer=initializer)(x_input)
    
    # Model has two layers already outside of for loop i.e. Input to Dense_1 and Softmax Layer. 
    # 2 is subtracting from num_layers to add appropriate number of layers for model i.e. 2 for 4 layer, and 3 for 5 layer.
    for _ in range(num_layers - 2):
        x = Dense(units=1000, activation=activation, kernel_initializer=initializer)(x)

    x_output = Dense(units=classes, activation='softmax', kernel_initializer=initializer)(x)

    model = Model(inputs=x_input, outputs=x_output)

    return model