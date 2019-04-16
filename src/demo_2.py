import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import theano.tensor as tf

training_data, validation_data, test_data = network3.load_data_shared()

# PARAMETERS
mini_batch_size = 10
epochs = 60
eta = 0.1

# LAYER 1: CONVOLUTIONAL POOL LAYER PARAMETERS
image_shape = (mini_batch_size, 1, 28, 28)
filter_shape = (20, 1, 5, 5)
poolsize = (2, 2)

# LAYER 2: FULLY CONNECTED LAYER PARAMETERS
input_cells_2 = 20*12*12
output_cells_2 = 100

# LAYER 3: SOFTMAX LAYER PARAMETERS
output_cells_3 = 10

# NETWORK
n = Network([
    ConvPoolLayer(image_shape=image_shape, filter_shape=filter_shape, poolsize=poolsize), 
    FullyConnectedLayer(n_in=input_cells_2, n_out=output_cells_2), 
    SoftmaxLayer(n_in=output_cells_2, n_out=output_cells_3)
], mini_batch_size)

# EXECUTION
n.SGD(training_data, epochs, mini_batch_size, eta, validation_data, test_data)  