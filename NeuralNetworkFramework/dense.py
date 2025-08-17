import numpy as np
from layer import Layer

class Dense(Layer):

    # Initialize the layer
    # input_size: size of the input
    # output_size: size of the output
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01 # Initialize the weights
        self.bias = np.random.randn(output_size, 1) # Initialize the bias
    
    # Forward pass
    # input: input to the layer
    # return: output of the layer
    def forward(self, input):
        self.input = input # Store the input
        return np.dot(self.weights, self.input) + self.bias # Return the output


    # Backward pass
    # output_gradient: gradient of the loss function with respect to the output
    # learning_rate: learning rate
    # return: gradient of the loss function with respect to the input
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T) # Calc the derivative of the loss function with respect to the weights
        self.weights -= learning_rate * weights_gradient # Update the weights
        self.bias -= learning_rate * output_gradient # Update the bias
        return np.dot(self.weights.T, output_gradient) # Return the gradient of the loss function with respect to the input



