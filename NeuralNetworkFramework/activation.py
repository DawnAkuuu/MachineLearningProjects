from layer import Layer
import numpy as np


class Activation(Layer):

    # Initialize the layer
    # activation: activation function
    # activation_prime: derivative of the activation function
    def __init__(self, activation, activation_prime):
        self.activation = activation # Store the activation function
        self.activation_prime = activation_prime # Store the derivative of the activation function

    # Forward pass
    # input: input to the layer
    # return: output of the layer
    def forward(self, input):
        self.input = input # Store the input
        return self.activation(self.input) # Return the output

    # Backward pass
    # output_gradient: gradient of the loss function with respect to the output
    # learning_rate: learning rate
    # return: gradient of the loss function with respect to the input
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input)) 
        
