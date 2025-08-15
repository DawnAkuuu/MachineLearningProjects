
class Layer:
    # Initialize the layer
    def __init__(self):
        self.input = None # Input to the layer
        self.output = None # Output of the layer

    # Forward pass
    # input: input to the layer
    # return: output of the layer
    def forward(self, input):
        # TODO: Return output
        pass
    
    # Backward pass
    # output_gradient: gradient of the loss function with respect to the output
    # learning_rate: learning rate
    # return: gradient of the loss function with respect to the input
    def backward(self, output_gradient, learning_rate):
        # TODO: Update parameters and return input gradient
        pass

