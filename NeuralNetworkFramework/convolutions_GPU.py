import cupy as cp
import numpy as np
from layer import Layer
from cupyx.scipy import signal as cp_signal

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        
        # Initialize on GPU with proper scaling
        std = cp.sqrt(2.0 / (input_depth * kernel_size * kernel_size))
        self.kernels = cp.random.normal(0, std, self.kernels_shape)
        self.biases = cp.zeros(self.output_shape)  # Start with zero bias
        
    def forward(self, input):
        # Ensure input is on GPU
        if isinstance(input, np.ndarray):
            input = cp.asarray(input)
            
        self.input = input
        self.output = cp.copy(self.biases)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                # Use CuPy's correlate2d
                self.output[i] += cp_signal.correlate2d(self.input[j], self.kernels[i, j], mode="valid")
                
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # Ensure gradient is on GPU
        if isinstance(output_gradient, np.ndarray):
            output_gradient = cp.asarray(output_gradient)
            
        kernels_gradient = cp.zeros(self.kernels_shape)
        input_gradient = cp.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                # Kernel gradients
                kernels_gradient[i, j] = cp_signal.correlate2d(self.input[j], output_gradient[i], mode="valid")
                # Input gradients  
                input_gradient[j] += cp_signal.convolve2d(output_gradient[i], self.kernels[i, j], mode="full")
        
        # Update parameters on GPU
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        
        return input_gradient