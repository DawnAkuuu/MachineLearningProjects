import numpy as np
from layer import Layer

# Try to import CuPy for GPU support, fallback to NumPy if not available
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with CuPy")
except ImportError:
    print("CuPy not found, falling back to CPU computation")
    cp = np  # Use NumPy as fallback
    GPU_AVAILABLE = False
except Exception as e:
    print(f"GPU initialization failed ({e}), falling back to CPU computation")
    cp = np  # Use NumPy as fallback  
    GPU_AVAILABLE = False

class Dense(Layer):
    # Initialize the layer
    # input_size: size of the input
    # output_size: size of the output
    def __init__(self, input_size, output_size):
        if GPU_AVAILABLE:
            try:
                # Try GPU random generation first
                self.weights = cp.random.randn(output_size, input_size) * 0.01  # Initialize on GPU
                self.bias = cp.random.randn(output_size, 1)  # Initialize on GPU
                self.use_gpu = True
            except Exception as e:
                print(f"GPU random generation failed ({e}), using CPU->GPU hybrid approach")
                # Fallback: generate on CPU, transfer to GPU
                try:
                    weights_cpu = np.random.randn(output_size, input_size) * 0.01
                    bias_cpu = np.random.randn(output_size, 1)
                    self.weights = cp.array(weights_cpu)  # Transfer to GPU
                    self.bias = cp.array(bias_cpu)  # Transfer to GPU
                    self.use_gpu = True
                    print("✓ Using CPU random generation + GPU computation")
                except Exception as e2:
                    print(f"GPU transfer also failed ({e2}), falling back to pure CPU")
                    self.weights = np.random.randn(output_size, input_size) * 0.01  # Pure CPU
                    self.bias = np.random.randn(output_size, 1)  # Pure CPU
                    self.use_gpu = False
        else:
            self.weights = np.random.randn(output_size, input_size) * 0.01  # CPU computation
            self.bias = np.random.randn(output_size, 1)  # CPU computation
            self.use_gpu = False
    
    # Forward pass
    # input: input to the layer (can be NumPy or CuPy array)
    # return: output of the layer (NumPy array for compatibility)
    def forward(self, input):
        if self.use_gpu:
            # GPU computation path
            if isinstance(input, cp.ndarray):
                gpu_input = input
            else:
                gpu_input = cp.array(input)  # Convert NumPy to CuPy for GPU computation
            
            self.input = gpu_input  # Store GPU version for backward pass
            
            # Compute on GPU then convert back to NumPy
            gpu_output = cp.dot(self.weights, gpu_input) + self.bias
            return cp.asnumpy(gpu_output)  # Convert back to NumPy for activation layers
        else:
            # CPU computation path
            if hasattr(input, 'get'):  # CuPy array, convert to NumPy
                cpu_input = input.get()
            else:
                cpu_input = input  # Already NumPy
            
            self.input = cpu_input  # Store for backward pass
            return np.dot(self.weights, cpu_input) + self.bias  # NumPy computation
    
    # Backward pass
    # output_gradient: gradient from activation layer (NumPy array)
    # learning_rate: learning rate
    # return: gradient of the loss function with respect to the input (NumPy array)
    def backward(self, output_gradient, learning_rate):
        if self.use_gpu:
            # GPU computation path
            if isinstance(output_gradient, cp.ndarray):
                gpu_gradient = output_gradient
            else:
                gpu_gradient = cp.array(output_gradient)  # Convert NumPy to CuPy
            
            # Compute gradients on GPU
            weights_gradient = cp.dot(gpu_gradient, self.input.T)  # GPU computation
            self.weights -= learning_rate * weights_gradient  # Update weights on GPU
            self.bias -= learning_rate * gpu_gradient  # Update bias on GPU
            
            # Compute input gradient on GPU then convert to NumPy
            input_gradient_gpu = cp.dot(self.weights.T, gpu_gradient)
            return cp.asnumpy(input_gradient_gpu)  # Return NumPy array for next layer
        else:
            # CPU computation path
            if hasattr(output_gradient, 'get'):  # CuPy array, convert to NumPy
                cpu_gradient = output_gradient.get()
            else:
                cpu_gradient = output_gradient  # Already NumPy
            
            # Compute gradients on CPU
            weights_gradient = np.dot(cpu_gradient, self.input.T)  # CPU computation
            self.weights -= learning_rate * weights_gradient  # Update weights on CPU
            self.bias -= learning_rate * cpu_gradient  # Update bias on CPU
            
            return np.dot(self.weights.T, cpu_gradient)  # Return NumPy array
    
    # Helper method to move data to CPU if needed
    def to_cpu(self):
        """Convert weights and bias to CPU (NumPy arrays)"""
        if self.use_gpu:
            self.weights = cp.asnumpy(self.weights)
            self.bias = cp.asnumpy(self.bias)
            self.use_gpu = False
        return self
    
    # Helper method to move data to GPU if needed
    def to_gpu(self):
        """Convert weights and bias to GPU (CuPy arrays)"""
        if GPU_AVAILABLE and not self.use_gpu:
            try:
                self.weights = cp.array(self.weights)
                self.bias = cp.array(self.bias)
                self.use_gpu = True
            except Exception as e:
                print(f"Failed to move layer to GPU: {e}")
        return self