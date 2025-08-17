import cupy as cp
import numpy as np
from layer import Layer
from cupyx.scipy import signal as cp_signal

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

def safe_random_normal(loc, scale, shape, use_gpu=True):
    """Safely generate random numbers, falling back to CPU if GPU fails"""
    if use_gpu and GPU_AVAILABLE:
        try:
            # Use CPU generation + GPU transfer (avoids CURAND issues)
            cpu_array = np.random.normal(loc, scale, shape).astype(np.float32)
            return cp.array(cpu_array)
        except Exception as e:
            print(f"GPU random generation failed ({e}), using CPU fallback")
    
    # CPU fallback
    cpu_array = np.random.normal(loc, scale, shape).astype(np.float32)
    if GPU_AVAILABLE:
        return cp.array(cpu_array)
    else:
        return cpu_array

def safe_zeros(shape, use_gpu=True):
    """Safely create zeros array, falling back to CPU if GPU fails"""
    if use_gpu and GPU_AVAILABLE:
        try:
            return cp.zeros(shape, dtype=cp.float32)
        except Exception as e:
            print(f"GPU zeros creation failed ({e}), using CPU fallback")
    
    # CPU fallback
    cpu_array = np.zeros(shape, dtype=np.float32)
    if GPU_AVAILABLE:
        return cp.array(cpu_array)
    else:
        return cpu_array

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # Test CUDA availability first
        self.use_gpu = False
        if GPU_AVAILABLE:
            try:
                cp.cuda.Device(0).use()  # Use GPU 0
                # Test basic operation
                test_array = cp.array([1, 2, 3])
                print("CuPy initialized successfully")
                self.use_gpu = True
            except Exception as e:
                print(f"CuPy initialization failed: {e}")
                self.use_gpu = False
        
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        
        # Initialize with proper scaling using safe methods
        std = float(np.sqrt(2.0 / (input_depth * kernel_size * kernel_size)))
        
        # Use safe initialization
        self.kernels = safe_random_normal(0, std, self.kernels_shape, use_gpu=self.use_gpu)
        self.biases = safe_zeros(self.output_shape, use_gpu=self.use_gpu)
        
    def forward(self, input):
        if self.use_gpu:
            # GPU computation path - USE OPTIMIZED CORRELATE2D
            if isinstance(input, np.ndarray):
                gpu_input = cp.asarray(input)
            else:
                gpu_input = input
                
            self.input = gpu_input
            self.output = cp.copy(self.biases)
            
            # Use optimized CuPy correlate2d (NOT manual loops!)
            for i in range(self.depth):
                for j in range(self.input_depth):
                    # This should be fast - uses optimized CUDA kernels
                    self.output[i] += cp_signal.correlate2d(self.input[j], self.kernels[i, j], mode="valid")
            
            # Convert back to NumPy for compatibility with other layers
            return cp.asnumpy(self.output)
        else:
            # CPU computation path
            if hasattr(input, 'get'):  # CuPy array, convert to NumPy
                cpu_input = input.get()
            else:
                cpu_input = input  # Already NumPy
                
            self.input = cpu_input
            self.output = np.copy(self.biases)
            
            # Use scipy for CPU correlation
            from scipy import signal as scipy_signal
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.output[i] += scipy_signal.correlate2d(self.input[j], self.kernels[i, j], mode="valid")
            
            return self.output
    
    def backward(self, output_gradient, learning_rate):
        if self.use_gpu:
            # GPU computation path - USE OPTIMIZED FUNCTIONS
            if isinstance(output_gradient, np.ndarray):
                gpu_gradient = cp.asarray(output_gradient)
            else:
                gpu_gradient = output_gradient
                
            kernels_gradient = cp.zeros(self.kernels_shape, dtype=cp.float32)
            input_gradient = cp.zeros(self.input_shape, dtype=cp.float32)
            
            for i in range(self.depth):
                for j in range(self.input_depth):
                    # Use optimized correlate2d and convolve2d (NOT manual loops!)
                    kernels_gradient[i, j] = cp_signal.correlate2d(self.input[j], gpu_gradient[i], mode="valid")
                    input_gradient[j] += cp_signal.convolve2d(gpu_gradient[i], self.kernels[i, j], mode="full")
            
            # Update parameters on GPU
            self.kernels -= learning_rate * kernels_gradient
            self.biases -= learning_rate * gpu_gradient
            
            # Convert back to NumPy for compatibility with other layers
            return cp.asnumpy(input_gradient)
        else:
            # CPU computation path
            if hasattr(output_gradient, 'get'):  # CuPy array, convert to NumPy
                cpu_gradient = output_gradient.get()
            else:
                cpu_gradient = output_gradient  # Already NumPy
                
            from scipy import signal as scipy_signal
            kernels_gradient = np.zeros(self.kernels_shape, dtype=np.float32)
            input_gradient = np.zeros(self.input_shape, dtype=np.float32)
            
            for i in range(self.depth):
                for j in range(self.input_depth):
                    # Kernel gradients
                    kernels_gradient[i, j] = scipy_signal.correlate2d(self.input[j], cpu_gradient[i], mode="valid")
                    # Input gradients  
                    input_gradient[j] += scipy_signal.convolve2d(cpu_gradient[i], self.kernels[i, j], mode="full")
            
            # Update parameters
            self.kernels -= learning_rate * kernels_gradient
            self.biases -= learning_rate * cpu_gradient
            
            return input_gradient