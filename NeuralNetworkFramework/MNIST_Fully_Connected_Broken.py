# MNIST Dataset with predictions of 1 or 0 using CNN
import numpy as np
from keras.datasets import mnist
from tensorflow.keras import utils

from dense import Dense
from convolutions import Convolutional
from reshape import Reshape
from activations import Tanh, Sigmoid, ReLU, Softmax
from losses import cat_cross_entropy, cat_cross_entropy_prime

def preprocess_data(x, y, limit):
    # Get indices for all digits (0-9), limiting each class
    all_indices = []
    for digit in range(10):
        digit_indices = np.where(y == digit)[0][:limit]
        all_indices.extend(digit_indices)

    # Convert to numpy array and shuffle
    all_indices = np.array(all_indices)
    all_indices = np.random.permutation(all_indices)
    
    x, y = x[all_indices], y[all_indices]

    # Flatten for fully connected layer
    x = x.reshape(len(x), 784)
    x = x.astype("float32") / 255

    # One hot encoding
    y = utils.to_categorical(y, num_classes=10)

    # Convert to list of individual samples for single sample processing
    x_samples = [x[i] for i in range(len(x))]  # Each element is shape (784,)
    y_samples = [y[i] for i in range(len(y))]  # Each element is shape (10,)
    return x_samples, y_samples

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 500)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# Neural network


network = [
    Dense(784, 100),
    Sigmoid(),
    Dense(100, 10),
    Softmax() 
]

epochs = 50
learning_rate = 0.1

# Train
for e in range(epochs):
    error = 0
    print(np.shape(x_train))

    
    for x, y in zip(x_train, y_train):
        # Forward pass
        output = x
        for layer in network:
            output = layer.forward(output)

        # Calculate the error
        error += cat_cross_entropy(y, output)

        # Backward pass
        grad = cat_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    
    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error = {error}")


# Test
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")