# MNIST Dataset with predictions of 1 or 0 using CNN
from numba import jit, cuda
import numpy as np
from keras.datasets import mnist
from tensorflow.keras import utils

from dense import Dense
from convolutions import Convolutional
from reshape import Reshape
from activations import Softmax, Tanh, Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime, cat_cross_entropy, cat_cross_entropy_prime

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]

    # indices = []
    # for label in np.unique(y):
    #     class_idx = np.where(y == label)[0][:limit]
    #     indices.append(class_idx)
    # all_indices = np.hstack(indices)

    # all_indices = np.random.permutation(all_indices)
    # x, y = x[all_indices], y[all_indices]

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255

    # One hot encoding
    num_classes = len(np.unique(y))
    y = utils.to_categorical(y, num_classes=num_classes)
    y = y.reshape(len(y), num_classes, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 6000)
x_test, y_test = preprocess_data(x_test, y_test, 4000)

# Neural network


network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.1

# Train
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # Forward pass
        output = x
        for layer in network:
            output = layer.forward(output)

        # Calculate the error
        error += binary_cross_entropy(y, output)

        # Backward pass
        grad = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    
    error /= len(x_train)
    if e % 5 == 0:
        print(f"{e}/{epochs}, error = {error}")


# Test
for x, y in zip(x_test, y_test):
    output = x
    i = 0
    for layer in network:
        output = layer.forward(output)
        if i % 100 == 0:
             print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
