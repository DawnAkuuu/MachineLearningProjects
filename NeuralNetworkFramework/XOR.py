from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh(),
]

epochs = 1000
learning_rate = 0.1

# Train the network
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # Forward pass
        output = x
        for layer in network:
            output = layer.forward(output)

        # Calculate the error
        error += mse(y, output)

        # Backward pass
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    
    error /= len(X)
    print('%d/%d, error=%f' % (e + 1, epochs, error))
