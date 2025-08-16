import numpy as np

# Mean Square Error
# y_true: true values
# y_pred: predicted values
# return: mean square error
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# Mean Square Error Prime
# y_true: true values
# y_pred: predicted values
# return: mean square error prime
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def cat_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))

def cat_cross_entropy_prime(y_true, y_pred):
    return (y_pred - y_true)