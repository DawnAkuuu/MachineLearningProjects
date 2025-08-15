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
