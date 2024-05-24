import numpy as np


# activation function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    if type(x) == type(np.zeros(3)):
        shape = x.shape
        new_array = np.zeros(shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                new_array[i][j] = max(0.0, x[i][j])
        return new_array
    else :
        return max(0.0, x)


def relu_prime(x):
    if type(x) == type(np.zeros(3)):
        shape = x.shape
        new_array = np.zeros(shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                value = 0
                if x[i][j] > 0:
                    value = 1
                new_array[i][j] = value
        return new_array
    else:
        value = 0
        if x > 0:
            value = 1
        return value


def sig(x):
    return 1 / (1 + np.exp(-x))


def sig_prime(x):
    sigmoid = sig(x)
    return sigmoid * (1 - sigmoid)
