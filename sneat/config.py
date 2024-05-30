import numpy as np
import configparser as cp
import os

# activation functions
activation_functions = {
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20))), # clipped to prevent overflow
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x: np.maximum(0, x),
    'leaky_relu': lambda x: np.maximum(0.01 * x, x),
    'linear': lambda x: x,
    'gaussian': lambda x: np.exp(-np.clip(x, -20, 20) ** 2), # clipped to prevent underflow
    'sin': lambda x: np.sin(x),
    'cos': lambda x: np.cos(x),
}

# config and utilities
def get_config():
    config = cp.ConfigParser()
    dirpath = os.path.abspath(os.path.dirname(__file__))
    config.read(f'{dirpath}/default_config.ini')
    return config