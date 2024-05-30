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

def get_config():
    """
    Loads the default configuration and merges it with any user-provided 'config.ini' 
    in the directory of the executing script.
    """
    # initialize the ConfigParser
    config = cp.ConfigParser()

    # path to the default configuration file
    dirpath = os.path.abspath(os.path.dirname(__file__))
    default_config_path = os.path.join(dirpath, 'default_config.ini')

    # read the default configuration
    config.read(default_config_path)

    # path to a potential user-defined configuration in the execution directory
    execution_path = os.path.abspath(os.getcwd())
    user_config_path = os.path.join(execution_path, 'config.ini')

    # check if there's a user config in the execution directory and read it if it exists
    if os.path.exists(user_config_path):
        config.read(user_config_path)

    return config