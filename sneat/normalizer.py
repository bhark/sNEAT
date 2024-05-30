import numpy as np

class Normalizer:

    def __init__(self, size):
        self.n = np.zeros(size)
        self.mean = np.zeros(size)
        self.mean_diff = np.zeros(size)
        self.var = np.zeros(size)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = np.maximum(self.var, 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std