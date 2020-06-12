import numpy as np


class layer:

    def __init__(self):
        pass

    def forward(self, input: np.ndarray):
        return input

    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        return np.array([[], []])