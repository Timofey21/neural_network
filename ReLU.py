import numpy as np
import layer


class ReLU(layer.layer):
    def __init__(self):
        pass

    def forward(self, input: np.ndarray):
        output = np.zeros((input.shape[0], input.shape[1]))
        for i in range(input.shape[0]):
            output[i] = np.maximum(input[i], 0)
        return output

    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        relu_grad = input > 0
        return grad_output * relu_grad

