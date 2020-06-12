import numpy as np
import layer


class Dense(layer.layer):
    def __init__(self, input_units, ouput_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_units, ouput_units) * 0.01
        self.biases = np.zeros(ouput_units)

    def forward(self, input: np.ndarray):
        output = np.zeros((input.shape[0], len(self.biases)))
        for i in range(len(input)):
            output[i] = np.dot(input[i], self.weights) + self.biases
        return output

    def backward(self, input: np.ndarray, grad_output: np.ndarray):
        transposed_weights = self.weights.T
        grad_input = np.dot(grad_output, transposed_weights)

        grad_weights = input.T.dot(grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        self.weights = self.weights - grad_weights * self.learning_rate
        self.biases = self.biases - grad_biases * self.learning_rate

        return grad_input




