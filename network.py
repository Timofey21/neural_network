import matplotlib
import numpy as np
from jedi.api.refactoring import inline

import ReLU
import Dense
from helper.mnist import load_dataset
from loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt



def forward(network, X):
    """
    Compute activations of all network layers by applying them sequentially.
    Return a list of activations for each layer.
    Make sure last activation corresponds to network logits.
    """
    activations = []
    input = X

    for i in range(len(network)):
        activations.append(network[i].forward(input))
        input = activations[-1]

    assert len(activations) == len(network)
    return activations


def predict(network, X):
    """
    Use network to predict the most likely class for each sample.
    """
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    """
    Train your network on a given batch of X and y.
    You first need to run forward to get all layer activations.
    You can estimate loss and loss_grad, obtaining dL / dy_pred
    Then you can run layer.backward going from last layer to first,
    propagating the gradient of input to previous layers.

    After you called backward for all layers, all Dense layers have already made one gradient step.
    """

    # Get the layer activations
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    # propagate gradients through network layers using .backward
    # hint: start from last layer and move to earlier layers
    for i in range(len(network) - 1, -1, -1):
        loss_grad = network[i].backward(layer_inputs[i], loss_grad)

    return np.mean(loss)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


if __name__ == "__main__":

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

    plt.figure(figsize=[6, 6])
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title("Label: %i" % y_train[i])
        plt.imshow(X_train[i].reshape([28, 28]), cmap='gray')

    input_shape = 0
    network = []

    network.append(Dense.Dense(X_train.shape[1], 100))
    network.append(ReLU.ReLU())
    network.append(Dense.Dense(100, 200))
    network.append(ReLU.ReLU())
    network.append(Dense.Dense(200, 10))

    train_log = []
    val_log = []

    for epoch in range(25):

        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32, shuffle=True):
            train(network, x_batch, y_batch)

        train_log.append(np.mean(predict(network, X_train) == y_train))
        val_log.append(np.mean(predict(network, X_val) == y_val))

        clear_output()
        print("Epoch", epoch)
        print("Train accuracy:", train_log[-1])
        print("Val accuracy:", val_log[-1])
        plt.plot(train_log, label='train accuracy')
        plt.plot(val_log, label='val accuracy')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    test_log = np.mean(predict(network, X_test) == y_test)

    clear_output()
    print("Test accuracy:", test_log)

