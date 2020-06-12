import numpy as np
from helper.util import eval_numerical_gradient

from Dense import Dense


l = Dense(128, 150)

assert -0.05 < l.weights.mean() < 0.05 and 1e-3 < l.weights.std() < 1e-1,\
    "The initial weights must have zero mean and small variance. "\
    "If you know what you're doing, remove this assertion."
assert -0.05 < l.biases.mean() < 0.05, "Biases must be zero mean. Ignore if you have a reason to do otherwise."

# To test the outputs, we explicitly set weights with fixed values. DO NOT DO THAT IN ACTUAL NETWORK!
l = Dense(3,4)

x = np.linspace(-1,1,2*3).reshape([2,3])
l.weights = np.linspace(-1,1,3*4).reshape([3,4])
l.biases = np.linspace(-1,1,4)

assert np.allclose(l.forward(x),np.array([[ 0.07272727,  0.41212121,  0.75151515,  1.09090909],
                                          [-0.90909091,  0.08484848,  1.07878788,  2.07272727]]))
print("Well done!")


x = np.linspace(-1,1,10*32).reshape([10,32])
l = Dense(32,64,learning_rate=0)

numeric_grads = eval_numerical_gradient(lambda x: l.forward(x).sum(),x)
grads = l.backward(x,np.ones([10,64]))

assert np.allclose(grads,numeric_grads,rtol=1e-3,atol=0), "input gradient does not match numeric grad"
print("Well done!")


# test gradients w.r.t. params
def compute_out_given_wb(w, b):
    l = Dense(32, 64, learning_rate=1)
    l.weights = np.array(w)
    l.biases = np.array(b)
    x = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
    return l.forward(x)


def compute_grad_by_params(w, b):
    l = Dense(32, 64, learning_rate=1)
    l.weights = np.array(w)
    l.biases = np.array(b)
    x = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
    l.backward(x, np.ones([10, 64]) / 10.)
    return w - l.weights, b - l.biases


w, b = numpy.random.randn(32, 64), np.linspace(-1, 1, 64)

numeric_dw = eval_numerical_gradient(lambda w: compute_out_given_wb(w, b).mean(0).sum(), w)
numeric_db = eval_numerical_gradient(lambda b: compute_out_given_wb(w, b).mean(0).sum(), b)
grad_w, grad_b = compute_grad_by_params(w, b)

assert np.allclose(numeric_dw, grad_w, rtol=1e-3, atol=0), "weight gradient does not match numeric weight gradient"
assert np.allclose(numeric_db, grad_b, rtol=1e-3, atol=0), "weight gradient does not match numeric weight gradient"
print("Well done!")