import numpy as np


def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

x = NN(3, 1.5, w1, w2, b)

print(x)