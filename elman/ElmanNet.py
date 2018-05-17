import numpy as np

from plotting_helpers.plot_utils import *
from plotting_helpers.helpers import *


## Simple Recurrent Network with Elman's original simple BP training

class ElmanNet():

    def __init__(self, dim_in, dim_hid, dim_out):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.W_in = 0.01 * np.random.randn(dim_hid, dim_in + 1)
        self.W_rec = 0.01 * np.random.randn(dim_hid, dim_hid)
        self.W_out = 0.01 * np.random.randn(dim_out, dim_hid + 1)

    ## activation functions & derivations

    def cost(self, targets, outputs):
        return np.sum((targets - outputs) ** 2, axis=0)

    def f_hid(self, x):
        return 1 / (1 + np.exp(-x))

    def df_hid(self, x):
        return self.f_hid(x) * (1 - self.f_hid(x))

    def f_out(self, x):
        return x

    def df_out(self, x):
        return 1

    ## reset
    # init context values to empty
    # [what is empty? zeroes? how about expected value for an infinitely long random input sequence... ;]

    def reset(self):
        self.context = self.f_hid(np.zeros((self.dim_hid,)))

    ## forward pass -- single step
    # (single input vector)

    def forward(self, x):

        a = np.dot(self.W_in, augment(x)) + np.dot(self.W_rec, self.context)
        h = self.f_hid(a)
        b = np.dot(self.W_out, augment(h))
        y = self.f_out(b)

        c = self.context
        self.context = h

        return y, b, h, a, c

    ## forward & backprop pass -- single step
    # (single input and target vector)

    def backward(self, x, d):
        y, b, h, a, c = self.forward(x)

        g_out = self.df_out(b) * (d - y)

        g_hid = self.df_hid(a) * np.dot(self.W_out.T[:-1], g_out)

        dW_in = np.outer(g_hid, augment(x))
        dW_rec = np.outer(g_hid, c)
        dW_out = np.outer(g_out, augment(h))

        return y, dW_in, dW_rec, dW_out

    ## training

    def train(self, inputs, targets, alpha=0.1, eps=100):
        (_, count) = inputs.shape

        errors = []

        for ep in range(eps):
            print('Ep {:3d}/{}: '.format(ep + 1, eps), end='')
            E = 0

            self.reset()  # start new sequence with empty state

            for i in range(count):
                x = inputs[:, i]
                d = targets[:, i]

                y, dW_in, dW_rec, dW_out = self.backward(x, d)

                E += self.cost(d, y)

                self.W_in += alpha * dW_in
                self.W_rec += alpha * dW_rec
                self.W_out += alpha * dW_out

            E /= count
            errors.append(E)
            print('E = {:.3f}'.format(E))

        return errors

    ## testing

    def forward_seq(self, inputs):
        self.reset()

        outputs = []

        for x in inputs.T:
            y, *_ = self.forward(x)
            outputs.append(y)

        return np.array(outputs)

    def predict_seq(self, inputs, count):
        self.reset()

        outputs = []

        for x in inputs.T:
            y, *_ = self.forward(x)
            outputs.append(y)

        for i in range(count):
            y, *_ = self.forward(y)
            outputs.append(y)

        return np.array(outputs)
