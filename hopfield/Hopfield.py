import itertools
from plotting_helpers.plot_utils import *


class Hopfield():

    def __init__(self, dim):
        self.dim  = dim


    def train(self, patterns):
        self.W = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                sum = 0
                for k in range(len(patterns)):
                    sum += patterns[k][i] * patterns[k][j]
                if i != j:
                    self.W[i][j] = sum * 1 / len(patterns)
                else:
                    self.W[i][j] = 0

    def energy(self, s):
        sum = 0
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    sum += self.W[i][j]*s[i]*s[j]
        return -0.5 * sum

    # compute next state
    # - if neuron=None, synchronous dynamic: return a new state for all neurons
    # - otherwise, asynchronous dynamic: return a new state for the `neuron`-th neuron
    def forward(self, s, neuron=None):
        net = []
        for i in range(self.dim):
            net.append(np.inner(self.W[i], s))

        if self.beta is not None: # stochastic
            probabilities = []
            for i in range(self.dim):
                probability = 1 / (1 + np.exp(-net[i] / (1 / self.beta)))
                probabilities.append(probability)
            if neuron is None:
                result = []
                for probability in probabilities:
                    result.append(np.random.choice([1, -1], self.dim, p=[probability, 1 - probability]))
                return np.array(result)
            else:
                return np.random.choice([1, -1], 1, p=[probabilities[neuron], 1 - probabilities[neuron]])[0]


        else: # deterministic
            result = []
            for i in range(len(net)):
                result.append(self.sgn(net[i]))
            if neuron is None:
                return np.array(result)
            else:
                return self.sgn(net[neuron])



    def sgn(self, value):
        if value >= 0:
            return 1
        return -1

    def run_sync(self, x, eps=None, beta=None):
        s = x.copy()
        e = self.energy(s)
        S = [s]
        E = [e]

        self.beta = beta # set temperature for stochastic / None for deterministic

        for _ in itertools.count() if eps is None else range(eps): # enless loop (eps=None) / up to eps epochs

            s = self.forward(s, neuron=None) # update ALL neurons
            e = self.energy(s)

            S.append(s)
            E.append(e)

            sizeS = len(S)
            if sizeS > 1:
                if np.array_equal(S[sizeS - 1], S[sizeS - 2]):
                    return S, E

            for k in range(sizeS - 3, -1, -2):
                if np.array_equal(S[sizeS - 1], S[k]):
                    return S, E

        return S, E # if eps run out


    def run_async(self, x, eps=None, beta_s=None, beta_f=None, row=1, rows=1, trace=False):
        s = x.copy()
        e = self.energy(s)
        E = [e]

        title = 'Running: asynchronous {}'.format('stochastic' if beta_s is not None else 'deterministic')

        for ep in range(eps):
            if beta_s: # stochastic => temperature schedule
                self.beta = beta_s * ( (beta_f/beta_s) ** (ep/(eps-1)) )
                print('Ep {:2d}/{:2d}:  stochastic, beta = 1/T = {:7.4f}'.format(ep+1, eps, self.beta))
            else: # deterministic
                self.beta = None
                print('Ep {:2d}/{:2d}:  deterministic'.format(ep+1, eps))

            for i in np.random.permutation(self.dim):
                s[i] = self.forward(s, neuron=i) # update ONE neuron
                e = self.energy(s)
                E.append(e)

                if trace:
                    plot_state(s, errors=E, index=i, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=False)
                    redraw()

            if not trace:
                plot_state(s, errors=E, index=None, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=False)
                redraw()

            # terminate deterministic when stuck in a local/global minimum (loops generally don't occur)

            if self.beta is None:
                if np.all(self.forward(s) == s):
                    break