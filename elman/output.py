import numpy as np


class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[int(y)])

    def diff(self, x, y):
        probs = self.predict(x)
        probs[int(y)] -= 1.0
        return probs
