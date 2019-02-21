import numpy as np


def L1_norm(X, Y):
    return np.abs(X[0] - X[1]) + np.abs(Y[0] - Y[1])


def euclidean_distance(X, Y):
    return np.linalg.norm(X - Y)


def Lmax_norm(X, Y):
    return max(abs(X[0] - X[1]), abs(Y[0] - Y[1]))
