import numpy as np


## Mackey-Glass discrete equation.

def mackey_glass(n, a=0.2, b=0.8, c=0.9, d=23, e=10, initial=0.1):
    x = np.zeros(n)
    x[0] = initial
    d = int(d)
    for k in range(0, n - 1):
        x[k + 1] = c * x[k] + ((a * x[k - d]) / (b + (x[k - d] ** e)))
    return x
