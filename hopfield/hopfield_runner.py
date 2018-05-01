import numpy as np
import random

from hopfield import Hopfield
from plotting_helpers.plot_utils import *

## 1. load data

dataset = 'medium'
# dataset = 'medium'

patterns = []

with open(dataset + '.txt') as f:
    count, width, height = [int(x) for x in f.readline().split()]
    dim = width * height

    for _ in range(count):
        f.readline()  # skip empty line
        x = np.empty((height, width))
        for r in range(height):
            x[r, :] = np.array(list(f.readline().strip())) == '#'

        patterns.append(2 * x.flatten() - 1)  # flatten to 1D vector, rescale {0,1} -> {-1,+1}

util_setup(width, height)

## 2. (optionally) select a subset of patterns

# patterns = patterns[:]
# count = len(patterns)


## 3. analytically train the model

plot_states(patterns, 'Training patterns')

model = Hopfield(dim)
model.train(patterns)

## 4. generate an input

# a) from before: random binary pattern

input = np.random.choice([-1, 1], size=dim)

# b) corrupted input

# q
# index = random.randint(0, len(patterns) - 1)
# input = patterns[index]
#
# random_indices = random.sample(set(range(0, len(input))), 7)
# for i in range(len(input)):
#     for j in range(len(random_indices)):
#         if i == random_indices[j]:
#             input[i] *= -1

# TODO select random input pattern
# TODO pick some indices ("pixels")
# TODO flip (+1 <-> -1) those input bits


## 5. run the model

plot_states([input], 'Random/corrupted input')

# a) synchronous deterministic

# S, E = model.run_sync(input, eps=15)
# plot_states(S, 'Synchronous run')


# b) asynchronous deterministic

# model.run_async(input, eps=20, trace=True)
# model.run_async(input, eps=20, trace=False)


# c) asynchronous stochastic vs. deterministic

model.run_async(input, eps=20, rows=2, row=1)
model.run_async(input, eps=20, rows=2, row=2, beta_s=1, beta_f=10)