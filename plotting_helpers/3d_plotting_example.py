#!/usr/bin/env python3

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# import time


## utility

def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0 - xg, x1 + xg))


## "hack" #1: non-blocking figures still block at end

import atexit


def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)

## "hack" #2: pressing Q kills the program

import os


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work


## generate data

data = np.random.randn(6, 500)


def rot(a, b, angle):
    res = np.eye(6)
    res[a, a] = +np.cos(angle * 2 * np.pi)
    res[a, b] = +np.sin(angle * 2 * np.pi)
    res[b, a] = -np.sin(angle * 2 * np.pi)
    res[b, b] = +np.cos(angle * 2 * np.pi)
    return res


T = 240
t = 1 / T

mat = rot(0, 3, 1 * t) @ rot(1, 4, 2 * t) @ rot(2, 5, 3 * t) @ rot(3, 4, 4 * t)  # magic

## plot

plt.ion()  # turn on _interactive_ plotting

plt.figure(4)
plt.gcf().canvas.set_window_title('Plot demo/test')
plt.show(block=False)

ax = Axes3D(plt.gcf())

xlim = limits(data[0, :])
ylim = limits(data[1, :])
zlim = limits(data[2, :])

# for i in range(T):
while True:
    plt.figure(4)
    plt.gcf().canvas.mpl_connect('key_press_event', keypress)

    ax.cla()

    ax.scatter(data[0, :], data[1, :], data[2, :])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # plt.gcf().canvas.draw()
    plt.waitforbuttonpress(timeout=0.001)  # actually redraw and process inputs
    # time.sleep(0.01)

    data = mat @ data  # animate

    # plt.savefig('demo-{:03d}.png'.format(i))
