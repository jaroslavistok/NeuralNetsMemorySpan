import atexit
import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # todo: remove or change if not working
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

## plotting

palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure(1)
    plt.clf()

    plt.plot(errors)

    if test_error:
        plt.plot([test_error] * len(errors))

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title)
    plt.show(block=block)


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0 - xg, x1 + xg))


def plot_grid_2d(inputs, weights, i_x=0, i_y=1, s=60, block=True):
    plt.figure(1).canvas.mpl_connect('key_press_event', keypress)
    plt.clf()

    plt.gcf().canvas.set_window_title('SOM neurons and inputs (2D)')
    plt.scatter(inputs[i_x, :], inputs[i_y, :], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        plt.plot(weights[r, :, i_x], weights[r, :, i_y], c=palette[0])

    for c in range(n_cols):
        plt.plot(weights[:, c, i_x], weights[:, c, i_y], c=palette[0])

    plt.xlim(limits(inputs[i_x, :]))
    plt.ylim(limits(inputs[i_y, :]))
    plt.tight_layout()
    plt.show(block=block)


def plot_grid_3d(inputs, weights, i_x=0, i_y=1, i_z=2, s=60, block=True):
    fig = plt.figure(2)
    fig.canvas.mpl_connect('key_press_event', keypress)
    plt.gcf().canvas.set_window_title('SOM neurons and inputs (3D)')

    if plot_grid_3d.ax is None:
        plot_grid_3d.ax = Axes3D(fig)

    ax = plot_grid_3d.ax
    ax.cla()

    ax.scatter(inputs[i_x, :], inputs[i_y, :], inputs[i_z, :], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        ax.plot(weights[r, :, i_x], weights[r, :, i_y], weights[r, :, i_z], c=palette[0])

    for c in range(n_cols):
        ax.plot(weights[:, c, i_x], weights[:, c, i_y], weights[:, c, i_z], c=palette[0])

    ax.set_xlim(limits(inputs[i_x, :]))
    ax.set_ylim(limits(inputs[i_y, :]))
    ax.set_zlim(limits(inputs[i_z, :]))
    plt.show(block=block)


plot_grid_3d.ax = None

width = None
height = None


def util_setup(w, h):
    global width, height
    width = w
    height = h


def plot_state(s, errors=None, index=None, max_eps=None, rows=1, row=1, size=2, aspect=2, title=None, block=True):
    if plot_state.fig is None:
        plot_state.fig = plt.figure(
            figsize=(size, size * rows) if errors is None else ((1 + aspect) * size, size * rows))
        plot_state.fig.canvas.mpl_connect('key_press_event', keypress)

        gs = gridspec.GridSpec(rows, 2, width_ratios=[1, aspect])
        plot_state.grid = {(r, c): plt.subplot(gs[r, c]) for r in range(rows) for c in range(2 if errors else 1)}

        plt.subplots_adjust()
        plt.tight_layout()

    plot_state.fig.show()  # foreground, swith plt.(g)cf

    ax = plot_state.grid[row - 1, 0]
    ax.clear()
    ax.imshow(s.reshape((height, width)), cmap='gray', interpolation='nearest', vmin=-1, vmax=+1)
    ax.set_xticks([])
    ax.set_yticks([])

    if index:
        ax.scatter(index % width, index // width, s=150)

    if errors is not None:
        ax = plot_state.grid[row - 1, 1]
        ax.clear()
        ax.plot(errors)

        if max_eps:
            ax.set_xlim(0, max_eps - 1)

        ylim = ax.get_ylim()
        ax.vlines(np.arange(0, len(errors), width * height)[1:], ymin=ylim[0], ymax=ylim[1], color=[0.8] * 3, lw=1)
        ax.set_ylim(ylim)

    plt.gcf().canvas.set_window_title(title or 'State')
    plt.show(block=block)


plot_state.fig = None


def plot_states(S, title=None, block=True):
    plt.figure(2, figsize=(9, 3)).canvas.mpl_connect('key_press_event', keypress)
    plt.clf()

    for i, s in enumerate(S):
        plt.subplot(1, len(S), i + 1)
        plt.imshow(s.reshape((height, width)), cmap='gray', interpolation='nearest', vmin=-1, vmax=+1)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title or 'States')
    plt.show(block=block)


def plot_sequence(targets, outputs=None, split=None, title=None, block=True):
    plt.figure(2).canvas.mpl_connect('key_press_event', keypress)

    if outputs is None:
        plt.plot(targets)
        lim = limits(targets)

    else:
        plt.plot(targets, lw=5, alpha=0.3)
        plt.plot(outputs)
        lim = limits(np.concatenate((outputs.flat, targets)))

    if split is not None:
        plt.vlines([split], ymin=lim[0], ymax=lim[1], color=palette[-1], lw=1)

    plt.ylim(lim)
    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title or ('Prediction' if outputs is not None else 'Sequence'))
    plt.show(block=block)


## interactive drawing, very fragile....

wait = 0.0


def clear():
    plt.clf()


def ion():
    plt.ion()
    # time.sleep(wait)


def ioff():
    plt.ioff()


def redraw():
    plt.gcf().canvas.draw()
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(wait)


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work


## non-blocking figures still block at end

def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)
