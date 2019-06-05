"""
Main file
"""
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow

from helpers.DataLoader import DataLoader
from helpers.Encoder import Encoder
from helpers.LongestCommonSubsequence import LongestCommonSubsequence
from helpers.LCS import LCS
from plotting_helpers.PlottingHelper import PlottingHelper
from reber.reber_automaton import ReberAutomaton
from plotting_helpers.plot_utils import *


"""
reber = ReberAutomaton()
reber.generate_reber_strings_dataset(1000)
quit()
"""


"""
sequences = []

lcs = LongestCommonSubsequence()
print(lcs.lcs_test(sequences))
quit()
"""
#lcs = LCS()

"""
lcs = LongestCommonSubsequence()
sequences = ['sadada', 'qsdada', 'ssdada']
print(lcs.get_longest_subsequence_length(sequences))
quit()
"""

# Alpha - y
# Beta - x
"""
with open('recsom/abs.csv_errors', 'r') as file:
        rows = file.read().split('\n')
        x = []
        y = []
        for row in rows:
            if row:
                values = row.split(',')
                x.append(float(values[0]))
                scaled_value = (float(values[2]) * 30 * 30) / 100
                y.append(round(scaled_value, 2))

        plt.figure(1)
        plt.clf()
        plt.plot(x, y)

        plt.xlabel('alpha')
        plt.ylabel('kvantizačná chyba')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.gcf().canvas.set_window_title('recsom memory span')
        plt.show()
quit()
"""


with open('activity_recsom/distance.csv', 'r') as file:
    rows = file.read().split('\n')
    x = []
    y = []
    z = []
    for row in rows:
        if row:
            values = row.split(',')
            x.append(float(values[0])*10)
            y.append(float(values[1])*10)
            z.append(round(float(values[2]), 2))

    y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    PlottingHelper.plot_memory_span_heatmap(z, 10, 'rec_fig.png', x_ticks, y_ticks)

"""
with open('random_text_benchmark_merge.csv', 'r') as file:
    rows = file.read().split('\n')
    x = []
    y = []
    z = []
    for row in rows:
        if row:
            values = row.split(',')
            x.append(float(values[0]) * 10)
            y.append(float(values[1]) * 10)
            z.append(float(values[2]))
    PlottingHelper.plot_memory_span_heatmap(z, 10, 'merge_fig.png')


with open('random_text_benchmark_recsom.csv', 'r') as file:
    rows = file.read().split('\n')
    x = []
    y = []
    z = []
    for row in rows:
        if row:
            values = row.split(',')
            x.append(float(values[0]) * 10)
            y.append(float(values[1]) * 10)
            z.append(float(values[2]))
    PlottingHelper.plot_memory_span_heatmap(z, 10, 'vanishing.png')

"""