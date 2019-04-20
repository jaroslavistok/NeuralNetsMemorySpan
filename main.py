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
sequences = ["brownasdfoersjumps",
             "foxsxzxasis12sa[[#brown",
             "thissasbrownxc-34a@s;"]

lcs = LongestCommonSubsequence()
print(lcs.get_longest_subsequence(sequences))
print(lcs.get_longest_subsequence_length(sequences))
"""

#lcs = LCS()

lcs = LongestCommonSubsequence()
sequences = ['abbbbbbccb', 'ccbbb']
print(lcs.get_longest_subsequence_length(sequences))
quit()


# Alpha - y
# Beta - x


with open('results/recsom/final1.csv_max', 'r') as file:
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
    PlottingHelper.plot_memory_span_heatmap(z, 10, 'rec_fig.png')

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