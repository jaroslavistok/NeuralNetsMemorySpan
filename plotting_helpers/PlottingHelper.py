import numpy as np
import matplotlib.pyplot as plt

class PlottingHelper:
    def __init__(self):
        pass

    @staticmethod
    def plot_memory_span_heatmap(z, number_of_values):
        plot_data = np.zeros((number_of_values, number_of_values))
        iterator = 0
        for i in range(10):
            for j in range(10):
                plot_data[i][j] = z[iterator]
                iterator += 1

        plt.imshow(plot_data, cmap='hot', interpolation='nearest')
        plt.show()

