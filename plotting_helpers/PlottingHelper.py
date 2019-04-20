import numpy as np
import matplotlib.pyplot as plt

class PlottingHelper:
    def __init__(self):
        pass

    @staticmethod
    def plot_memory_span_heatmap(z, number_of_values, fig_name):
        plot_data = np.zeros((number_of_values, number_of_values))
        print(z)
        iterator = 0
        for i in range(number_of_values):
            for j in range(number_of_values):
                plot_data[i][j] = z[iterator]
                iterator += 1

        plt.imshow(plot_data, cmap='hot', interpolation='nearest')

        for i in range(number_of_values):
            for j in range(number_of_values):
                plt.text(j, i, plot_data[i][j], va='center', ha='center', color='#0095FB')

        plt.show()
        plt.savefig(fig_name)

