import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

class PlottingHelper:
    def __init__(self):
        pass

    @staticmethod
    def plot_memory_span_heatmap(z, number_of_values, fig_name, x_tick, y_tick):
        plot_data = np.zeros((10, 10))
        print(z)
        iterator = 0
        for i in range(10):
            for j in range(10):
                plot_data[i][j] = z[iterator]
                iterator += 1

        #plt.imshow(plot_data, cmap='hot', interpolation='nearest', aspect='auto')


        heatmap = sb.heatmap(data=plot_data, annot=True, xticklabels=x_tick, yticklabels=y_tick)
        heatmap.set(xlabel='beta', ylabel='alpha')
        """
        for i in range(number_of_values):
            for j in range(number_of_values):
                plt.text(j, i, plot_data[i][j], va='center', ha='center', color='#0095FB')
        """

        plt.show()
        plt.savefig(fig_name)

