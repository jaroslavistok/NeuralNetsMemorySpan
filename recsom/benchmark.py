import sys
sys.path.append('/home/jaro/school/memory_span')

from helpers.DataLoader import DataLoader
from recsom.RecSom import RecSom
from plotting_helpers.plot_utils import *
from helpers.norms import *

dim = 26
rows = 10
cols = 10
metric = euclidean_distance

top_left = np.array((0, 0))
bottom_right = np.array((rows - 1, cols - 1))

lambda_s = metric(top_left, bottom_right) * 0.5

train_data = DataLoader.load_data('abcd')

values1 = [x*.1 for x in range(1, 11)]
values2 = [x*.1 for x in range(1, 11)]
for alpha in values1:
    for beta in values2:
        log_file_name = 'log_{}_{}.log'.format(alpha, beta)
        model = RecSom(dim, rows, cols, alpha, beta)
        model.train(train_data, discrete=False, metric=metric, alpha_s=0.7, alpha_f=0.01, lambda_s=lambda_s,
                    lambda_f=1, eps=20, in3d=False, trace=False, trace_interval=5, sliding_window_size=30, log=True,
                    log_file_name=log_file_name)


# print(model.distances_between_adjacent_neurons_horizontal())
# print(model.distances_between_adjacent_neurons_vertical())

# sns.heatmap(model.distances_between_adjacent_neurons_horizontal())
# plt.show()
# sns.heatmap(model.distances_between_adjacent_neurons_vertical())
# plt.show()

# class1_x, class1_y, class2_x, class2_y, class3_x, class3_y = model.neuron_activations_data(
#   np.loadtxt('data/seeds_dataset.txt').T)

# plt.scatter(class1_x, class1_y, c='red')
# plt.scatter(class2_x, class2_y, c='blue')
# plt.scatter(class3_x, class3_y, c='green')

# plt.show()

# heatmaps for attributes values
# for i in range(7):
#     sns.heatmap(model.weights[:, :, i])
#     plt.show()


# print(sns.heatmap(model.generate_umatrix_data()))
