import sys
sys.path.append('/home/jaro/school/memory_span')
from msom.MergeSom import MergeSom
from helpers.DataLoader import DataLoader
from recsom.RecSom import RecSom
from plotting_helpers.plot_utils import *
from helpers.norms import *



dim = 26
rows = 5
cols = 5
metric = euclidean_distance

top_left = np.array((0, 0))
bottom_right = np.array((rows - 1, cols - 1))

lambda_s = metric(top_left, bottom_right) * 0.5

train_data = DataLoader.load_data('simple_sequences')

model = MergeSom(dim, rows, cols)
model.train(train_data, discrete=False, metric=metric, alpha_s=0.6, alpha_f=0.01, lambda_s=lambda_s,
            lambda_f=0.8, eps=50, in3d=False, trace=False, trace_interval=1, sliding_window_size=30,
            log=True)

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
