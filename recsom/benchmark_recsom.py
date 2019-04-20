import sys
sys.path.append('/home/jaro/school/memory_span')

from helpers.DataLoader import DataLoader
from recsom.RecSom import RecSom
from helpers.norms import *

dimension = 26
number_of_rows = 20
number_of_columns = 20
metric = euclidean_distance

top_left = np.array((0, 0))
bottom_right = np.array((number_of_rows - 1, number_of_columns - 1))

lambda_s = metric(top_left, bottom_right) * 0.5

train_data = DataLoader.load_data('abcd')

alpha_values = [x*.01 for x in range(0, 101)]
for alpha in alpha_values:
    log_file_name = 'final2.csv'
    model = RecSom(input_dimension=dimension, rows_count=number_of_rows, columns_count=number_of_columns)

    model.train(train_data, metric=metric, alpha_s=1.0, alpha_f=0.05, lambda_s=lambda_s,
                lambda_f=1, eps=10, in3d=False, trace=False, trace_interval=5, sliding_window_size=15, log=True,
                log_file_name=log_file_name, alpha=alpha)




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
