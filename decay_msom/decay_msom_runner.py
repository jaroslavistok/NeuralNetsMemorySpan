from helpers.DataLoader import DataLoader
from helpers.norms import *
from decay_msom.DecayMergeSom import DecayMergeSom

dim = 26
rows = 30
cols = 30
metric = euclidean_distance

top_left = np.array((0, 0))
bottom_right = np.array((rows - 1, cols - 1))

lambda_s = metric(top_left, bottom_right) * 0.5

train_data = DataLoader.load_data('abcd_short')

for i in range(1):
    model = DecayMergeSom(dim, rows, cols)
    model.train(train_data, metric=metric, alpha_s=1.0, alpha_f=0.05, lambda_s=lambda_s,
                        lambda_f=1, eps=20, in3d=False, trace=True, trace_interval=5, sliding_window_size=10, log=True,
                        log_file_name='top_corpus_test.csv', alpha=0.5, beta=1.0)
# print(model.distances_between_adjacent_neurons_horizontal())
# print(model.distances_between_adjacent_neurons_vertical())

# sns.heatmap(model.distances_between_adjacent_neurons_horizontal())
# plt.show()
# sns.heatmap(model.distances_between_adjacent_neurons_vertical())
# plt.show()

#class1_x, class1_y, class2_x, class2_y, class3_x, class3_y = model.neuron_activations_data(
 #   np.loadtxt('data/seeds_dataset.txt').T)

#plt.scatter(class1_x, class1_y, c='red')
#plt.scatter(class2_x, class2_y, c='blue')
#plt.scatter(class3_x, class3_y, c='green')

#plt.show()

# heatmaps for attributes values
# for i in range(7):
#     sns.heatmap(model.weights[:, :, i])
#     plt.show()


# print(sns.heatmap(model.generate_umatrix_data()))
