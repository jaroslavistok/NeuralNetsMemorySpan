from helpers.DataLoader import DataLoader
from recsom.RecSom import RecSom
from plotting_helpers.plot_utils import *


def L1_norm(X, Y):
    return np.abs(X[0] - X[1]) + np.abs(Y[0] - Y[1])


def euclidean_distance(X, Y):
    return np.linalg.norm(X - Y)


def Lmax_norm(X, Y):
    return max(abs(X[0] - X[1]), abs(Y[0] - Y[1]))


## load data

# skewed square
# inputs = np.random.rand(2, 100)
# inputs[1,:] += 0.5 * inputs[0,:]

# # circle
# inputs = 2*np.random.rand(2, 100) - 1
# inputs = inputs[:,np.abs(np.linalg.norm(inputs, axis=0)) < 1]

# # first two features of iris
# inputs = np.loadtxt('data/iris.dat').T[:3]

inputs = np.loadtxt('../data/seeds_dataset.txt').T[:7]
# print(inputs.shape)
# print(inputs)


# dimension is an alphabet size
dim = 26

# # first three features of iris
# inputs = np.loadtxt('data/iris.dat').T[:2]

# # all features of iris
# inputs = np.loadtxt('data/iris.dat').T

# (dim, count) = inputs.shape

## train model
rows = 5
cols = 5

metric = euclidean_distance

top_left = np.array((0, 0))
bottom_right = np.array((rows - 1, cols - 1))

lambda_s = metric(top_left, bottom_right) * 0.5

model = RecSom(dim, rows, cols)

train_sequence = 'aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabb aabbaabb aabbaabb aabbaa bbaabb aabbaabb aabb aabbaabb aabbaabb aabbaa bbaabbaabb aabbaabb aabbaabbaabbaabbaabbaa bbaabbaabbaabbaabbaabbaabbaabbaabbaabb'
lorem_ipsum = "Lorem ipsum dolor sit amet Lorem ipsum dolor sit amet Lorem ipsum dolor sit amet"

train_data = DataLoader.load_data('simple_sequences')

model.train(train_data, discrete=False, metric=metric, alpha_s=0.7, alpha_f=0.01, lambda_s=lambda_s,
            lambda_f=1, eps=100, in3d=False, trace=True, trace_interval=5, sliding_window_size=5)


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
