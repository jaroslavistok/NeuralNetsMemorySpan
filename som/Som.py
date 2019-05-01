from plotting_helpers.plot_utils import *


class Som:
    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.weights = np.random.randn(n_rows, n_cols, dim_in)

    def find_winner_for_given_input(self, x):
        winner_row = -1
        winner_column = -1
        distance_from_winner = float('inf')

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                current_distance = np.linalg.norm(x - self.weights[i][j])
                if current_distance < distance_from_winner:
                    distance_from_winner = current_distance
                    winner_row = i
                    winner_column = j
        return winner_row, winner_column

    def generate_umatrix_data(self):
        umatrix_data = np.zeros((self.n_rows, self.n_cols))
        for x in range(len(self.weights)):
            for y in range(len(self.weights[x])):
                umatrix_data[x, y] = self.count_average_distance_from_neighbors(x, y)
        return umatrix_data

    def distances_between_adjacent_neurons_horizontal(self):
        horizontal_distances = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if j < (self.n_cols - 1):
                    horizontal_distances[i, j] = np.linalg.norm(self.weights[i, j] - self.weights[i, j + 1])
                else:
                    horizontal_distances[i, j] = np.linalg.norm(self.weights[i, j - 1] - self.weights[i, j])
        return horizontal_distances

    def distances_between_adjacent_neurons_vertical(self):
        horizontal_distances = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                if j < (self.n_rows - 1):
                    horizontal_distances[j, i] = np.linalg.norm(self.weights[j, i] - self.weights[j + 1, i])
                else:
                    horizontal_distances[j, i] = np.linalg.norm(self.weights[j - 1, i] - self.weights[j, i])
        return horizontal_distances

    def neuron_activations_data(self, inputs):
        (_, count) = inputs.shape
        activations_data = np.zeros((self.n_rows, self.n_cols))

        class1_x = []
        class1_y = []

        class2_x = []
        class2_y = []

        class3_x = []
        class3_y = []

        for i in range(count - 1):
            x = inputs[:, i]
            input_class = x[7]

            x = x[:7]
            winner_row, winner_column = self.find_winner_for_given_input(x)
            if input_class == 1:
                class1_x.append(winner_row)
                class1_y.append(winner_column)
            elif input_class == 2:
                class2_x.append(winner_row)
                class2_y.append(winner_column)
            elif input_class == 3:
                class3_x.append(winner_row)
                class3_y.append(winner_column)
        return class1_x, class1_y, class2_x, class2_y, class3_x, class3_y

    def count_average_distance_from_neighbors(self, neuron_x, neuron_y):
        sum_of_distances = 0
        for x in range(len(self.weights)):
            for y in range(len(self.weights[x])):
                if x != neuron_x and y != neuron_y:
                    sum_of_distances += np.linalg.norm(self.weights[neuron_x, neuron_y] - self.weights[x, y])
        return sum_of_distances / ((self.n_rows * self.n_cols) - 1)

    def train(self, inputs, discrete=True, metric=lambda x, y: 0, alpha_s=0.01, alpha_f=0.001, lambda_s=None,
              lambda_f=1, eps=100, in3d=True, trace=True, trace_interval=10):

        (_, count) = inputs.shape

        if trace:
            ion()
            (plot_grid_3d if in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        cumulated_quantization_error = []
        adjustments = []

        for ep in range(eps):

            alpha_t = alpha_s * (alpha_f / alpha_s) ** ((ep - 1) / (eps - 1))
            lambda_t = lambda_s * (lambda_f / lambda_s) ** ((ep - 1) / (eps - 1))

            print()
            print('Ep {:3d}/{:3d}:'.format(ep + 1, eps))
            print('  alpha_t = {:.3f}, lambda_t = {:.3f}'.format(alpha_t, lambda_t))

            sum_of_distances = 0
            last_adjustment = 0
            adjustment_deltas = []

            for i in np.random.permutation(count):
                x = inputs[:, i]

                print("x dimension")
                print(x.shape)

                winner_row, winner_column = self.find_winner_for_given_input(x)

                # quantization error
                sum_of_distances += np.linalg.norm(x - self.weights[winner_row, winner_column])

                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        winner_position = np.array([r, winner_row])
                        current_position = np.array([c, winner_column])
                        distance_from_winner = metric(winner_position, current_position)

                        if discrete:
                            if distance_from_winner < lambda_t:
                                h = 1
                            else:
                                h = 0
                        else:
                            argument = -((distance_from_winner ** 2) / lambda_t ** 2)
                            h = np.exp(argument)

                        current_weight_adjustment = alpha_t * (x - self.weights[r, c]) * h

                        self.weights[r, c] += current_weight_adjustment
                        adjustment_deltas.append(current_weight_adjustment - last_adjustment)
                        last_adjustment = current_weight_adjustment

            quantization_error = sum_of_distances / (self.n_rows * self.n_cols)

            cumulated_quantization_error.append(quantization_error)

            average_amount_of_adjustments = 0
            for delta in adjustment_deltas:
                average_amount_of_adjustments += np.linalg.norm(np.array(delta))

            adjustments.append(average_amount_of_adjustments)

            print("adjustments: {}".format(adjustments))

            print("Quantization error: {}".format(quantization_error))
            print(self.n_rows)
            print(self.n_cols)

            if trace and ((ep + 1) % trace_interval == 0):
                (plot_grid_3d if in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()
                plot_errors('Quantization error', cumulated_quantization_error, block=False)
                plot_errors('Adjustments changes', adjustments, block=False)

        if trace:
            ioff()
