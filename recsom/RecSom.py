import numpy as np
from plotting_helpers.plot_utils import *



class RecSom:
    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.weights = np.random.randn(n_rows, n_cols, dim_in)
        self.context = np.random.randn(n_rows, n_cols, dim_in)
        self.previous_activity_weights = np.random.randn(n_rows, n_cols, dim_in)

        self.alpha = 0.5
        self.beta = 0.5

    def find_winner_for_given_input(self, x):
        winner_row = -1
        winner_column = -1
        distance_from_winner = float('inf')

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                current_distance = self.alpha * np.linalg.norm(x - self.weights[i][j]) + \
                                   self.beta * np.linalg.norm(self.previous_activity_weights[i][j] - self.context[i][j])

                if current_distance < distance_from_winner:
                    distance_from_winner = current_distance
                    winner_row = i
                    winner_column = j
        return winner_row, winner_column


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

                # find a winner
                winner_row, winner_column = self.find_winner_for_given_input(x)

                # quantization error
                sum_of_distances += self.alpha * np.linalg.norm(x - self.weights[winner_row][winner_column]) + \
                                    self.beta * np.linalg.norm(self.previous_activity_weights[winner_row][winner_column] - self.context[winner_row][winner_column])

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
                        current_context_adjustment = alpha_t * (self.previous_activity_weights[r, c] - self.context[r, c]) * h

                        self.previous_activity_weights[r, c] = self.weights[r, c]
                        self.weights[r, c] += current_weight_adjustment
                        self.context[r, c] += current_context_adjustment

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

