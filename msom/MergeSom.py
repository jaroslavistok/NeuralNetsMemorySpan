import numpy as np

from helpers.Encoder import Encoder
from helpers.LongestCommonSubsequence import LongestCommonSubsequence
from plotting_helpers.plot_utils import *


class MergeSom:
    def __init__(self, input_dimension, rows_count, columns_count, inputs=None):
        self.input_dimension = input_dimension
        self.rows_count = rows_count
        self.columns_count = columns_count
        self.number_of_neurons_in_map = self.rows_count * self.columns_count

        self.weights = np.random.randn(rows_count, columns_count, input_dimension)

        self.context_weights = np.random.randn(rows_count, columns_count, input_dimension)

        self.previous_winner_context = np.zeros(self.input_dimension)
        self.previous_winner_weights = np.zeros(self.input_dimension)

        # meta parameters
        self.beta = 0.5
        self.alpha = 0.5

    def find_winner_for_given_input(self, x):
        winner_row = -1
        winner_column = -1
        distance_from_winner = float('inf')

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                current_distance = (1 - self.beta) * np.linalg.norm(x - self.weights[i][j]) + \
                                   self.beta * np.linalg.norm(self.previous_winner_context - self.context_weights[i][j])

                if current_distance < distance_from_winner:
                    distance_from_winner = current_distance
                    winner_row = i
                    winner_column = j

        return winner_row, winner_column

    def train(self, inputs, discrete=True, metric=lambda x, y: 0, alpha_s=0.01, alpha_f=0.001, lambda_s=None,
              lambda_f=1, eps=100, in3d=True, trace=True, trace_interval=10, sliding_window_size=3):

        count = len(inputs)

        if trace:
            ion()
            (plot_grid_3d if in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        cumulated_quantization_error = []
        adjustments = []

        for ep in range(eps):

            self.memory_window = [[['' for x in range(count)] for x in range(self.columns_count)] for y in
                                  range(self.rows_count)]

            alpha_t = alpha_s * (alpha_f / alpha_s) ** ((ep - 1) / (eps - 1))
            lambda_t = lambda_s * (lambda_f / lambda_s) ** ((ep - 1) / (eps - 1))

            print()
            print('Ep {:3d}/{:3d}:'.format(ep + 1, eps))
            print('  alpha_t = {:.3f}, lambda_t = {:.3f}'.format(alpha_t, lambda_t))

            sum_of_distances = 0
            last_adjustment = 0
            adjustment_deltas = []

            for i in range(count):
                x = Encoder.encode_character(inputs[i])

                # find a winner
                winner_row, winner_column = self.find_winner_for_given_input(x)

                window_size = i - sliding_window_size
                if window_size < 0:
                    window_size = 0

                self.memory_window[winner_row][winner_column][i] = inputs[window_size:i]
                print(self.memory_window[winner_row][winner_column][i])

                if np.count_nonzero(self.previous_winner_context) == 0:
                    context = np.zeros(self.input_dimension)
                else:
                    context = (1 - self.alpha) * self.previous_winner_weights + self.alpha * self.previous_winner_context

                self.previous_winner_weights = self.weights[winner_row][winner_column]
                self.previous_winner_context = context


                # quantization error.
                sum_of_distances += (1 - self.beta) * np.linalg.norm(x - self.weights[winner_row][winner_column]) + \
                                    self.beta * np.linalg.norm(context - self.context_weights[winner_row][winner_column])

                winner_position = np.array([winner_row, winner_column])
                for row_index in range(self.rows_count):
                    for column_index in range(self.columns_count):
                        current_position = np.array([row_index, column_index])
                        distance_from_winner = metric(winner_position, current_position)

                        if discrete:
                            if distance_from_winner < lambda_t:
                                h = 1
                            else:
                                h = 0
                        else:
                            argument = -((distance_from_winner ** 2) / lambda_t ** 2)
                            h = np.exp(argument)

                        current_weight_adjustment = alpha_t * (x - self.weights[row_index, column_index]) * h

                        current_context_weight_adjustment = alpha_t * (
                                self.previous_winner_context - self.context_weights[row_index, column_index]) * h

                        self.weights[row_index, column_index] += current_weight_adjustment
                        self.context_weights[row_index, column_index] += current_context_weight_adjustment

                        adjustment_deltas.append(current_weight_adjustment - last_adjustment)
                        last_adjustment = current_weight_adjustment

            quantization_error = sum_of_distances / (self.rows_count * self.columns_count)

            cumulated_quantization_error.append(quantization_error)

            average_amount_of_adjustments = 0
            for delta in adjustment_deltas:
                average_amount_of_adjustments += np.linalg.norm(np.array(delta))

            adjustments.append(average_amount_of_adjustments)

            print("adjustments: {}".format(adjustments))

            print("Quantization error: {}".format(quantization_error))
            print(self.rows_count)
            print(self.columns_count)

            if trace and ((ep + 1) % trace_interval == 0):
                (plot_grid_3d if in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()
                plot_errors('Quantization error', cumulated_quantization_error, block=False)
                plot_errors('Adjustments changes', adjustments, block=False)

        if trace:
            ioff()

        def calculate_memory_span_of_net(self):
            lcs = LongestCommonSubsequence()
            sum_of_weighted_lcs = 0

            for i in range(self.rows_count):
                for j in range(self.columns_count):
                    sequences = list(filter(str.strip, self.memory_window[i][j]))
                    if not sequences:
                        continue
                    longest_common_subsequence_length = lcs.get_longest_subsequence_length(sequences)
                    if longest_common_subsequence_length == 0:
                        continue
                    weight = len(sequences) / longest_common_subsequence_length
                    longest_common_subsequence_length *= weight
                    sum_of_weighted_lcs += longest_common_subsequence_length

            memory_span = sum_of_weighted_lcs / (self.rows_count * self.columns_count)
            return memory_span
