import numpy as np

from helpers.Encoder import Encoder
from helpers.LongestCommonSubsequence import LongestCommonSubsequence
from plotting_helpers.plot_utils import *


class MergeSom:
    def __init__(self, input_dimension, rows_count, columns_count):
        self.input_dimension = input_dimension
        self.rows_count = rows_count
        self.columns_count = columns_count
        self.number_of_neurons_in_map = self.rows_count * self.columns_count

        self.weights = np.random.randn(rows_count, columns_count, input_dimension)
        self.context_weights = np.random.randn(rows_count, columns_count, input_dimension)

        # context
        self.previous_winner_context = np.zeros(self.input_dimension)
        self.previous_winner_weights = np.zeros(self.input_dimension)

        self.previous_step_activities = np.zeros(self.number_of_neurons_in_map)
        self.current_step_activities = np.array([])

        self.memory_window = []
        self.receptive_field = []

        # alpha "distance" parameter initialisation
        self.alpha = 0.5

        # beta "context" parameter initialisation
        self.beta = 0.5

        self.sliding_window_size = 30

    def find_winner_for_given_input(self, x):
        winner_row = -1
        winner_column = -1
        distance_from_winner = float('inf')

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                current_distance = (1 - self.alpha) * np.linalg.norm(x - self.weights[i][j]) + \
                                   self.alpha * np.linalg.norm(self.previous_winner_context - self.context_weights[i][j])

                if current_distance < distance_from_winner:
                    distance_from_winner = current_distance
                    winner_row = i
                    winner_column = j

        return winner_row, winner_column

    def train(self, inputs, metric=lambda x, y: 0, alpha_s=0.01, alpha_f=0.001, lambda_s=None,
              lambda_f=1, eps=100, in3d=True, trace=True, trace_interval=10, sliding_window_size=3, log=True,
              log_file_name='', alpha=0.5, beta=0.5):
        count = len(inputs)

        print("Alpha {}".format(alpha))
        print("Beta {}".format(beta))

        self.alpha = alpha
        self.beta = beta

        if trace:
            ion()
            (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.weights, block=False)
            redraw()
            ion()
            (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.context_weights, block=False)
            redraw()

        quantization_errors = []
        memory_spans = []
        sum_of_memory_spans = 0

        for ep in range(eps):
            self.memory_window = [[['' for x in range(count)] for x in range(self.columns_count)] for y in
                                  range(self.rows_count)]
            alpha_t = alpha_s * (alpha_f / alpha_s) ** ((ep - 1) / (eps - 1))
            lambda_t = lambda_s * (lambda_f / lambda_s) ** ((ep - 1) / (eps - 1))

            print()
            print('Ep {:3d}/{:3d}:'.format(ep + 1, eps))
            print('  alpha_t = {:.3f}, lambda_t = {:.3f}'.format(alpha_t, lambda_t))

            base_learning_rate = 1

            sum_of_distances = 0
            for i in range(sliding_window_size, count):
                x = Encoder.encode_character(inputs[i])
                # find a winner
                winner_row, winner_column = self.find_winner_for_given_input(x)

                self.memory_window[winner_row][winner_column].append(inputs[i - sliding_window_size:i])
                self.sliding_window_size = sliding_window_size

                if np.count_nonzero(self.previous_winner_context) == 0:
                    context = np.random.rand(self.input_dimension)
                else:
                    context = (self.beta * self.previous_winner_weights) + ((1 - self.beta) * self.previous_winner_context)

                self.previous_winner_weights = self.weights[winner_row][winner_column]
                self.previous_winner_context = context

                # quantization error.
                sum_of_distances += (1 - self.alpha) * np.linalg.norm(x - self.weights[winner_row][winner_column]) + \
                                        self.alpha * np.linalg.norm(context - self.context_weights[winner_row][winner_column])

                winner_position = np.array([winner_row, winner_column])
                for row_index in range(self.rows_count):
                    for column_index in range(self.columns_count):
                        current_position = np.array([row_index, column_index])
                        distance_from_winner = metric(winner_position, current_position)

                        argument = -((distance_from_winner ** 2) / lambda_t ** 2)
                        h = np.exp(argument)

                        # current_weight_adjustment = alpha_t * (x - self.weights[row_index, column_index]) * h
                        current_weight_adjustment = alpha_t * (x - self.weights[row_index, column_index]) * h

                        # current_context_weight_adjustment = alpha_t * (self.previous_winner_context - self.context_weights[row_index, column_index]) * h
                        current_context_weight_adjustment = alpha_t * (self.previous_winner_context - self.context_weights[row_index, column_index]) * h


                        self.weights[row_index, column_index] += current_weight_adjustment
                        self.context_weights[row_index, column_index] += current_context_weight_adjustment

                        last_adjustment = current_weight_adjustment

                base_learning_rate -= 0.001

            quantization_error = sum_of_distances / (count - sliding_window_size)

            quantization_errors.append(quantization_error)


            memory_span = self.calculate_memory_span_of_net()

            memory_spans.append(memory_span)

            print("Quantization error: {}".format(quantization_error))
            print("Memory span of the net {}:".format(memory_span))

            # receptive field
            # self.create_receptive_field()
            # print("Receptive field")
            # print(np.matrix(self.receptive_field))
            sum_of_memory_spans += memory_span

            if log:
                if ep == eps - 1:
                    with open(log_file_name, 'a') as file:
                        file.write('{},{},{}'.format(round(self.alpha, 2), round(self.beta, 2),
                                                     round(memory_span, 2)))
                        file.write('\n')

                    with open(log_file_name + '_max', 'a') as file:
                        file.write('{},{},{}'.format(round(self.alpha, 2), round(self.beta, 2),
                                                     round(max(memory_spans), 2)))
                        file.write('\n')

                    with open(log_file_name + '_errors', 'a') as file:
                        file.write('{},{},{}'.format(round(self.alpha, 2), round(self.beta, 2),
                                                    round(quantization_error, 2)))
                        file.write('\n')

            if trace and ((ep + 1) % trace_interval == 0):
                (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.weights, block=False)
                redraw()
                plot_errors('Quantization error', quantization_errors, block=False)

        if trace:
            ioff()

    def calculate_memory_span_of_net(self):
        longest_common_subsecquence = LongestCommonSubsequence()
        sum_of_weighted_lcs = 0
        sum_of_weigths = 0

        for i in range(self.rows_count):
            for j in range(self.columns_count):
                sequences = list(filter(str.strip, self.memory_window[i][j]))
                if not sequences:
                    continue
                longest_common_subsequence_length = longest_common_subsecquence.get_longest_subsequence_length(sequences)
                weight = len(sequences)
                longest_common_subsequence_length *= weight
                sum_of_weighted_lcs += longest_common_subsequence_length
                sum_of_weigths += weight

        if sum_of_weigths == 0:
            return sum_of_weighted_lcs
        return sum_of_weighted_lcs / sum_of_weigths

    def create_receptive_field(self):
        lcs = LongestCommonSubsequence()
        # creates empty receptive field
        self.receptive_field = [['' for x in range(self.rows_count)] for y in range(self.columns_count)]

        for i in range(self.rows_count):
            for j in range(self.columns_count):
                sequences = list(filter(str.strip, self.memory_window[i][j]))
                if not sequences:
                    continue
                longest_common_subsequence = lcs.get_longest_subsequence(sequences)
                self.receptive_field[i][j] = longest_common_subsequence
