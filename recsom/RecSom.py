import numpy as np

from helpers.Encoder import Encoder
from helpers.LongestCommonSubsequence import LongestCommonSubsequence
from plotting_helpers.plot_utils import *


class RecSom:
    def __init__(self, input_dimension, rows_count, columns_count, alpha=0.5, beta=0.5):
        self.input_dimension = input_dimension
        self.rows_count = rows_count
        self.columns_count = columns_count

        self.number_of_neurons_in_map = self.rows_count * self.columns_count

        # weights vectors
        self.weights = np.random.randn(rows_count, columns_count, input_dimension)
        self.context_weights = np.random.randn(rows_count, columns_count, self.number_of_neurons_in_map)

        # activities
        self.previous_step_activities = np.zeros(self.number_of_neurons_in_map)
        self.current_step_activities = np.array([])
        self.memory_window = []
        self.receptive_field = []

        # mixing parameters
        self.alpha = alpha
        self.beta = beta

    def find_winner_for_given_input(self, x):
        winner_row = -1
        winner_column = -1
        distance_from_winner = float('inf')

        self.current_step_activities = np.array([])

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                current_distance = self.alpha * np.linalg.norm(x - self.weights[i][j]) + \
                                   self.beta * np.linalg.norm(self.previous_step_activities - self.context_weights[i][j])

                self.current_step_activities = np.append(self.current_step_activities, np.exp(-current_distance))
                if current_distance < distance_from_winner:
                    distance_from_winner = current_distance
                    winner_row = i
                    winner_column = j

        return winner_row, winner_column

    def train(self, inputs, discrete=True, metric=lambda x, y: 0, alpha_s=0.01, alpha_f=0.001, lambda_s=None,
              lambda_f=1, eps=100, in3d=True, trace=True, trace_interval=10, sliding_window_size=3, log=True, log_file_name=''):

        count = len(inputs)

        if trace:
            ion()
            (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.weights, block=False)
            redraw()

        quantizations_errors = []
        memory_spans = []
        adjustments = []

        for ep in range(eps):
            self.memory_window = [[[] for x in range(self.columns_count)] for y in
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
                self.memory_window[winner_row][winner_column].append(inputs[window_size:i])

                # quantization error
                sum_of_distances += self.alpha * np.linalg.norm(x - self.weights[winner_row][winner_column]) + \
                                    self.beta * np.linalg.norm(self.previous_step_activities - self.context_weights[winner_row][winner_column])

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

                        current_context_weight_adjustment = alpha_t * (self.previous_step_activities - self.context_weights[row_index, column_index]) * h

                        self.previous_step_activities = self.current_step_activities
                        self.weights[row_index, column_index] += current_weight_adjustment

                        self.context_weights[row_index, column_index] += current_context_weight_adjustment

                        adjustment_deltas.append(current_weight_adjustment - last_adjustment)
                        last_adjustment = current_weight_adjustment

            quantization_error = sum_of_distances / (self.rows_count * self.columns_count)

            quantizations_errors.append(quantization_error)
            memory_spans.append(self.calculate_memory_span_of_net())
            average_amount_of_adjustments = 0
            for delta in adjustment_deltas:
                average_amount_of_adjustments += np.linalg.norm(np.array(delta))

            adjustments.append(average_amount_of_adjustments)

            print("adjustments: {}".format(adjustments))
            print("Quantization error: {}".format(quantization_error))
            print("Memory span of the net {}:".format(self.calculate_memory_span_of_net()))

            # receptive field
            self.create_receptive_field()
            print("Receptive field")
            print(np.matrix(self.receptive_field))

            # with open(log_file_name, 'w') as file:
            #     file.write('Aplha: {}'.format(self.alpha))
            #     file.write('Beta: {}'.format(self.beta))
            #     file.write('Epoch {}'.format(ep))
            #     file.write('\n')
            #     file.write('Quantization error: {}'.format(quantization_error))
            #     file.write('\n')
            #     file.write('Memory span: {}'.format(self.calculate_memory_span_of_net()))
            #     file.write('\n')
            #     file.write(str(np.matrix(self.receptive_field)))
            #     file.write('\n')

            with open('rec_som.csv', 'a') as file:
                file.write('{},{},{}'.format(round(self.alpha, 2), round(self.beta, 2), round(self.calculate_memory_span_of_net(), 2)))
                file.write('\n')

            if trace and ((ep + 1) % trace_interval == 0):
                (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.weights, block=False)
                redraw()
                # plot_errors('Quantization error', cumulated_quantization_error, block=False)
                # plot_errors('Adjustments changes', adjustments, block=False)
                # plot_errors('Memory spans over time', memory_spans, block=False)
                # plot_receptive_field(self.receptive_field)

        if trace:
            ioff()

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
                if longest_common_subsequence_length == 0:
                    continue
                weight = len(sequences) / longest_common_subsequence_length
                print(len(sequences))
                print(longest_common_subsequence_length)
                longest_common_subsequence_length *= weight
                sum_of_weighted_lcs += longest_common_subsequence_length
                sum_of_weigths += weight

        if sum_of_weigths == 0:
            return sum_of_weighted_lcs
        return sum_of_weighted_lcs / sum_of_weigths

