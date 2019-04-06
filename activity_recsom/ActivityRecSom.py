import numpy as np

from helpers.Encoder import Encoder
from helpers.LongestCommonSubsequence import LongestCommonSubsequence
from plotting_helpers.plot_utils import *


class ActivityRecSom:
    def __init__(self, input_dimension, rows_count, columns_count):
        self.input_dimension = input_dimension
        self.rows_count = rows_count
        self.columns_count = columns_count

        self.number_of_neurons_in_map = self.rows_count * self.columns_count

        # weights vectors
        self.weights = np.random.rand(rows_count, columns_count, input_dimension)
        self.context_weights = np.random.rand(rows_count, columns_count, self.number_of_neurons_in_map)

        # activities
        self.previous_step_activities = np.zeros(self.number_of_neurons_in_map)
        self.current_step_activities = np.array([])
        self.memory_window = []
        self.receptive_field = []

        # mixing parameter initialisation
        self.alpha = 0.5

        # activity parameter
        self.beta = 1.0

        self.sliding_window_size = 30

    def find_winner_for_given_input(self, x):
        winner_row = -1
        winner_column = -1
        distance_from_winner = float('inf')

        self.current_step_activities = np.array([])

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                current_distance = (1 - self.alpha) * np.linalg.norm(x - self.weights[i][j]) + \
                                   self.alpha * np.linalg.norm(self.previous_step_activities - self.context_weights[i][j])

                self.current_step_activities = np.append(self.current_step_activities,
                                                         self.calculate_activity(x, current_distance))

                if current_distance < distance_from_winner:
                    distance_from_winner = current_distance
                    winner_row = i
                    winner_column = j


        return winner_row, winner_column

    def calculate_activity(self, x, current_distance):
        return np.exp(-self.beta*(current_distance ** 2)) / self.calculate_sum_of_activities(x)

    def calculate_sum_of_activities(self, x):
        sum_of_activations = 0
        distance = 0
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                distance = (1 - self.alpha) * np.linalg.norm(x - self.weights[i][j]) + \
                                   self.alpha * np.linalg.norm(self.previous_step_activities - self.context_weights[i][j])
        sum_of_activations += np.exp(-self.beta*(distance ** 2))
        return sum_of_activations

    def train(self, inputs, metric=lambda x, y: 0, alpha_s=0.01, alpha_f=0.001, lambda_s=None,
              lambda_f=1, eps=10, in3d=False, trace=False, trace_interval=10, sliding_window_size=10,
              log=False, log_file_name='', alpha=0.5, beta=1.0):

        self.alpha = alpha
        self.beta = beta

        count = len(inputs)

        if trace:
            ion()
            (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.weights, block=False)
            (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.context_weights, block=False)
            redraw()

        quantizations_errors = []
        memory_spans = []
        adjustments = []

        sum_of_memory_spans = 0

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

            for i in range(sliding_window_size, count):
                x = Encoder.encode_character(inputs[i])

                # find a winner
                winner_row, winner_column = self.find_winner_for_given_input(x)

                self.memory_window[winner_row][winner_column].append(inputs[i - sliding_window_size:i])

                self.sliding_window_size = sliding_window_size

                # quantization error
                sum_of_distances += (1 - self.alpha) * np.linalg.norm(x - self.weights[winner_row][winner_column]) + \
                                    self.alpha * np.linalg.norm(self.previous_step_activities - self.context_weights[winner_row][winner_column])

                winner_position = np.array([winner_row, winner_column])
                for row_index in range(self.rows_count):
                    for column_index in range(self.columns_count):
                        current_position = np.array([row_index, column_index])
                        distance_from_winner = metric(winner_position, current_position)

                        argument = -((distance_from_winner ** 2) / lambda_t ** 2)
                        h = np.exp(argument)

                        current_weight_adjustment = alpha_t * (x - self.weights[row_index, column_index]) * h
                        current_context_weight_adjustment = alpha_t * (self.previous_step_activities -
                                                                       self.context_weights[
                                                                           row_index, column_index]) * h

                        self.previous_step_activities = self.current_step_activities
                        self.weights[row_index, column_index] += current_weight_adjustment
                        self.context_weights[row_index, column_index] += current_context_weight_adjustment

                        adjustment_deltas.append(current_weight_adjustment - last_adjustment)
                        last_adjustment = current_weight_adjustment

            quantization_error = sum_of_distances / (self.rows_count * self.columns_count)

            quantizations_errors.append(quantization_error)
            memory_spans.append(self.calculate_memory_span_of_net())

            print("adjustments: {}".format(adjustments))
            print("Quantization error: {}".format(quantization_error))
            print("Memory span of the net {}:".format(self.calculate_memory_span_of_net()))

            # receptive field
            self.create_receptive_field()
            print("Receptive field")
            print(np.matrix(self.receptive_field))

            sum_of_memory_spans += self.calculate_memory_span_of_net()

            if log:
                if ep == eps - 1:
                    with open(log_file_name, 'a') as file:
                        file.write('{},{},{},{}'.format(round(1 - self.alpha, 2), round(self.alpha, 2),
                                                     round(sum_of_memory_spans / eps, 2), quantization_error))
                        file.write('\n')

            if trace and ((ep + 1) % trace_interval == 0):
                (plot_grid_3d if in3d else plot_grid_2d)(Encoder.transform_input(inputs), self.weights, block=False)
                redraw()
                plot_errors('Quantization error', quantizations_errors, block=False)
                plot_errors('Memory spans over time', memory_spans, block=False)
                plot_receptive_field(self.receptive_field)

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
                weight = len(sequences)
                longest_common_subsequence_length *= weight
                sum_of_weighted_lcs += longest_common_subsequence_length
                sum_of_weigths += weight

        if sum_of_weigths == 0:
            return sum_of_weighted_lcs
        return sum_of_weighted_lcs / sum_of_weigths


