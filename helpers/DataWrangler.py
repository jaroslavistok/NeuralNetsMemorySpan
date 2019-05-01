import numpy as np
from sklearn.preprocessing import OneHotEncoder

from helpers.Encoder import Encoder


class DataWrangler:
    @staticmethod
    def get_training_data(input_string):
        X_train = []
        Y_train = []

        for i in range(len(input_string) - 1):
            encoded_character_x = Encoder.encode_character(input_string[i])
            encoded_character_y = Encoder.encode_character(input_string[i + 1])
            X_train.append(encoded_character_x)
            Y_train.append(encoded_character_y)
        return X_train, Y_train
