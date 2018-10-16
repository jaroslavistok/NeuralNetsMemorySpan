import numpy as np

"""
Simple helper for encoding characters

"""


class Encoder:
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    @staticmethod
    def encode_character(character):
        encoded_character = np.zeros(26)
        position = Encoder.alphabet.find(character)
        encoded_character[position] = 1
        return encoded_character

    @staticmethod
    def decode_character(encoded_character):
        index = encoded_character.tolist().index(1)
        return Encoder.alphabet[index]