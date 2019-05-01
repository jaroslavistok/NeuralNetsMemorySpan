from functools import partial, reduce
from itertools import chain
from typing import Iterator


class LongestCommonSubsequence:
    def __init__(self):
        pass

    def ngram(self, seq: str, n: int) -> Iterator[str]:
        return (seq[i: i + n] for i in range(0, len(seq) - n + 1))

    def allngram(self, seq: str) -> set:
        lengths = range(len(seq))
        ngrams = map(partial(self.ngram, seq), lengths)
        return set(chain.from_iterable(ngrams))

    def get_longest_subsequence(self, sequences):
        seq_length = len(sequences)
        if seq_length == 0:
            return 0
        if seq_length == 1:
            return sequences[0]+str(len(sequences))
        longest_sequence = ''
        for i in range(seq_length):
            postfix = ''
            character = None
            for sequence in sequences:
                reversed_sequence = sequence[::-1]
                current_character = reversed_sequence[i]
                character = current_character
                postfix += current_character
            if postfix.count(character) == len(postfix):
                longest_sequence = postfix
            else:
                break
        return longest_sequence+str(len(sequences))

    def lcs_test(self, sequences):
        seq_length = len(sequences)
        if seq_length == 0:
            return 0
        if seq_length == 1:
            return len(sequences[0])
        longest_sequence = 0
        for i in range(seq_length):
            postfix = ''
            character = None
            for sequence in sequences:
                reversed_sequence = sequence[::-1]
                current_character = reversed_sequence[i]
                character = current_character
                postfix += current_character
            if postfix.count(character) == len(postfix):
                longest_sequence += 1
            else:
                break
        return longest_sequence



    def get_longest_subsequence_length(self, sequences):
        seq_length = len(sequences)
        if seq_length == 0:
            return 0
        if seq_length == 1:
            return len(sequences[0])
        longest_sequence = 0
        for i in range(len(sequences[0])):
            postfix = ''
            character = None
            for sequence in sequences:
                reversed_sequence = sequence[::-1]
                current_character = reversed_sequence[i]
                character = current_character
                postfix += current_character
            if postfix.count(character) == len(postfix):
                longest_sequence += 1
            else:
                break
        return longest_sequence


        """
        longest = self.get_longest_subsequence(sequences)
        length = len(longest)

        if not len(longest) == 0:

            if not any(c not in 'a' for c in longest):
                length += 1

            if not any(c not in 'b' for c in longest):
                length += 1

            if not any(c not in 'c' for c in longest):
                length += 1

            if not any(c not in 'd' for c in longest):
                length += 1
        else:
            return 1

        return length
        """


