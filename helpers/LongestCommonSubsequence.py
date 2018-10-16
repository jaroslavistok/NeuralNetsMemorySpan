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
        seqs_ngrams = map(self.allngram, sequences)
        intersection = reduce(set.intersection, seqs_ngrams)
        longest = max(intersection, key=len)
        return longest

    def get_longest_subsequence_length(self, sequences):
        return len(self.get_longest_subsequence(sequences))


