"""
Main file
"""
import numpy as np


from helpers.Encoder import Encoder
from helpers.LongestCommonSubsequence import LongestCommonSubsequence

sequences = ["brownasdfoersjumps",
             "foxsxzxasis12sa[[#brown",
             "thissasbrownxc-34a@s;"]

lcs = LongestCommonSubsequence()
print(lcs.get_longest_subsequence(sequences))
print(lcs.get_longest_subsequence_length(sequences))

