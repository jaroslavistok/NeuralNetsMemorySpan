import sys

from helpers.Encoder import Encoder
from helpers.DataWrangler import DataWrangler
from scipy.spatial.distance import pdist,squareform

sys.path.append('/home/jaro/school/memory_span')
import numpy as np

from elman.preprocessing import getSentenceData
import pandas as pd
from elman.rnn import Model

"""
word_dim = 8000
hidden_dim = 100
X_train, y_train = getSentenceData('../data/reddit-comments-2015-08.csv', word_dim)
"""

# data_string = "bbadcbdacdccaaaacacabdcddadbdbabaabccbbabacaadabdccccdcdaaacbaadcddbbbbbdcbaccdacbcaadabbbcaacbdcccbbabbdcdabdcbdabdadadbabcbbdbcbcdaaaccbacbcbaccaacaddbdaabaaaddcabdcbbbbdccaccbaaccccbcabacadddbbaabdbbcacbbbbdcccbbaaabacdccbcdcacacbdbadaaaacaddcbdcbabdbdcbbddaadcbbbdbbccabdabbabcddcbdaddbbdbbbabddbacddacccbabdcbbbdbbcacaddabaabadbcadbccccdaccbacacabacaadbcaddddbaabbdaabdbcdccdacacbcdacbaccdacbcbdaabcbaaadccdaaabcbbadbbbdddacbcbcdbadcadcdbccdcbccabbcbccbddbbabbaccabbccbdccdabaddacacaddabccdcabdbdbcbdcbdbaccccddcbcdbcddbdcdabbacccdbadaadddcdaabcbaadbbccccbaddcaacdcaacdacbddaddcbdddbbbadabbdbaddacadcdbacbaaddcbacbcaadbaccbacbcdcdddccdbbddbdbaccdcaddcbddbacbcddbdccbbabacaaccdbcbdbadaaaabccccdbcacacbacabcdbcdaabcacbddcabddbaacbbabdaadbbccadcdadcbcbcacbcbbdcbdbcbcbbbaabbdbddccacbccccddddcacbbcddacbcadcbbbddacdaacacbadbcdbcddccccbccacabbacbbaabdcbaccaaadcbddcdabacddaadbbdcdbbacaaabccabcaaacabddabdbbdaadddaabbdbddabdbcadabacbbdcbbbcadaaabbbccbccddbadadaabcaaaddccbccdbcadacdabcbdbacababcaacdcc"
data_string = 'bbadcbdacdccaaaacacabdcddadbdbabaabccbbabacaadabdccccdcdaaacbaadcddbbbbbdcbaccdacbcaadabbbcaacbdccc'


X_train, Y_train = DataWrangler.get_training_data(data_string)
np.random.seed(10)
rnn = Model(26, 30)

losses = rnn.train(X_train, Y_train, learning_rate=0.5, nepoch=10, evaluate_loss_after=1)

hidden_activations = dict()
sliding_window_size = 3

for i in range(sliding_window_size, len(data_string)):
    window = data_string[i - sliding_window_size:i]
    hidden_activations[window] = rnn.get_context(Encoder.encode_character(data_string[i]))
# print(hidden_activations)

data_frame = pd.DataFrame.from_dict(hidden_activations).transpose()
data_frame.index = list(hidden_activations.keys())
distances = squareform(pdist(data_frame, metric='euclidean'))
distances = pd.DataFrame(data=distances, index=list(hidden_activations.keys()))
print(distances.to_string())
string_representation = str(distances)

with open('file.txt', 'w') as file:
    file.write(distances.to_string())
