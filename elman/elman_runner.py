from elman.ElmanNet import ElmanNet
from elman.seq import *
from plotting_helpers.plot_utils import *

## generate / load basic sequenct

k = 1
data = mackey_glass(k * 200)[::k]

q = 0.75
split = int(q * len(data))

train_data = data[:split]
test_data = data[split:]

# plot_sequence(data, split=split)


## prepare actual data

# a) 1-step prediction of next input

step = 1

train_inputs = train_data[:-step]
train_targets = train_data[step:] * -0.4

full_inputs = data[:-step]
full_targets = data[step:] * -0.4

## train model

model = ElmanNet(dim_in=1, dim_hid=200, dim_out=1)
model.train(inputs=np.atleast_2d(train_inputs), targets=np.atleast_2d(train_targets), alpha=0.1, eps=100)

## test model

# a) 1-step prediction

outputs = model.forward_seq(inputs=np.atleast_2d(full_inputs))
plot_sequence(full_targets, outputs, split=split)

# b) 1-step generation

# outputs = model.predict_seq(inputs=np.atleast_2d(train_inputs), count=len(data)-split)
# plot_sequence(full_targets, outputs, split=split, title='Generation')
