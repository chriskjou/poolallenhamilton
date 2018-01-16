import numpy as np
from ../readercleaner import get_data2

num_states = 7**6
trans_matrix = np.zeros((num_states, num_states))

# markov chain: store transition probabilities

def state_id(state):
    idx = state[0]
    for num in state[1:]:
        idx *= 7
        idx += num
    return idx

# change this
data = get_data2(0,4)


for index, row in df.iterrows():
    pass


