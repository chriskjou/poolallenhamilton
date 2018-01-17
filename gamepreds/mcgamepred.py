import numpy as np
import sys
sys.path.insert(0, '../')
from readercleaner import get_data1

# TODO: Just realized a big mistake: I don't know when the eight ball gets sunk!
# Right now I'm just assuming you win when you pot all your non-eight balls!
# If lazy, I could just get the winner from the metadata

num_states = 2**8
trans_matrix = np.zeros((num_states+2, num_states+2))

# markov chain: store transition probabilities
# start with 2-dimensional
# TODO: for 6-dim, need more efficient storing mech
def state_id(state):
    idx = state[0]
    for num in state[1:]:
        idx *= 7
        idx += num
    return idx

# populate transition matrix
for x in range(200):
    # TODO: if we have another fn that gives strictly 1 image per shot, use that instead of get_data1
    data = get_data1(x,x+1)
    data['state_id'] = data[['numstripe','numsolid']].apply(state_id)
    for i in range(len(data)-1):
        trans_matrix[data.iloc[i].state_id, data.iloc[i+1].state_id] += 1

# normalize rows
row_sums = trans_matrix.sum(axis=1)
trans_matrix = trans_matrix / row_sums[:, np.newaxis]

# set absorbing states to 1
# state -2 is stripe win, -1 is solid win
trans_matrix[-2,-2] = 1
trans_matrix[-1,-1] = 1
for x in range(1,8):
    # for stripes
    sx = state_id([0,x])
    trans_matrix[sx,:] = np.zeros(num_states+2)
    trans_matrix[sx,-2] = 1
    # for solids
    sx = state_id([x,0])
    trans_matrix[sx,:] = np.zeros(num_states+2)
    trans_matrix[sx,-1] = 1


# a way to calculate probabilities of reaching absorbing states
# Q is matrix part of all absorbing states. N=(I-Q)^-1. 
# expected # steps before absorption, given a nonabsorbing start
# expected # times visit nonabsorbing state j given nonabsorbing start
# B = RN -> prob absorption in state i given nonabsorbing start
# or take the trans matrix to infty power, instead of using B


# ADAM AND EVE

trans_matrix = np.zeros((1000,1000)) # ludicrously high
nstates = 2
coords = {}
# hm, is there a python struct that auto assigns a new unique id to each new element

# populate transition matrix
for x in range(200):
    # TODO: if we have another fn that gives strictly 1 image per shot, use that instead of get_data1
    data = get_data1(x,x+1)
    transes = data[['numstripe','numsolid']].apply(tuple, axis=1)
    for i in range(len(transes)-1):
        if transes[i] in coords:
            s0 = coords[transes[i]]
        else:
            coords[transes[i]] = nstates
            s0 = nstates
            nstates += 1
        if transes[i+1] in coords:
            s1 = coords[transes[i+1]]
        else:
            coords[transes[i+1]] = nstates
            s1 = nstates
            nstates += 1
        trans_matrix[s0,s1] += 1

# normalize rows
trans_matrix = trans_matrix[nstates, nstates]
row_sums = trans_matrix.sum(axis=1)
trans_matrix = trans_matrix / row_sums[:, np.newaxis]

# make right states to absorb
# state 0 is solid win, 1 is stripe win
trans_matrix[0, 0] = 1
trans_matrix[1, 1] = 1 
for x in range(1,8):
    # for stripes
    sx = state_id([0,x])
    trans_matrix[sx,:] = np.zeros(num_states+2)
    trans_matrix[sx,-2] = 1
    # for solids
    sx = state_id([x,0])
    trans_matrix[sx,:] = np.zeros(num_states+2)
    trans_matrix[sx,-1] = 1

# there's only 2 absorbing states

