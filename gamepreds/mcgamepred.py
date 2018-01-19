import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from readercleaner import get_data1

trans_matrix = np.zeros((1000,1000)) # ludicrously high
nstates = 2
coords = {}
# hm, is there a python struct that auto assigns a new unique id to each new element

# populate transition matrix
for x in range(200):
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
trans_matrix[0,0] = 1
trans_matrix[1,1] = 1 
trans_matrix = trans_matrix[:nstates,:nstates]
row_sums = trans_matrix.sum(axis=1)
trans_matrix = trans_matrix / row_sums[:,np.newaxis]

# make right states to absorb
# state 0 is solid win, 1 is stripe win
# TODO: Just realized a big mistake: I don't know when the eight ball gets sunk!
# Right now I'm just assuming you win when you pot all your non-eight balls!
# If lazy, I could just get the winner from the metadata
for x in range(1,8):
    # for stripes win
    if (0,x) in coords.keys():
        sx = coords[(0,x)]
        trans_matrix[sx,:] = 0
        trans_matrix[sx,1] = 1
    # for solids win
    if (x,0) in coords.keys():
        sx = coords[(x,0)]
        trans_matrix[sx,:] = 0
        trans_matrix[sx,0] = 1

# there's only 2 absorbing states- don't need to rearrange matrix rows
# and i don't need to chop away states that are all zeros

# TODO: confirm that the matrix looks sane

# Fundamental Matrix Math
# E(nsteps before absorption) starting from i is sum_j N_ji
# E(nvisits to j) starting from i is N_ji
# P(absorption in j) starting from i is B_ji
# or take the trans matrix to infty power, instead of using B
trans_matrix = trans_matrix.transpose()
r = trans_matrix[:2,2:]
q = trans_matrix[2:,2:]
n = np.linalg.inv(np.eye(q.shape[0])-q)
b = np.matmul(r,n)
# np.sum(b,axis=0)
print('prob of solid win', sum(b[0])/(nstates-2))
print('prob of stripe win', sum(b[1])/(nstates-2))

# untested
heatmap = np.zeros((8,8))
for numsolid in range(8):
    for numstripe in range(8):
        if (numsolid,numstripe) in coords.keys():
            x = coords[(numsolid,numstripe)]
            heatmap[numsolid,numstripe] = b[1,x-2]

plt.imshow(heatmap,origin='lower')
plt.colorbar()
plt.title('MC Predictions (1 indicates stripe victory)')
plt.xlabel('# solids on table')
plt.ylabel('# stripes on table')
plt.show()
