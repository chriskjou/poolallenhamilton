import numpy as np
import matplotlib.pyplot as plt

pockets = np.array([[0,0],[790,0],[1580,0],[0,790],[790,790],[1580,790]])

def unit_vector(vec):
    return  vec / np.linalg.norm(vec)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def diff1(ball, cue):
    d = 0 # artificially low number
    if (ball == cue).all():
        return 0
    for pocket in pockets:
        theta = angle_between(ball - cue, pocket - ball)
        if theta > 1.58:
            continue # skip if not cuttable angle
        d1 = np.linalg.norm(ball - cue) / 100 # `\_(">)_/`
        d2 = np.linalg.norm(pocket - ball) / 100
        diff = np.cos(theta)**(4.1-2.7*theta) / d1**0.33 / d2**0.38
        if diff > d:
            d = diff
    return d

##################

cue = np.array([1200,300])

heatmap = np.zeros((1579,789))
for x in range(1,1579):
    for y in range(1,789):
        ball = np.array([x,y])
        heatmap[x,y] = diff1(ball,cue)
# it's really slow

# TODO
# why is it slow?
# change the diff fn (use e^-x instead?)

plt.imshow(heatmap.transpose(),interpolation='none')
plt.plot(cue[0],cue[1],marker='o',markersize=12,color='white')
plt.colorbar()
plt.title('Difficulty (cue at 1200,300)')
plt.xlabel('long edge target ball pos')
plt.ylabel('short edge target ball pos')
plt.show()
