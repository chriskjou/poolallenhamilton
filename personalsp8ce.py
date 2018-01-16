import numpy as np
# np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
# worry about these
thresh = 1.5
heatmap = np.ones((48,24))
heatmap[35,21] = 2
heatmap[1,1] = 2
heatmap[8,3] = 2

squisher = np.array([[.5,.5,.5,.5,.5],[.5,.1,.1,.1,.5],[.5,.1,.01,.1,.5],[.5,.1,.1,.1,.5],[.5,.5,.5,.5,.5]])

# add probabilities
def personalspace(heatmap):
	balls = []
	bustmap = np.zeros((heatmap.shape[0]+4,heatmap.shape[1]+4))
	bustmap[2:-2,2:-2] = heatmap
	maxcoord = np.argmax(bustmap)
	maxcoord = (maxcoord // bustmap.shape[1], maxcoord % bustmap.shape[1]) # converts into 2d
	maxprob = bustmap[maxcoord[0],maxcoord[1]]
	while(maxprob > thresh):
		balls.append(maxcoord)
		bustmap[maxcoord[0]-2:maxcoord[0]+3,maxcoord[1]-2:maxcoord[1]+3] *= squisher
		
		maxcoord = np.argmax(bustmap)
		maxcoord = (maxcoord // bustmap.shape[1], maxcoord % bustmap.shape[1]) # converts into 2d
		maxprob = bustmap[maxcoord[0],maxcoord[1]]
		print(bustmap)
		print(maxcoord)
		print(maxprob)
		plt.imshow(bustmap) # mpl y u no work?
		plt.show()
	balls = list(map(lambda ball: (ball[0]-2,ball[1]-2), balls))
	return balls

balls = personalspace(heatmap)
print(balls)
