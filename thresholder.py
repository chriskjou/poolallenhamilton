# added a thresholder class so we can store multiple thresholding algos

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

# plots heatmap
def lovelyplot(arr, name, bsq):
    plt.imshow(arr.transpose()+1-1, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('long edge')
    plt.ylabel('short edge')
    plt.title(name)
    plt.savefig(name + str(bsq), vmin=0, vmax=1)
    plt.show()

class Thresholder:
    def __init__(self, heatmap, threshold, ballsquare):
        self.heatmap = heatmap
        self.balls = []
        self.threshold = threshold
        self.ballsquare = ballsquare

    def get_heatmap(self):
        return self.heatmap

    # sorts in order of predictions 'cue' 'eight' 'solid' 'stripe'
    def sortballs(self):
        self.balls.sort(key = lambda x: x[0])

    # https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    # TODO: also try skimage peak_local_max, imageJ findmaxima function
    def thresh(self):
        nb_sz = 3 # play with neighborhood size (see docs)
        # for solids and stripes
        for balltype in [1,2]:
            data = self.heatmap[:,:,balltype]
            name = 'stripe' if balltype - 1 else 'solid'
            # lovelyplot(data, name+'inthresh', self.ballsquare)
            data_max = filters.maximum_filter(data, nb_sz)
            maxima = (data == data_max)
            data_min = filters.minimum_filter(data, nb_sz)
            diff = ((data_max - data_min) > self.threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(diff) # play with arguments to this
            xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
            for ball in xy:
                self.balls.append((name, ball[0], ball[1]))
        self.sortballs()
        return self.balls

    def general_thresh(self):
        nb_sz = 2
        data = self.heatmap
        data_max = filters.maximum_filter(data, nb_sz)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, nb_sz)
        diff = ((data_max - data_min) > self.threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(diff)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
        for ball in xy:
            self.balls.append((ball[1], ball[0]))
        return self.balls

# personal space ballfinder
r = 2 # radius of squisher
squisher = np.array([[.51,.51,.51,.51,.51],[.51,.1,.1,.1,.51],[.51,.1,.01,.1,.51],[.51,.1,.1,.1,.51],[.51,.51,.51,.51,.51]])
assert(squisher.shape[0] == 2 * r + 1)

# gets a list of balls identified as (type, x, y)
def personalspace(heatmap,thresh):
    count = 0
    balls = []
    bustmap = np.zeros((heatmap.shape[0]+2*r,heatmap.shape[1]+2*r))
    bustmap[r:-r,r:-r] = heatmap
    maxcoord = np.argmax(bustmap)
    maxcoord = (maxcoord // bustmap.shape[1], maxcoord % bustmap.shape[1]) # converts into 2d
    maxprob = bustmap[maxcoord[0],maxcoord[1]]
    while(maxprob > thresh):
        balls.append(maxcoord)
        bustmap[(maxcoord[0]-r):(maxcoord[0]+r+1),(maxcoord[1]-r):(maxcoord[1]+r+1)] *= squisher
        maxcoord = np.argmax(bustmap)
        maxcoord = (maxcoord // bustmap.shape[1], maxcoord % bustmap.shape[1])
        maxprob = bustmap[maxcoord[0],maxcoord[1]]
    balls = list(map(lambda ball: (ball[1]-r,ball[0]-r), balls))
    return balls
