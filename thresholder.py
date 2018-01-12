# added a thresholder class so we can store multiple thresholding algos

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

def lovelyplot(arr, name, bsq):
    # on windows, use interpolation='none' to stop blurring effect
    plt.imshow(arr.transpose()+1-1, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('long edge')
    plt.ylabel('short edge')
    plt.title(name)
    plt.savefig("../memes/" + name + str(bsq), vmin=0, vmax=1)
    plt.show()

def uglyplot(arr, name, bsq):
        plt.imshow(arr+1-1, vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel('long edge')
        plt.ylabel('short edge')
        plt.title(name)
        plt.savefig("../memes/" + name + str(bsq), vmin=0, vmax=1)
        plt.show()

class Thresholder:
    def __init__(self, heatmap, threshold, ballsquare):
        self.heatmap = heatmap
        self.balls = []
        self.threshold = threshold
        self.ballsquare = ballsquare

    def get_heatmap(self):
        return self.heatmap

    # sorts in order 'cue' 'eight' 'solid' 'stripe'
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
            lovelyplot(data, name+'inthresh', self.ballsquare)
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

    # todo: also try skimage peak_local_max, imageJ findmaxima function

    def general_thresh(self):
        nb_sz = 2 # todo: play with this
        # for solids and stripes
        data = self.heatmap
        uglyplot(data, 'threshold', self.ballsquare)
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
