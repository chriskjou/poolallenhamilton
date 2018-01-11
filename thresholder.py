# added a thresholder class so we can store multiple thresholding algos

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

class Thresholder:
    def __init__(self, heatmap, threshold, ballsquare):
        self.heatmap = heatmap
        self.balls = []
        self.threshold = threshold
        self.ballsquare = ballsquare

    def lovelyplot(self, arr, name):
        plt.imshow(arr.transpose()+1-1, vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel('long edge')
        plt.ylabel('short edge')
        plt.title(name)
        plt.savefig(name + str(self.ballsquare), vmin=0, vmax=1)
        plt.show()

    # https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    # todo: annotation techniques described at these links
    def thresh(self):
        nb_sz = 3 # todo: play with this
        # for solids and stripes
        for balltype in [1,2]:
            data = self.heatmap[:,:,balltype]
            name = 'stripe' if balltype - 1 else 'solid'
            self.lovelyplot(data, name)
            data_max = filters.maximum_filter(data, nb_sz)
            maxima = (data == data_max)
            data_min = filters.minimum_filter(data, nb_sz)
            diff = ((data_max - data_min) > self.threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(diff)
            xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
            for ball in xy:
                self.balls.append((name, ball[0], ball[1]))
        return self.balls
    # todo: also try skimage peak_local_max, imageJ findmaxima function