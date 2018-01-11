# added a thresholder class so we can store multiple thresholding algos

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

class Thresholder:
    def __init__(self, heatmap, threshold):
        self.heatmap = heatmap
        self.balls = []
        self.threshold = threshold

    # https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    # todo: annotation techniques described at these links
    def thresh(self):
        nb_sz = 3 # todo: play with this
        # for stripes and solids
        for balltype in [1,2]:
            data = self.heatmap[:,:,balltype]
            data_max = filters.maximum_filter(data, nb_sz)
            maxima = (data == data_max)
            data_min = filters.minimum_filter(data, nb_sz)
            diff = ((data_max - data_min) > self.threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima) # todo: play with this (size, generate_binary_structure)
            xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

            name = balltype-1 ? 'stripe' : 'solid'
            for ball in xy:
                self.balls = self.balls.append((name, ball[0], ball[1]))
    # todo: also try skimage peak_local_max, imageJ findmaxima function