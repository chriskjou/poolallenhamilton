# added a thresholder class so we can store multiple thresholding algos

import numpy as np

class Thresholder:
    def __init__(self, fullheatmap, has_ball):
        self.fullheatmap = fullheatmap
        self.has_ball = has_ball
        self.balls = []

    def thresh0(self):
    	pass
    	# todo