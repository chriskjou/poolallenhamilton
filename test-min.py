from thresholder import Thresholder
import numpy as np
import csv

has_ball = [False, False, False, False, True, False, True]
smallwindow_threshold = .5
xt = 0
yt = 0

# file = open("elbytest/heatmap1.txt", "r")
# heatmap = np.genfromtext("elbytest/test.csv")

heatmap = []
with open('elbytest/test.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		heatmap.append(row)

heatmap = np.array(heatmap).astype(float)
heatmap = np.reshape(heatmap, (16,16,3)) # i think?
t = Thresholder(heatmap, smallwindow_threshold)
balls = t.thresh()
balls = list(map(lambda ball: (ball[0],ball[1]+xt,ball[2]+yt), balls))
print(balls)