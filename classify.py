import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os
import csv
from pyimagesearch.helpers import sliding_window
import argparse
import cv2
import classify2
import classify3
import utils2
import time
from thresholder import Thresholder, lovelyplot, personalspace

### TENSORFLOW SETUP
# TODO: disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### GLOBAL VARIABLES
window_size = 395
window_threshold = 0.9
smallwindow_size = 48
smallwindow_threshold = 0.5
smallwindow_step = 23
num_scans = (window_size - smallwindow_size) // smallwindow_step + 1
print("num_scans is", num_scans)

### SLIDER SETUP
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

def truncate(f, n):
  # Truncates/pads a float f to n decimal places without rounding
  s = '{}'.format(f)
  if 'e' in s or 'E' in s:
    return '{0:.{1}f}'.format(f, n)
  i, p, d = s.partition('.')
  return '.'.join([i, (d+'0'*n)[:n]])

image = cv2.imread(args["image"])
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(winW, winH) = (smallwindow_size, smallwindow_size)

# I had difficulty getting ffmpeg to crop, so I'll do it here.
# if image.shape != (1580,790):
#   image = image[40:1620,60:850]

heatmap = np.zeros((67,33,3)) # np.zeros((154,75,3)) or np.zeros((67,33,3))
count = 0
start = time.time()
for (x, y, window) in sliding_window(new_image, stepSize=smallwindow_step, windowSize=(winW, winH)):
  if window.shape[0] != winH or window.shape[1] != winW:
    continue
  window = utils2.skimage.transform.resize(window, (24, 24))
  predictions = classify3.isball(window)
  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),0] = truncate(predictions[0],3)
  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),1] = truncate(predictions[1],3)
  count += 1
  #print(predictions)
  # if predictions[1] > smallwindow_threshold:
  #   plt.imshow(window)
  #   plt.show()

end = time.time()
print("RUNTIME", end - start)
heatmap = heatmap.transpose(1,0,2)
plt.imshow(heatmap)
# plt.show()
# print(heatmap)

# Old classify
heatmap = heatmap[:,:,1]
# t = Thresholder(heatmap, smallwindow_threshold, 0)
# balls = t.general_thresh()
# print(balls)

balls = personalspace(heatmap,0.5)

plt.imshow(heatmap)
for each in balls:
  plt.plot(int(each[0]), int(each[1]), 'ro')
plt.show()

# adding ball classifier to interesting points
interesting_count = 0
names = []
for ball in balls:
  xcoord = int(ball[0] * smallwindow_step)
  ycoord = int(ball[1] * smallwindow_step)
  small_image = new_image[max(ycoord, 0): min(ycoord + smallwindow_size, 780), max(xcoord, 0): min(xcoord + smallwindow_size, 1580)]
  predictions = classify2.isball(small_image)
  small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
  cv2.imwrite("interestingnew%d.jpg" % interesting_count, small_image)
  print("WHICH BALL:", ball)
  print(predictions)

  # labels = [blacksolid bluesolid bluestripe greensolid greenstripe neither
  # orangesolid orangestripe pinksolid pinkstripe purplesolid purplestripe
  # redsolid redstripe white yellowsolid yellowstripe]

  # # labels = ['blacksolid', 'bluesolid', 'bluestripe', 'greensolid', 'greenstripe', 
  # 'neither', 'orangesolid', 'orangestripe', 'pinksolid', 'pinkstripe', 'purplesolid', 
  # 'purplestripe', 'redsolid', 'redstripe', 'white', 'yellowsolid', 'yellowstripe']

  # labels = [black cue neither solids stripes]

  labels = ['eight_ball', 'cue', 'neither', 'solids', 'stripes'] # consist with predictions
  maxnum = predictions[0][0]
  index = 0
  for i in range(1, len(labels)):
      if maxnum < predictions[0][i]:
        maxnum = predictions[0][i]
        index = i

  names.append(labels[index])
  interesting_count += 1

# f = open("../memes/where_balls_transform%d.txt" % 0,"w")
# f.write(str(where_balls))
# f.close()

# todo: change coordinates in small 16 square to big square
# todo: from total list of balls given by thresholder, annotate raw images

# TODO: from total list of balls given by thresholder, annotate raw images

plt.imshow(heatmap)
for i in range(len(balls)):
  plt.plot(int(balls[i][0]), int(balls[i][1]), 'ro')
  plt.text(int(balls[i][0]), int(balls[i][1]), names[i])
cv2.imwrite("labels%d.jpg" % 0, image)
plt.show()
