# usage: python classify.py --image [path to image]
# example: ptyhon classify.py --image ../../images

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
from matplotlib.patches import Rectangle

### TENSORFLOW SETUP ###
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### GLOBAL VARIABLES ###
window_size = 395
window_threshold = 0.9
smallwindow_size = 48
smallwindow_threshold = 0.5
smallwindow_step = 23
max_window_width = 790
max_window_height = 1580
num_scans = (window_size - smallwindow_size) // smallwindow_step + 1
print("num_scans is", num_scans)

### ARGUMENT PARSER ###
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

# truncate float function
def truncate(f, n):
  # Truncates/pads a float f to n decimal places without rounding
  s = '{}'.format(f)
  if 'e' in s or 'E' in s:
    return '{0:.{1}f}'.format(f, n)
  i, p, d = s.partition('.')
  return '.'.join([i, (d+'0'*n)[:n]])

# reads image into opencv
image = cv2.imread(args["image"])
name = args["image"].split(".")[0]

# if you want to print out the image in the normal colors, 
# change following line to 'new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)'
# since openCV reads the RGB as BGR color channels
new_image = image
(winW, winH) = (smallwindow_size, smallwindow_size)

# dimensions of heatmap numpy array will need to be changed according to global variables
# numbers were chosen to decrease padding loss when reading the image in the sliding window
heatmap = np.zeros((67,33,3)) # np.zeros((154,75,3)) or np.zeros((67,33,3))
start = time.time()

# lightweight network determining whether or not ball-sized image has a ball or does not have a ball
for (x, y, window) in sliding_window(new_image, stepSize=smallwindow_step, windowSize=(winW, winH)):
  if window.shape[0] != winH or window.shape[1] != winW:
    continue
  window = utils2.skimage.transform.resize(window, (24, 24))
  predictions = classify3.isball(window)
  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),0] = truncate(predictions[0],3)
  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),1] = truncate(predictions[1],3)

  # UNCOMMENT to print out every single small heatmap generated
  # test = heatmap.transpose(1,0,2)
  # plt.imshow(test)
  # plt.savefig("../heatmapprocess/frame%d" % count)
  # cv2.imwrite(name + 'miniheatmaps%d.jpg' % count, test)
  # UNCOMMENT to print out every single small heatmap generated

  # print(predictions)
  # if predictions[1] > smallwindow_threshold:
  #   plt.imshow(window)
  #   plt.show()

print("COUNT:", count)
end = time.time()
print("RUNTIME", end - start)
heatmap = heatmap.transpose(1,0,2)
plt.imshow(heatmap)
# UNCOMMENT to print out heatmap
# plt.show()
# UNCOMMENT to print out heatmap

# old classify
heatmap = heatmap[:,:,1]

# t = Thresholder(heatmap, smallwindow_threshold, 0)
# balls = t.general_thresh()
# print(balls)

balls = personalspace(heatmap,0.5)

# UNCOMMENT to see points of interest
# plt.imshow(heatmap)
# for each in balls:
#   plt.plot(int(each[0]), int(each[1]), 'ro')
#   # rectangle = plt.Rectangle((30, 30), 2, 2, fill=False, edgecolor="red")
#   # plt.gca().add_patch(rectangle)
# plt.show()
# UNCOMMENT to see points of interest

# adding ball classifier to interesting points
interesting_count = 0
names = []
for ball in balls:
  xcoord = int(ball[0] * smallwindow_step)
  ycoord = int(ball[1] * smallwindow_step)
  small_image = new_image[max(ycoord, 0): min(ycoord + smallwindow_size, max_window_width), max(xcoord, 0): min(xcoord + smallwindow_size, max_window_height)]
  predictions = classify2.isball(small_image)
  small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

  # UNCOMMENT to save photos of what is sent to the heavyweight classifier
  # cv2.imwrite("interestingnew%d.jpg" % interesting_count, small_image)
  # UNCOMMENT to save photos of what is sent to the heavyweight classifier

  print("WHICH BALL:", ball)
  print(predictions)

  
  # labels should be written based on the order and dimensions of what labels you have trained
  # example: if your predictions are in the form:
  # [blacksolid bluesolid bluestripe greensolid greenstripe neither
  # orangesolid orangestripe pinksolid pinkstripe purplesolid purplestripe
  # redsolid redstripe white yellowsolid yellowstripe]
  #
  # then labels should be as follows:
  # # labels = ['blacksolid', 'bluesolid', 'bluestripe', 'greensolid', 'greenstripe', 
  # 'neither', 'orangesolid', 'orangestripe', 'pinksolid', 'pinkstripe', 'purplesolid', 
  # 'purplestripe', 'redsolid', 'redstripe', 'white', 'yellowsolid', 'yellowstripe']

  # predictions we trained on was:
  # [black cue neither solids stripes]

  labels = ['eight', 'cue', 'n/a', 'solid', 'stripe']
  maxnum = predictions[0][0]
  index = 0
  for i in range(1, len(labels)):
      if maxnum < predictions[0][i]:
        maxnum = predictions[0][i]
        index = i

  names.append(labels[index])
  interesting_count += 1

  # UNCOMMENT to save locations of balls
  # f = open("where_balls_transform%d.txt" % 0,"w")
  # f.write(str(where_balls))
  # f.close()
  # UNCOMMENT to save locations of balls

# to plot labels on heatmap keep as follows
# to plot labels on actual image change below to plt.imshow(image) and remove all other comments below
plt.imshow(heatmap)
for i in range(len(balls)):
  if names[i] != 'n/a':
    xcoord = balls[i][0] # * 23
    ycoord = balls[i][1] # * 23
    plt.plot(max(xcoord, 0), max(ycoord, 0), 'ro')
    #rectangle = plt.Rectangle((max(xcoord - 1, 0), max(ycoord - 2, 0)), 48, 48, fill=False, edgecolor="red") #3
    #plt.gca().add_patch(rectangle)
    #plt.text(max(int(balls[i][0] - 48), 48), max(int(balls[i][1] - 23), 48 + 23), names[i])
plt.savefig(name + 'labels.jpg')
newimage = cv2.imread(name + 'labels.jpg')
newimage = newimage[117:367,80:578]
cv2.imwrite(name + 'labels.jpg', newimage)
#plt.show()