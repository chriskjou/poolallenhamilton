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
import matplotlib.pyplot as plt
from thresholder import Thresholder, lovelyplot


### TENSORFLOW SETUP
# TODO: disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### GLOBAL VARIABLES
window_size = 395
window_threshold = 0.9
smallwindow_size = 48
smallwindow_threshold = 0.50
smallwindow_step = 23
num_scans = (window_size - smallwindow_size) // smallwindow_step + 1
print("num_scans is", num_scans)

### SLIDER SETUP
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(winW, winH) = (smallwindow_size, smallwindow_size)

### cutting for now
# I had difficulty getting ffmpeg to crop, so I'll do it here.
if image.shape != (1580,790):
  image = image[40:1620,60:850]

# cut pool table into 8 images

# eight_images = []

heatmap = np.zeros((67,33,3)) # np.zeros((154,75,3)) #np.zeros((67,33,3))
count = 0
start = time.time()
for (x, y, window) in sliding_window(image, stepSize=smallwindow_step, windowSize=(winW, winH)):
  if window.shape[0] != winH or window.shape[1] != winW:
    continue
  window = utils2.skimage.transform.resize(window, (24, 24))
  predictions = classify3.isball(window)
  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),0] = predictions[0]
  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),1] = predictions[1]
  count+=1
  print(predictions)
  # if predictions[1] > smallwindow_threshold:
  #   plt.imshow(window)
  #   plt.show()

end = time.time()
print("RUNTIME", end-start)
heatmap = heatmap.transpose(1,0,2)
plt.imshow(heatmap)
plt.show()

heatmap = heatmap[:,:,1]
t = Thresholder(heatmap, smallwindow_threshold, 0)
balls = t.general_thresh()
print(balls)

plt.imshow(heatmap)
for each in balls:
  plt.plot(int(each[0]), int(each[1]), 'ro')
plt.show()

####


# loads graph (assumed to be stored in ../logs)
# with tf.gfile.FastGFile("../logs/trained_graph.pb", 'rb') as f:
#   graph_def = tf.GraphDef()
#   graph_def.ParseFromString(f.read())
#   g1 = tf.import_graph_def(graph_def, name='g1')

# # CLASSIFY BIGWINDOW
# label_lines = [line.rstrip() for line
#                    in tf.gfile.GFile("../logs/trained_labels.txt")]


# where_balls = []
# has_ball = []
# for img in eight_images:
#   # todo: run all 8 images in a big batch?
#   with tf.Session(graph=g1) as sess:
#     softmax_tensor = sess.graph.get_tensor_by_name('g1/final_result:0')
#     predictions = sess.run(softmax_tensor, {'g1/DecodeJpeg:0': img})

#     # [has ball, no ball]
#     print(predictions)

#     has_ball.append(predictions[0][0] > window_threshold)

# print(has_ball)

# CLASSIFY SMALLWINDOW
fullheatmap = np.zeros((4*num_scans,2*num_scans,3))
for i in range(len(has_ball)):

  heatmap = np.zeros((num_scans,num_scans,3))

  big_image = windows[i]
  print("bigimage %d has a ball (0-indexed btw)" % i)

  count = 0
  (winW, winH) = (smallwindow_size, smallwindow_size)
  for (x, y, smallwindow) in sliding_window(big_image, stepSize=smallwindow_step, windowSize=(winW, winH)):
    if smallwindow.shape[0] != winH or smallwindow.shape[1] != winW:
      continue
    count += 1

    # todo: run everything in one session (move the iterator within the session, in the other file)
    # todo: since we don't save to jpeg anymore, we don't need the "subimages" folder one level back, right?
    predictions = classify2.isball(smallwindow) #smallwindow is nparray (smallwindow_size,smallwindow_size,3)

    print("x and y transformed are", int(x/smallwindow_step),int(y/smallwindow_step))
    heatmap[int(x/smallwindow_step),int(y/smallwindow_step),:] = predictions

  f = open("elbytest/heatmap%d.txt" % i,"w")
  f.write(str(heatmap))
  f.close()

  # heatmap for a bigsmall done
  # todo: is the thing
  ### heatmap = heatmap * 255
  # todo: test fullheatmap insertion for correctness
  xt = num_scans * (i % 4)
  yt = num_scans * (i >= 4)
  fullheatmap[xt:(xt+num_scans), yt:(yt+num_scans), :] = heatmap

  # todo: on windows, use interpolation='none' to stop blurring effect
  # ELB ^^^

  # todo: transpose the heatmaps before plotting

  lovelyplot(heatmap[:,:,1], 'solidoutthresh', i)
  lovelyplot(heatmap[:,:,2], 'stripeoutthresh', i)

  t = Thresholder(heatmap, smallwindow_threshold, i)
  balls = t.thresh()

  balls = list(map(lambda ball: (ball[0],ball[1]+xt,ball[2]+yt), balls))
  print(balls)
  where_balls.extend(balls)

  lovelyplot(fullheatmap[:,:,1], 'solidfullheatmap', i)
  lovelyplot(fullheatmap[:,:,2], 'stripefullheatmap', i)

print("before transform", where_balls)
where_balls = list(map(lambda ball: (ball[0], ball[1] * window_size/num_scans, ball[2] * window_size/num_scans), balls))
print("after transform", where_balls)

# TODO: from total list of balls given by thresholder, annotate raw images
