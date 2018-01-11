import tensorflow as tf
import numpy as np
import sys
import os
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import glob
import classify2
import matplotlib.pyplot as plt
from thresholder import Thresholder
import csv

### TENSORFLOW SETUP
# todo: disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### GLOBAL VARIABLES
window_size = 395
window_threshold = 0.9
smallwindow_size = 131
smallwindow_threshold = 0.7 * 255
smallwindow_step = 131
# elby testing- set back to original fine-grained vals
# you can change them back to 395, .9, 131, .7, 3
num_scans = (window_size - smallwindow_size) // smallwindow_step + 1
print("num_scans is", num_scans) # p sure this is right

### SLIDER SETUP
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(winW, winH) = (window_size, window_size)

# cut pool table into 8 images
eight_images = []
count = 0
for (x, y, window) in sliding_window(image, stepSize=window_size, windowSize=(winW, winH)):
  if window.shape[0] != winH or window.shape[1] != winW:
    continue
  eight_images.append(window)
  count+=1

# loads graph (assumed to be stored in ../logs)
with tf.gfile.FastGFile("../logs/trained_graph.pb", 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  g1 = tf.import_graph_def(graph_def, name='g1')

# CLASSIFY BIGWINDOW
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("../logs/trained_labels.txt")]

where_balls = []
has_ball = []
for img in eight_images:
  # todo: run all 8 images in a big batch?
  with tf.Session(graph=g1) as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('g1/final_result:0')
    predictions = sess.run(softmax_tensor, {'g1/DecodeJpeg:0': img})

    # [has ball, no ball]
    print(predictions)

    has_ball.append(predictions[0][0] > window_threshold)

print(has_ball)

# CLASSIFY SMALLWINDOW
fullheatmap = np.zeros((4*num_scans,2*num_scans,3))
for i in range(len(has_ball)):
  if has_ball[i]:

    heatmap = np.zeros((num_scans,num_scans,3))

    big_image = eight_images[i]
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
      heatmap[int(x/smallwindow_step),int(y/smallwindow_step),:] = predictions[0]

    f = open("elbytest/heatmap%d.txt" % i,"w")
    f.write(str(heatmap))
    f.close()

    # heatmap for a bigsmall done
    # todo: is the thing
    heatmap = heatmap * 255
    # todo: test fullheatmap insertion for correctness
    xt = num_scans * (i % 4)
    yt = num_scans * (i >= 4)
    fullheatmap[xt:(xt+num_scans), yt:(yt+num_scans), :] = heatmap

    # todo: on windows, use interpolation='none' to stop blurring effect
    # ELB ^^^

    # todo: get rid of elbytest
    # todo: transpose the heatmaps before plotting
    t = Thresholder(heatmap, smallwindow_threshold, i)
    balls = t.thresh()
    t.lovelyplot(heatmap[:,:,1], i+'thisthingthatearlier')
    t.lovelyplot(heatmap[:,:,2], i+'thisthingthatearlier')

    #
    balls = list(map(lambda ball: (ball[0],ball[1]+xt,ball[2]+yt), balls))
    print(balls)
    where_balls.extend(balls)

    t.lovelyplot(fullheatmap[:,:,1], i + "full")
    t.lovelyplot(fullheatmap[:,:,2], i + "full")
<<<<<<< HEAD
    # todo: test threshold code and improve thresholding algos
    """
    plt.imshow(fullheatmap[:,:,1])
    plt.savefig("elbytest/heatmap_solid%d" % i)
    plt.imshow(fullheatmap[:,:,1])
    plt.savefig("elbytest/heatmap_stripe%d" % i)
    """
# todo: from total list of balls given by thresholder, annotate raw images
=======

# todo: change coordinates in small 16 square to big square
# todo: from total list of balls given by thresholder, annotate raw images
>>>>>>> ee6f2a30246ccea6a0e429c3cfde33a15a88889c
