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

### TENSORFLOW SETUP
# todo: disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### GLOBAL VARIABLES
window_size = 395
window_threshold = 0.9
smallwindow_size = 48
smallwindow_threshold = 0.7
smallwindow_step = 23
# elby testing- set back to original fine-grained vals
# you can change them back to 395, .9, 131, .7, 3
num_scans = (window_size - smallwindow_size) // smallwindow_step + 1
print("num_scans is", num_scans) # p sure this is right

### LOCATION FROM TINY BALL TO BIG PICTURE
def getLocation(x,y,box):
  # todo: what if there are more/less than 8 big boxes?
  if box < 5:
    xreal = x + window_size * (box-1)
    yreal = y
  else:
    xreal = x + window_size * (box-5)
    yreal = window_size + y
  return (xreal, yreal)

# alternatively
def getLoc(x,y,box):
  return (x + window_size * ((box - 1) % 4), y + window_size * (box > 4))

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
      # todo: stepSize should be smallwindow_step, not smallwindow_size, right????
      if smallwindow.shape[0] != winH or smallwindow.shape[1] != winW:
        continue
      count += 1

      # todo: run everything in one session (move the iterator within the session, in the other file)
      # todo: since we don't save to jpeg anymore, we don't need the "subimages" folder one level back, right?
      predictions = classify2.isball(smallwindow) #smallwindow is nparray (smallwindow_size,smallwindow_size,3)

      print("x and y transformed are", int(x/smallwindow_step),int(y/smallwindow_step))
      heatmap[int(x/smallwindow_step),int(y/smallwindow_step),:] = predictions[0]
      # todo: these should all be smallwindow_step instead of smallwindow_size too, right?

    f = open("elbytest/heatmap%d.txt" % i,"w")
    f.write(str(heatmap))
    f.close()

    # heatmap for a bigsmall done
    heatmap = heatmap * 255
    # todo: insert to fullheatmap


    # todo: on windows, use interpolation='none' to stop blurring effect
    plt.imshow(heatmap[:,:,0])
    # todo: gotta get the colorbar to work! for beaut vizs
    #plt.colorbar(heatmap)
    plt.savefig("elbytest/heatmap_neither%d" % i)

    plt.imshow(heatmap[:,:,1])
    #plt.colorbar(heatmap)
    plt.savefig("elbytest/heatmap_solid%d" % i)

    plt.imshow(heatmap[:,:,2])
    #plt.colorbar(heatmap)
    plt.savefig("elbytest/heatmap_stripe%d" % i)
    # todo: get rid of elbytest
    assert(0==1)