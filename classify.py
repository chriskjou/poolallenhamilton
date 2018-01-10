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


### LOCATION FROM TINY BALL TO BIG PICTURE
def getLocation(x,y,box):
  if box < 5:
    xreal = x + 395 * (box-1)
    yreal = y
  else:
    xreal = x + 395 * (box-5)
    yreal = 395 + y
  return (xreal, yreal)

### GLOBAL VARIABLES
window_threshold = 0.9
small_window_size = 131
small_window_threshold = 0.7
number_of_scans = 3

### TENSORFLOW SETUP
# disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### SLIDER SETUP
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(winW, winH) = (395, 395)

# cut pool table into 8 images
eight_images = []
count = 0
for (x, y, window) in sliding_window(image, stepSize=395, windowSize=(winW, winH)):
  if window.shape[0] != winH or window.shape[1] != winW:
    continue
  eight_images.append(window)
  count+=1

# loads graph
with tf.gfile.FastGFile("../logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g1 = tf.import_graph_def(graph_def, name='g1')


# classify bigwindow
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("../logs/trained_labels.txt")]

has_ball = []
for i in range(len(eight_images)):
  with tf.Session(graph=g1) as sess:
      softmax_tensor = sess.graph.get_tensor_by_name('g1/final_result:0')
      predictions = sess.run(softmax_tensor, {'g1/DecodeJpeg:0': eight_images[i]})

      # [hasball, noball]
      print(predictions)

      has_ball.append(predictions[0][0] > window_threshold)

print(has_ball)

# classify smallwindow
for i in range(len(has_ball)):
  if has_ball[i]:

    heatmap = np.zeros((number_of_scans,number_of_scans,3))

    big_image = eight_images[i]
    print("bigimage %d has a ball" % i)

    count = 0
    (winW, winH) = (small_window_size, small_window_size) #131
    for (x, y, smallwindow) in sliding_window(big_image, stepSize=small_window_size, windowSize=(winW, winH)):
      if smallwindow.shape[0] != winH or smallwindow.shape[1] != winW:
        continue
      count+=1

      predictions = classify2.isball(smallwindow) #smallwindow is nparray (48,48,3)

      print("x and y transformed are", int(x/small_window_size),int(y/small_window_size))
      heatmap[int(x/small_window_size),int(y/small_window_size),:] = predictions[0]

    f = open("testfile%d" % i,"w")
    f.write(str(heatmap))
    f.close()

    # heatmap for a bigsmall done
    heatmap = heatmap * 255

    plt.imshow(heatmap[:,:,0])
    #plt.colorbar(heatmap)
    plt.savefig("heatmap_neither")

    plt.imshow(heatmap[:,:,1])
    #plt.colorbar(heatmap)
    plt.savefig("heatmap_solid")

    plt.imshow(heatmap[:,:,2])
    #plt.colorbar(heatmap)
    plt.savefig("heatmap_stripe")

