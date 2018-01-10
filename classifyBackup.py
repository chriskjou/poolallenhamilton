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

### TENSORFLOW SETUP
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]

### SLIDER SETUP

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(winW, winH) = (395, 395)

###

# Cut pool table into 8 images
count = 0
for (x, y, window) in sliding_window(image, stepSize=395, windowSize=(winW, winH)):
  if window.shape[0] != winH or window.shape[1] != winW:
    continue
  count+=1
  cv2.imwrite("8images/window%d.jpeg" % count, window)

img_paths = glob.glob('./8images/*.jpeg')

images = []
for img_path in img_paths:
  images.append(tf.gfile.FastGFile(img_path, 'rb').read())

# loads graph
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g1 = tf.import_graph_def(graph_def, name='g1')

has_ball = []
# classify windows
for image_data in images:
  with tf.Session(graph=g1) as sess:


      softmax_tensor = sess.graph.get_tensor_by_name('g1/final_result:0')
      predictions = sess.run(softmax_tensor, {'g1/DecodeJpeg/contents:0': image_data})

      # [hasball, noball]
      print(predictions)
      print(predictions[0][0])

      top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
      for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

      has_ball.append(predictions[0][0] > window_threshold)

print(has_ball)

### NEW TENSORFLOW SETUP



# Loads label file, strips off carriage return
# label_lines = [line.rstrip() for line
#                    in tf.gfile.GFile("logssmall/trained_labels.txt")]

# with tf.gfile.FastGFile("logssmall/trained_graph.pb", 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')

for i in range(len(has_ball)):
  if has_ball[i]:

    heatmap = np.zeros((16,16,3))

    big_image = cv2.imread("8images/window%d.jpeg" % (i+1))
    count = 0
    (winW, winH) = (48, 48) #131
    for (x, y, smallwindow) in sliding_window(big_image, stepSize=23, windowSize=(winW, winH)):
      if smallwindow.shape[0] != winH or smallwindow.shape[1] != winW:
        continue
      count+=1
      cv2.imwrite("subimages/subwindow%d.jpeg" % count, smallwindow)
      image_path = "subimages/subwindow%d.jpeg" % count

      predictions = classify2.isball(image_path)

      print("x and y transformed are", int(x/23),int(y/23))
      heatmap[int(x/23),int(y/23),:] = predictions[0]

    f = open("testfile%d" % i,"w")
    f.write(str(heatmap))
    f.close()

    # heatmap for a bigsmall done
    heatmap = heatmap * 255
    plt.imshow(heatmap[:,:,0])
    plt.legend()
    plt.colorbar(heatmap)
    plt.savefig("heatmap_neither")
    plt.imshow(heatmap[:,:,1])
    plt.legend()
    plt.colorbar(heatmap)
    plt.savefig("heatmap_solid")
    plt.imshow(heatmap[:,:,2])
    plt.legend()
    plt.colorbar(heatmap)
    plt.savefig("heatmap_stripe")













# img_paths = glob.glob('./subimages/*.jpeg')

# images = []
# for img_path in img_paths:
#   images.append(tf.gfile.FastGFile(img_path, 'rb').read())






















