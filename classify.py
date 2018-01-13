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

image = cv2.imread(args["image"])
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(winW, winH) = (smallwindow_size, smallwindow_size)

# ### cutting for now
# # I had difficulty getting ffmpeg to crop, so I'll do it here.
# if image.shape != (1580,790):
#   image = image[40:1620,60:850]

# cut pool table into 8 images

# eight_images = []

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

heatmap = np.zeros((67,33,3)) # np.zeros((154,75,3)) #np.zeros((67,33,3))
count = 0
start = time.time()
for (x, y, window) in sliding_window(new_image, stepSize=smallwindow_step, windowSize=(winW, winH)):
  if window.shape[0] != winH or window.shape[1] != winW:
    continue
  window = utils2.skimage.transform.resize(window, (24, 24))
  predictions = classify3.isball(window)

  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),0] = truncate(predictions[0],3)
  heatmap[int(x/smallwindow_step), int(y/smallwindow_step),1] = truncate(predictions[1],3)
  count+=1
  #print(predictions)
  # if predictions[1] > smallwindow_threshold:
  #   plt.imshow(window)
  #   plt.show()

end = time.time()
print("RUNTIME", end-start)
heatmap = heatmap.transpose(1,0,2)
plt.imshow(heatmap)
#plt.show()

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
  xcoord = int(ball[0] * 23)
  ycoord = int(ball[1] * 23)
  print("XCOORD:", xcoord)
  print("YCOORD:", ycoord)
  small_image = new_image[max(ycoord, 0): min(ycoord + 48, 780), max(xcoord, 0): min(xcoord + 48, 1580)]
  predictions = classify2.isball(small_image)
  small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
  cv2.imwrite("interestingnew%d.jpg" % interesting_count, small_image)
  print("WHICH BALL:", ball)
  print(predictions)

  # labels = [blacksolid bluesolid bluestripe greensolid greenstripe neither
  # orangesolid orangestripe pinksolid pinkstripe purplesolid purplestripe
  # redsolid redstripe white yellowsolid yellowstripe]

  # labels = [black cue neither solids stripes]

  labels = ['eight_ball', 'cue', 'neither', 'solids', 'stripes'] # consist with predictions
  # labels = ['blacksolid', 'bluesolid', 'bluestripe', 'greensolid', 'greenstripe', 'neither', 'orangesolid', 'orangestripe', 'pinksolid', 'pinkstripe', 'purplesolid', 'purplestripe', 'redsolid', 'redstripe', 'white', 'yellowsolid', 'yellowstripe']
  maxnum = predictions[0][0]
  index = 0
  for i in range(1, len(labels)):
      if maxnum < predictions[0][i]:
        maxnum = predictions[0][i]
        index = i

  names.append(labels[index])
  interesting_count+=1
  """
  if index == 0:
    names.append('neither')
  elif index == 1:
    names.append('solids')
  else:
    names.append('stripes')
  """
"""
  maxnum = predictions[0][0]
  index = 0
  for i in range(1, 17):
      if maxnum < predictions[0][i]:
        maxnum = predictions[0][i]
        index = i

  if index == 0:
    names.append('blacksolid')
  elif index == 1:
    names.append('bluesolid')
  elif index == 2:
    names.append('bluestripe')
  elif index == 3:
    names.append('greensolid')
  elif index == 4:
    names.append('greenstripe')
  elif index == 5:
    names.append('neither')
  elif index == 6:
    names.append('orangesolid')
  elif index == 7:
    names.append('orangestripe')
  elif index == 8:
    names.append('pinksolid')
  elif index == 9:
    names.append('pinkstripe')
  elif index == 10:
    names.append('purplesolid')
  elif index == 11:
    names.append('purplestripe')
  elif index == 12:
    names.append('redsolid')
  elif index == 13:
    names.append('redstripe')
  elif index == 14:
    names.append('white')
  elif index == 15:
    names.append('yellowsolid')
  else:
    names.append('yellowstripe')
    """
  #interesting_count+=1



plt.imshow(heatmap)
for i in range(len(balls)):
  plt.plot(int(balls[i][0]), int(balls[i][1]), 'ro')
  plt.text(int(balls[i][0]), int(balls[i][1]), names[i])
cv2.imwrite("labels%d.jpg" % 0, image)
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

"""
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
"""
# TODO: from total list of balls given by thresholder, annotate raw images
