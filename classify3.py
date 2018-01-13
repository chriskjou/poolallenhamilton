import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback

from keras.models import load_model

import cv2
import argparse
import time
import numpy as np
import utils2
import matplotlib.pyplot as plt

# image = utils2.load_image('neither.jpeg')

# load model
model = load_model('models/model4.h5')

def isball(image):
  image = image.reshape((1,24,24,3))
  predictions = model.predict(image)
  return predictions[0]












