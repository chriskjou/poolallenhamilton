# usage: python simple-motion.py --video [PATH TO VIDEO FOLDER]/*.mp4 --photo [PATH TO PHOTO FOLDER]
# example: python simple-motion.py --video ../videos/*.mp4 --photo /tosave

import cv2
import glob
import numpy as np
import subprocess
import argparse
print(cv2.__version__)

### ARGUMENT SETUP ###
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="folder path to the videos")
ap.add_argument("--photo", required=True, help="folder path to where you want to save photos")
args = vars(ap.parse_args())
image = args["video"]
photos = args["photo"]

vids = glob.glob(image)

### PARAMETERS ###
threshold = 10000 # frames with Euclidean distance above this number considered in 'motion'
step_size = 30 # one frame in every 'step-size' number of frames will be analyzed
minimum_cap = 6000 # frames must be less than this number to be considered not in 'motion'
diffs = 2000 # a frame not in motion must also come after a frame in 'motion' that has a Euclidean distance this much more
distances = [0] # initializes array of Euclidean distances

## FUNCTIONS ###

# determining whether image should be saved
def sameImage(first, second):
	# change dimensions of np array according to image size
	# example: if image is 790 by 1580, replace 'np.zeroes((922, 1640, 3))' with 'np.zeroes((790.1580,3))'
	zeroes = np.zeros((922, 1640, 3))
	newcheck = cv2.absdiff(first, second)
	distance = np.linalg.norm(newcheck - zeroes)
	return distance < threshold

# getting the Euclidean distance
def getDistance(first, second):
	# change dimensions of np array according to image size
	# example: if image is 790 by 1580, replace 'np.zeroes((922, 1640, 3))' with 'np.zeroes((790.1580,3))'
	zeroes = np.zeros((922, 1640, 3))
	newcheck = cv2.absdiff(first, second)
	distance = np.linalg.norm(newcheck - zeroes)
	return distance

# crops and saves image
def cropandsave(image, count, vid):
	# if you don't want to crop the image, remove the following line
	crop_img = image[60:850,40:1620]
	name = photos + vid+ '/frame%d.jpg'
	cv2.imwrite(name % count, crop_img)

def getImages(vids):
	for vid in vids:
		vidcap = cv2.VideoCapture(vid)
		vid = vid.split('.mp4')[0]
		success,image = vidcap.read()
		pictures = [image]
		count = 0

		# only takes snapshots when there is no motion
		while success: 
			# get two frames one at each second
			second+=1
			for i in range(step_size):
				success, image = vidcap.read()
			second+=1
			for i in range(step_size):
				success, nextimage = vidcap.read()

			if success:
				# no movement
				distances.append(getDistance(image, nextimage))
				# ensures motion assumptions
				if distances[-1] < minimum_cap and distances[-2] - distances[-1] > diffs:
					# checks to make sure the image isn't similar to last image already saved
					if not sameImage(pictures[-1], image):
						cropandsave(image, count, vid)
						count+=1
						pictures.append(nextimage)
getImages(vids)
