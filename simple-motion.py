import cv2
import glob
import numpy as np
import subprocess
print(cv2.__version__)

vids = glob.glob('../../../../media/wintern18/Seagate Expansion Drive/BALS_mp4s_csvs/*.mp4')
threshold = 10000
step_size = 30
minimum_cap = 6000
diffs = 2000
distances = [0]

def sameImage(first, second):
	zeroes = np.zeros((922, 1640, 3))
	# zeroes = np.zeros((790,1580,3))
	newcheck = cv2.absdiff(first, second)
	distance = np.linalg.norm(newcheck - zeroes)
	if distance < threshold:
		return True
	return False

def getDistance(first, second):
	zeroes = np.zeros((922, 1640, 3))
	# zeroes = np.zeros((790,1580,3))
	newcheck = cv2.absdiff(first, second)
	distance = np.linalg.norm(newcheck - zeroes)
	return distance

def cropandsave(image, count, vid):
	crop_img = image[60:850,40:1620]
	#name = '../memorable/' + vid + 'motionframe%d.jpg'
	name = '../cropped_images_new/' + vid+ '/frame%d.jpg'
	cv2.imwrite(name % count, crop_img)

def getImages(vids):
	for vid in vids:
		vidcap = cv2.VideoCapture(vid)
		vid = vid.split('BALS_2017-')[-1].split('.mp4')[0]
		subprocess.call(['mkdir ../cropped_images_new/'+vid], shell=True)
		# vid = vid.split('.mp4')[0]
		success,image = vidcap.read()
		pictures = [image]
		count = 0
		crop_img = image
		second = 0

		# only takes snapshots when there is no motion
		while success: 
			# get two frame one at each second
			second+=1
			for i in range(step_size):
				success, image = vidcap.read()
			second+=1
			for i in range(step_size):
				success, nextimage = vidcap.read()

			if success:
				# no movement
				distances.append(getDistance(image, nextimage))
				if distances[-1] < minimum_cap and distances[-2] - distances[-1] > diffs:
					if not sameImage(pictures[-1], image):
						cropandsave(image, count, vid)
						count+=1
						pictures.append(nextimage)
getImages(vids)

