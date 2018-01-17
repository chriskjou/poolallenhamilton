import cv2
import glob
print(cv2.__version__)

vids = glob.glob('1.mp4')
# vids = vids[99:]
for vid in vids:
	vidcap = cv2.VideoCapture(vid)
	vid = vid.split('.mp4')[0]
	print(vid)
	success,image = vidcap.read()
	count = 0
	success = True
	while success:
	    for i in range(900):
	        success,image = vidcap.read()
	    print('Read a new frame:', success)
	    if success:
	        crop_img = image[60:850,40:1620]
	        name = 'real-time-test/' + str(vid) + 'frame%d.jpg'
	        cv2.imwrite(name % count, crop_img)
	        count += 1
