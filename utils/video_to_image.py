import cv2
import glob
print(cv2.__version__)

# Make sure the Seagate hard drive is plugged in
vids = glob.glob('E:/BALS_mp4s_csvs/*.mp4')
for vid in vids:
	vidcap = cv2.VideoCapture(vid)
	vid = vid.split('BALS_2017-')[-1].split('.mp4')[0]
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
	        name = '../cropped_images/' + vid + 'frame%d.jpg'
	        cv2.imwrite(name % count, crop_img)
	        count += 1
