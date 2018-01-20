# Extracts images from video every 30 seconds using ffmpeg

import numpy as np
import glob
import subprocess

# Make sure the Seagate hard drive is plugged in
vids = glob.glob('../../../../media/wintern18/Seagate Expansion Drive/BALS_mp4s_csvs/*.mp4')
# vids = vids[99:]
assert(vids is not [])

for vid in vids:
	vshort = vid.split('BALS_2017-')[-1].split('.mp4')[0]
	print(vshort)
	name = '../cropped_images_ff/' + vshort
	subprocess.call(['mkdir ../cropped_images_ff/'+vshort], shell=True)
	subprocess.call(['ffmpeg', '-i', vid, '-vf', 'fps=1/30', '-q:v', '1', name+'/frame%d.jpg'])
		# '-f', 'mjpeg', 'frame%d.jpg'])
		# '-filter:v', 'crop=1580:790:40:60', '-q:v', '1', name])
		# filtering like that causes every image to be saved
	# how do I crop? well, what format does classify need? (what format would be ideal?)

# Other attempts
# process = subprocess.Popen(['ffmpeg -ss 10 -i BALS_2017-02-10-1017.mp4 -vf 1 -q:v 1 foo.jpg'], 
# 	universal_newlines=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# # stdout, stderr = process.communicate()

# while True:    
#         print(process.stderr.readline().rstrip('\r\n'))