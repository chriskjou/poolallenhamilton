import glob
import subprocess

photos = glob.glob('real-time-test/*.jpg')

for photo in photos:
	subprocess.call(['python', 'classify.py', '--image', photo])