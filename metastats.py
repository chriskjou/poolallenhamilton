import numpy as np
import pandas as pd
import glob
import csv
from readercleaner import get_image_data, get_meta

# % stripe wins
folders = glob.glob('/home/wintern18/Desktop/cropped_images_ff/*')
folders.sort()
stripewins = 0
for gamepath in folders:
	meta = get_meta(gamepath)
	if meta[2] == meta[3]:
		stripewins += 1
print('stripes won %d of %d games' % stripewins, len(folders))

# avg game len
totalframes = []
for gamepath in folders:
	nframes = len(glob.glob(gamepath+'/frame*'))//2
	totalframes.append(nframes)
print('average game length was %d' np.average(totalframes))

# avg nsol/nstr over time
balls = np.zeros((50,2))

for i in range(200):
    gamepath = folders[i]
    nframes = len(glob.glob(gamepath+'/frame*'))//2
    csvs = [gamepath+'/frame'+str(i+1) for i in range(nframes)]
    for ci in range(len(csvs)): # drop first 3 frames
        imgdf = get_image_data(csvs[ci])
        ct = imgdf['balltype'].value_counts()
        balls[ci,0] += ct.solids if 'solids' in ct.index else 0
        balls[ci,1] = ct.stripes if 'stripes' in ct.index else 0

plt.plot(balls[:,0])
plt.plot(balls[:,1])
plt.legend(['# solids','# stripes'])
plt.title('Average Number of Balls Left After x Frames')
plt.show()
