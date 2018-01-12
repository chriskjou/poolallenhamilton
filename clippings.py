import numpy as np
import glob

# Make sure the Seagate hard drive is plugged in
csvs = glob.glob('../../../../media/wintern18/Seagate Expansion Drive/BALS_mp4s_csvs/*.csv')
for csv in csvs:
	name = vid.split('BALS_2017-')[-1].split('.mp4')[0]