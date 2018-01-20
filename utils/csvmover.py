# Moves the metadata csvs from hard drive to local machine

import glob
import shutil
import os

csvs = glob.glob('../../../../media/wintern18/Seagate Expansion Drive/BALS_mp4s_csvs/*.csv')

for csv in csvs:
	name = csv.split('BALS_2017-')[-1]
	folder = name.split('_meta.csv')[0]
	print(name)
	if os.path.isdir('../cropped_images_new/'+folder):
		shutil.copyfile(csv,'../cropped_images_new/'+folder+'/'+name)
	else:
		print('no folder for this one!')
