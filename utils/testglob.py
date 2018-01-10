import glob
vids = glob.glob('e:/BALS_mp4s_csvs/*.mp4')

vid = vids[0].split('BALS_2017-')[-1].split('.mp4')[0]
print(vid)