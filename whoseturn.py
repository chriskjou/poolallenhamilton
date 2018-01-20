import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from readercleaner import get_image_data

# files in cropped_images_new are properly 0-indexed, unlike cropped_images_ff
folders = glob.glob('/home/wintern18/Desktop/cropped_images_new/*')
folders.sort()

# balltype, x, y, frame, winner
def get_game_data(gamepath):
    def append_frame(csvpath, i):
        df = get_image_data(csvpath)
        df['frame'] = i
        if 'cue' not in df['balltype'].values:
            return None # skip if no cue (it's a scratch)
        return df
    nframes = len(glob.glob(gamepath+'/frame*'))//2
    csvs = [gamepath+'/frame'+str(i) for i in range(nframes)]
    df = pd.concat([append_frame(csvs[i],i) for i in range(1,len(csvs))], ignore_index=True)
    # ignore first frame
    meta = get_meta(gamepath)
    winner = int(meta[2]==meta[3])
    df['winner'] = winner
    return df

def switchturn(x):
    if x == 0:
        return 1
    elif x == 1:
        return 0
    else:
        return -1

# Oh, an idea! Probabilistic model for turns, in coin-flipping style!
# Because it's only feasible to think about nsol/nstr rn, with noisy data
# How can we measure table similarity between 2 states?
# CRAP! If my turn info is based on changes in nsol/nstr, and I'm using it
# to predict changes in nsol/nstr, what's the point?
# If I did get a turn-inferrer, I could integrate it into dataduncanp
def whoseturn(gamedf):
	nframes = df.iloc[-1].frame
	turns = np.zeros(nframes)-1 # -1 for unsure
	lastturn = -1
	for i in range(nframes-1):
		ct = df[df['frame']==i].value_counts()
		nsol = ct.solids if 'solids' in ct.index else 0
		nstr = ct.stripes if 'stripes' in ct.index else 0
		ct = df[df['frame']==i+1].value_counts()
		nsoln = ct.solids if 'solids' in ct.index else 0
		nstrn = ct.stripes if 'stripes' in ct.index else 0
		if nsol==nsoln and nstr==nstrn:
			turns[i+1] = switchturns(turns[i])
		else:
			turns[i+1] = np.argmax([nsol-nsoln,nstr-nstrn])
			if turns[i] == -1:
				turns[i] = np.argmax([nsol-nsoln,nstr-nstrn])
	return turns



# I want an array of turns