import numpy as np
import matplotlib.pyplot as plt

def get_game_data(gamepath):
    def append_frame(csvpath, i):
        df = get_image_data(csvpath)
        df['frame'] = i
        if 'cue' not in df['balltype'].values:
            return None # skip if no cue
        cue = df[df['balltype']=='cue'][['x','y']].iloc[0]
        df = df[df['balltype']!='cue']
        if not len(df):
            return None # skip if only a cue ball
        df['cuex'], df['cuey'] = cue.x, cue.y # for "simple" network
        df['diff'] = df.apply(lambda x: diff1(x,cue), axis=1)
        return df
    nframes = len(glob.glob(gamepath+'/frame*'))//2
    csvs = [gamepath+'/frame'+str(i+1) for i in range(nframes)]
    df = pd.concat([append_frame(csvs[i],i) for i in range(len(csvs))], ignore_index=True)
    df.drop([0,1,2]) # drop first 3 frames
    meta = get_meta(gamepath)
    winner = int(meta[2]==meta[3])
    df['winner'] = winner
    return df

