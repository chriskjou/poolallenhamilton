import numpy as np
import pandas as pd
import glob
import csv

# change to suit your own needs
folders = glob.glob('/home/wintern18/Desktop/cropped_images_ff/*')
folders.sort()

# type, x, y
def get_image_data(csvpath):
    df = pd.read_csv(csvpath, names=['balltype','x','y'])
    # Scale by appropriate factor
    df['x'] *= 395/16 
    df['y'] *= 395/16
    # Eliminate neithers
    df = df[df['balltype'] != 'neither']
    # Eliminate duplicate cueballs by picking the first one
    df = df.drop(df[df['balltype']=='cue'].index[1:])
    df = df.drop(df[df['balltype']=='eight_ball'].index[1:])
    return df

def get_meta(gamepath):
    metacsv = glob.glob(gamepath+'/*_meta.csv')[0]
    with open(metacsv) as metafile:
        reader = csv.reader(metafile)
        _ = next(reader)
        meta = next(reader)
    return meta

# added a lot of nonsense into this one
# TODO: duplicate later frames! (or just give it the second half of the game?)
# type, x, y, frame, winner (1 if stripes wins)
def get_game_data(gamepath):
    def append_frame(csvpath, i):
        df = get_image_data(csvpath)
        df['frame'] = i
        if 'cue' not in df['balltype'].values: # delete all this nonsense later
            return None
        cue = df[df['balltype']=='cue'][['x','y']].iloc[0]
        df = df[df['balltype']!='cue']
        if not len(df):
            return None
        df['cuex'] = cue.x
        df['cuey'] = cue.y
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

def get_polar_data(gamepath):
    def append_frame(csvpath, i):
        df = get_image_data(csvpath)
        df['frame'] = i
        if 'cue' not in df['balltype'].values: # delete all this nonsense later
            return None
        cue = df[df['balltype']=='cue'][['x','y']].iloc[0]
        df = df[df['balltype']!='cue']
        if not len(df):
            return None
        df = df.merge(df.apply(lambda s: diffseries(s,cue),axis=1),left_index=True,right_index=True)
        return df
    nframes = len(glob.glob(gamepath+'/frame*'))//2
    csvs = [gamepath+'/frame'+str(i+1) for i in range(nframes)]
    df = pd.concat([append_frame(csvs[i],i) for i in range(len(csvs))], ignore_index=True)
    df.drop([0,1,2]) # drop first 3 frames
    meta = get_meta(gamepath)
    winner = int(meta[2]==meta[3])
    df['winner'] = winner
    return df

# type, x, y, frame, winner, game, (cuex, cuey, diff nonsense)
def get_data(start, end):
    def append_game(i):
        df = get_game_data(folders[i])
        df['game'] = i
        return df
    return pd.concat([append_game(i) for i in range(start, end)], ignore_index=True)

################

# numstripe, numsolid, winner, game
def get_data1(start, end):
    df = pd.DataFrame(columns=['numstripe','numsolid','winner', 'game'])
    for i in range(start, end):
        gamepath = folders[i]
        meta = get_meta(gamepath)
        winner = int(meta[2]==meta[3])
        nframes = len(glob.glob(gamepath+'/frame*'))//2
        csvs = [gamepath+'/frame'+str(i+1) for i in range(nframes)]
        for csv in csvs[3:]: # drop first 3 frames
            imgdf = get_image_data(csv)
            if imgdf.empty:
                continue
            ct = imgdf['balltype'].value_counts()
            newrow = np.zeros(4)
            newrow[0] = ct.stripes if 'stripes' in ct.index else 0
            newrow[1] = ct.solids if 'solids' in ct.index else 0
            newrow[2] = winner
            newrow[3] = i
            if newrow[0] > 7 or newrow[1] > 7:
                continue # throw out obvious mistakes (and it's 7 this time)
            df.loc[len(df)] = newrow
    return df

################

pockets = [[0,0],[790,0],[1580,0],[0,790],[790,790],[1580,790]]

# from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# d2
# lower is better
def diff(ball):
    d = 2000 # artificially high number
    ball = ball[['x','y']].values
    for pocket in pockets:
        d2 = np.linalg.norm(pocket - ball)
        d = d2 if d2 < d else d2
    return d

# analytical difficulty (other formulae also exist)
# higher is better
def diff1(ball, cue):
    d = 0 # artificially low number
    ball = ball[['x','y']].values
    if ball.tolist() in pockets:
        ball -= 1 # avoid div by 0 error
    cue = cue.values
    for pocket in pockets:
        theta = angle_between(cue - ball, pocket - ball)
        if theta > 1.58:
            continue
        d1 = np.linalg.norm(ball - cue)
        d2 = np.linalg.norm(pocket - ball)
        diff = np.cos(theta) / d1 / d2
        if diff > d:
            d = diff
            data = [theta,d1,d2]
    return d * 10 ** 6
    # These values are tiny! hence I multiply d by a large constant before returning?
    # or I could normalize the values afterward

    # TODO: what about obstacle balls?

# same exact thing
def diffseries(ball, cue):
    d = 0 # artificially low number
    ball = ball[['x','y']].values
    if ball.tolist() in pockets:
        ball -= 1 # avoid div by 0 error
    cue = cue.values
    for pocket in pockets:
        theta = angle_between(cue - ball, pocket - ball)
        if theta > 1.58:
            continue
        d1 = np.linalg.norm(ball - cue)
        d2 = np.linalg.norm(pocket - ball)
        diff = np.cos(theta) / d1 / d2
        if diff > d:
            d = diff
            data = [theta,d1,d2]
    return pd.Series({'theta':theta, 'd1':d1,'d2':d2})

# Just eyeballed these thresholds
# 0 for easy, 1 for med, 2 for hard
def zone(ball):
    d = diff(ball)
    if d < 200:
        return 0
    elif d < 430:
        return 1
    else:
        return 2

# easystripe, easysolid, medstripe, medsolid, hardstripe, hardsolid, winner, game
def get_data2(start, end):
    df = pd.DataFrame(columns = ['easystripe','easysolid','medstripe','medsolid','hardstripe','hardsolid','winner','game'])
    cols = [('stripes',0),('solids',0),('stripes',1),('solids',1),('stripes',2),('solids',2)]
    for i in range(start, end):
        gamepath = folders[i]
        meta = get_meta(gamepath)
        winner = int(meta[2]==meta[3])
        nframes = len(glob.glob(gamepath+'/frame*'))//2
        csvs = [gamepath+'/frame'+str(i+1) for i in range(nframes)]
        if len(csvs) < 14:
            continue
        for csv in csvs[3:-10]: # drop first 3 frames
            imgdf = get_image_data(csv)
            if imgdf.empty:
                continue
            imgdf['diff'] = imgdf.apply(zone, axis=1)
            imgdf = imgdf.groupby(['balltype','diff']).count()
            newrow = np.zeros(len(cols)+2)
            newrow[-2] = winner
            newrow[-1] = i
            for x in range(len(cols)):
                newrow[x] = imgdf.loc[cols[x]][0] if cols[x] in imgdf.index else 0
            if sum(newrow[0:3]) > 7 or sum(newrow[3:6]) > 7:
                continue # throw out obvious mistakes (and it's 7 this time)
            df.loc[len(df)] = newrow
    return df

# numstripes, numsolids, d2 for each stripe, d2 for each solid, winner, game
# each ball ordered by difficulty
def get_data3(start, end):
    df = pd.DataFrame(columns=['numstripe','numsolid']+['stripe'+str(i) for i in range(7)]+['solid'+str(i) for i in range(7)]+['winner','game'])
    for i in range(start, end):
        gamepath = folders[i]
        meta = get_meta(gamepath)
        winner = int(meta[2]==meta[3])
        nframes = len(glob.glob(gamepath+'/frame*'))//2
        csvs = [gamepath+'/frame'+str(i+1) for i in range(nframes)]
        if len(csvs) < 14:
            continue
        for csv in csvs[-10:]: # drop first 3 frames
            imgdf = get_image_data(csv)
            if imgdf.empty:
                continue
            if 'cue' not in imgdf['balltype'].values:
                continue
            imgdf = imgdf[imgdf['balltype']!='cue']
            cue = imgdf[imgdf['balltype']=='cue'][['x','y']].iloc[0]
            imgdf['diff'] = imgdf.apply(lambda x: diff1(x,cue), axis=1)
            # imgdf['diff'] = imgdf.apply(diff, axis=1)
            stripedf = imgdf[imgdf['balltype']=='stripes'].sort_values(by='diff')
            soliddf = imgdf[imgdf['balltype']=='solids'].sort_values(by='diff')
            if len(stripedf) > 7 or len(soliddf) > 7:
                continue # throw out obvious mistakes (and it's 7 this time)
            newrow = np.zeros(18)
            newrow[-2] = winner
            newrow[-1] = i
            newrow[:2] = [len(stripedf),len(soliddf)]
            newrow[2:(2+len(stripedf))] = stripedf['diff']
            newrow[9:(9+len(soliddf))] = soliddf['diff']
            df.loc[len(df)] = newrow
    return df

#####

# TODO: still wanna throw out frames so liberally?
def get_dataduncan(start, end):
    X = np.zeros((1,16,2))
    Y = np.zeros(1)
    for i in range(start, end):
        gamepath = folders[i]
        meta = get_meta(gamepath)
        winner = int(meta[2]==meta[3])
        nframes = len(glob.glob(gamepath+'/frame*'))//2
        csvs = [gamepath+'/frame'+str(i+1) for i in range(nframes)]
        if len(csvs) < 10:
            continue
        lastballs = (7,7)
        for csv in csvs[3:]: # drop first 3 frames
            newrow = np.zeros((16,2)) # cartesian... zeros has another interpretation
            imgdf = get_image_data(csv)
            stripedf = imgdf[imgdf['balltype']=='stripes'].sort_values(by='x') # arbitrary
            soliddf = imgdf[imgdf['balltype']=='solids'].sort_values(by='x')
            if len(stripedf) > 7 or len(soliddf) > 7:
                continue # throw out obvious mistakes (and it's 7 this time)
            newrow[:len(soliddf),:] = soliddf[['x','y']]
            newrow[7:(7+len(stripedf)),:] = stripedf[['x','y']]
            if 'cue' not in imgdf['balltype'].values or 'eight_ball' not in imgdf['balltype'].values:
                continue
            newrow[14,:] = imgdf[imgdf['balltype']=='cue'][['x','y']]
            newrow[15,:] = imgdf[imgdf['balltype']=='eight_ball'][['x','y']]
            X = np.concatenate((X,newrow[np.newaxis]))
            Y = np.append(Y,lastballs[1]-len(stripedf)-lastballs[0]+len(soliddf))
    X = X.reshape(X.shape[0],32)
    Y = Y.clip(-1,1)
    X = X[1:]
    Y = Y[1:]
    return (X,Y)




