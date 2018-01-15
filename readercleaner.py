import numpy as np
import pandas as pd
import glob
import csv

# untested
folders = glob.glob('../../cropped_images_ff/*').sort()

# type, x, y
def get_image_data(csvpath):
    df = pd.read_csv(csvpath, names=['balltype','x','y'])
    # Scale by appropriate factor
    df['x'] *= 395/16 
    df['y'] *= 395/16
    # Eliminate neithers
    df = df[df['balltype'] != 'neither']
    # Eliminate duplicate cueballs by averaging their position
    cueballs = df[df['balltype']=='cue']
    if not cueballs.empty:
        df = df[df['balltype'] != 'cue']
        cuemean = cueballs.mean()
        df = df.append({'balltype':'cue','x':cuemean.x,'y':cuemean.y}, ignore_index=True)
    eightballs = df[df['balltype']=='eight']
    if not eightballs.empty:
        df = df[df['balltype'] != 'eight']
        eightmean = eightballs.mean()
        df = df.append({'balltype':'eight','x':eightmean.x,'y':eightmean.y}, ignore_index=True)
    return df

def get_meta(gamepath):
    metacsv = glob.glob(gamepath+'*_meta.csv')[0]
    with open(metacsv) as metafile:
        reader = csv.reader(metafile)
        _ = next(reader)
        meta = next(reader)
    return meta

# type, x, y, frame, winner (1 if stripes wins)
def get_game_data(gamepath):
    def append_frame(csvpath, i):
        df = get_image_data(csvpath)
        df['frame'] = i
        return df
    csvs = glob.glob(gamepath+'/frame*.csv')
    df = pd.concat([append_frame(csvs[i],i) for i in range(len(csvs))], ignore_index=True) # untested
    meta = get_meta(gamepath)
    winner = int(meta[2]==meta[3])
    df['winner'] = winner

# type, x, y, frame, winner, game
def get_data(start, end):
    def append_game(i):
        df = get_game_data(folders[i])
        df['game'] = i
        return df
    return pd.concat([append_game(i) for i in range(start, end)], ignore_index=True) # untested



# numstripe, numsolid, winner
def get_data1(start, end):
    df = pd.DataFrame(columns=['numstripe','numsolid','winner'])
    for i in range(start, end):
        gamepath = folders[i]
        meta = get_meta(gamepath)
        winner = int(meta[2]==meta[3])
        csvs = glob.glob(gamepath+'/frame*.csv')
        for csv in csvs:
            imgdf = get_image_data(csv)
            ct = imgdf['balltype'].value_counts()
            df.append({'numstripe':ct.stripe,'numsolid':ct.solid,'winner':winner}, ignore_index=True)
    return df

# d2
def closest_pocket(ball):
    d = 2000
    for pocket in [(0,0),(790,0),(1580,0),(0,790),(790,790),(1580,790)]:
        d_p = np.sqrt((ball['x']-d[0])**2+(ball['y']-d[1])**2)
        d = d_p if d_p < p else d_p
    return d

# 0 for easy, 1 for med, 2 for hard
def zone(ball):
    d = closest_pocket(ball)
    if d < 20:
        return 0
    elif d < 50:
        return 1
    else:
        return 2

# easystripe, easysolid, medstripe, medsolid, hardstripe, hardsolid, winner
def get_data2(start, end):
    df = pd.DataFrame(columns = ['easystripe','easysolid','medstripe','medsolid','hardstripe','hardsolid','winner'])
    cols = [('stripe','easy'),('solid','easy'),('stripe','med'),('solid','med'),('stripe','hard'),('solid','hard')]
    for i in range(start, end):
        gamepath = folders[i]
        meta = get_meta(gamepath)
        winner = int(meta[2]==meta[3])
        csvs = glob.glob(gamepath+'/frame*.csv')
        for csv in csvs:
            imgdf = get_image_data(csv)
            imgdf['diff'] = imgdf.apply(zone, axis=1)
            imgdf = imgdf.groupby(['balltype','diff'])
            newrow = np.zeros(len(cols)+1)
            newrow[-1] = winner
            for x in range(len(cols)):
                newrow[x] = imgdf.loc(cols[x])[0]
            df.loc(len(df)) = newrow
    return df

# analytical difficulty
def difficulty(ball, cue):
    pass

# numstripes, numsolids, d2 for each stripe, d2 for each solid 
# each ball ordered by difficulty
def get_data3(start, end):
    df = pd.DataFrame(columns=['numstripe','numsolid']+['stripe'+str(i) for i in range(7)]+['solid'+str(i) for i in range(7)])
    for i in range(start, end):
        gamepath = folders[i]
        meta = get_meta(gamepath)
        winner = int(meta[2]==meta[3])
        csvs = glob.glob(gamepath+'/frame*.csv')
        for csv in csvs:
            imgdf = get_image_data(csv)
            imgdf['d2'] = imgdf.apply(closes_pocket, axis=1)
            stripedf = imgdf[imgdf['balltype']=='stripe'].sort_values(by='d2')
            soliddf = imgdf[imgdf['balltype']=='solid'].sort_values(by='d2')
            newrow = np.zeros(17)
            newrow[-1] = winner
            newrow[:2] = [len(stripedf),len(soliddf)]
            newrow[2:(2+len(stripedf))] = stripedf['d2']
            newrow[9:(9+len(soliddf))] = soliddf['d2']
            df.loc(len(df)) = newrow
    return df