import numpy as np
import pandas as pd
import glob
import csv

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

# type, x, y, frame, winner (1 if stripes wins)
def get_game_data(gamepath):
    csvs = glob.glob(gamepath+'/frame*.csv')
    df = pd.concat([get_image_data(csvpath) for csvpath in csvs], ignore_index=True) # untested
    metacsv = glob.glob(gamepath+'*_meta.csv')[0]
    with open(metacsv) as metafile:
        reader = csv.reader(metafile)
        _ = next(reader)
        meta = next(reader)
    winner = int(meta[2]==meta[3])
    df['winner'] = winner

# untested
folders = glob.glob('../../cropped_images_ff/*')

def append_game(i):
    df = get_game_data(folders[i])
    df['game'] = i
    return df

# type, x, y, frame, winner, game
def get_data(num_games):
    return pd.concat([append_game(i) for i in range(num_games)], ignore_index=True) # untested