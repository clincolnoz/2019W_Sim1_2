#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Creates an audio feature set for later use in finding similar
audio files within a ethonmusicology context.

Specifically, audio is converted to mono with sampling rate of 22050 using librosa.

All features as generated for a 2 second time window.

Each window is labelled as either solo singing, choir singing, instrumental or speech
using an existing tensorlfow model from https://github.com/matijama/field-recording-segmentation
that returns probabilities for each class.

Further mfcc_delta and chroma_stft are applied and generate features for each window resulting in 13 and 12 features, respectively.

Finally, all features are converted to a dataframe, which is filtered to remove sections where speech has the maximum probability
"""

__author__ = "Craig Lincoln"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Lincoln"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Craig Lincoln"
__email__ = "clincolnoz@gmail.com"
__status__ = "Testing"

#%% init
import librosa, librosa.display
import numpy as np
import pandas as pd

# import tensorflow as tf
import matplotlib.pyplot as plt
import os
from os.path import relpath, dirname, abspath

root_dir = os.getcwd()
root_dir

#%% load audio track
def load_tracks(filename, sr=22050, mono=True, duration=None, offset=0):
    y, sr = librosa.load(filename, sr=22050, mono=True, duration=duration)
    y = y / max(abs(y)) * 0.9

    return y, sr


#%% get mfcc
def get_mfcc(y, sr):
    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = sr

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=20)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # mfcc_delta = mfcc_delta[:,:output_rows].T

    return mfcc_delta


#%% Get Chroma
def get_chroma_stft(y, sr):
    c = librosa.feature.chroma_stft(y, sr, hop_length=sr)

    # c = c[:,:output_rows].T

    return c


def calc_features(file_path):
    y, sr = load_tracks(file_path, sr=22050, mono=True, duration=None)
    mfcc_delta = get_mfcc(y, sr)
    c = get_chroma_stft(y, sr)

    # Create a DataFrame
<<<<<<< HEAD
    df=pd.DataFrame(np.concatenate((mfcc_delta,c),axis=0)).T
    df.reset_index(inplace=True)
    df['index'] = df['index']+1
    df['file'] = os.path.split(file_path)[1].split('.avi')[0]
=======
    df = pd.DataFrame(np.concatenate((mfcc_delta, c), axis=0)).T
>>>>>>> 4e08d79613b6d8e1ceae61522411449f5849cdf3

    # # add filename and chunk number as key
    # df['filename']=filename

    # df['seq'] = 1
    # df['seq'] = df['seq'].cumsum()

    # df['key']=df['filename'] + ';' + df['seq'].astype(str)

    # df.drop(['filename','seq'],axis=1,inplace=True)

    # df.set_index('key',inplace=True)

    return df


#%% Plot MFCCS
def plot_mfcc_c(mfcc_delta, c, outfilename):

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mfcc_delta, x_axis="time", y_axis="mel", sr=22050, hop_length=sr
    )
    plt.colorbar()
    plt.title("MFCC Delta")
    plt.tight_layout()
    plt.savefig(outfilename + "_MFCC")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(c, x_axis="time", y_axis="chroma", sr=22050, hop_length=sr)
    plt.colorbar()
    plt.title("Chroma")
    plt.tight_layout()
    plt.savefig(outfilename + "_Chroma")


# %%
# filename = os.path.join(root_dir,'data/external/Muppets-02-01-01.avi')
# outfilename = os.path.join(root_dir,'reports/Muppets-02-01-01')
# y, sr = load_tracks(filename, sr=22050, mono=True, duration=None)
# print(librosa.get_duration(y=y, sr=sr)/60)
# mfcc_delta=get_mfcc(y, sr)
# c=get_chroma_stft(y,sr)

# plot_mfcc_c(mfcc_delta,c,outfilename)

# %%
# files = ['data/external/Muppets-02-01-01.avi','data/external/Muppets-02-04-04.avi','data/external/Muppets-03-04-03.avi']
def save_audio_featureto_csv(files):
    dfs = []
    for file in files:
        file_path = os.path.join(root_dir,file)
        dfs.append(calc_features(file_path))
    df = pd.concat(dfs)
    df.to_csv(os.path.join(root_dir,'data/audio_features/audio_features.csv'),index=False)
    
