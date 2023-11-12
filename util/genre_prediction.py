""" This module contains the 'feature_extraction' and 'pick_random_song'
functions for our streamlit app.
"""
import librosa
from keras import Sequential
import numpy as np
from numpy import mean, var, array
import streamlit as st
import random
import os
from sklearn.preprocessing import StandardScaler


def feature_extraction(filename: str) -> array:
    """ This function will read a song's path and extract all its audio
    features using the librosa library.

    Args:
        filename (str): The relative or absolute path of a song file.

    Returns:
        array: A nupmy array containing the extracted features.
    """
    all_features = []
    y, sr = librosa.load(filename)
    features = librosa.feature.chroma_stft(y=y, sr=sr)
    all_features.append(mean(features))
    all_features.append(var(features))
    features = librosa.feature.rms(y=y)
    all_features.append(mean(features))
    all_features.append(var(features))
    features = librosa.feature.spectral_centroid(y=y, sr=sr)
    all_features.append(mean(features))
    all_features.append(var(features))
    features = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    all_features.append(mean(features))
    all_features.append(var(features))
    features = librosa.feature.spectral_rolloff(y=y, sr=sr)
    all_features.append(mean(features))
    all_features.append(var(features))
    features = librosa.feature.zero_crossing_rate(y)
    all_features.append(mean(features))
    all_features.append(var(features))
    harmony, perceptr = librosa.effects.hpss(y=y)
    all_features.append(mean(harmony))
    all_features.append(var(harmony))
    all_features.append(mean(perceptr))
    all_features.append(var(perceptr))
    features, _ = librosa.beat.beat_track(y=y, sr=sr)
    all_features.append(features)
    # n_mfcc = number of MFCCs to return, using 20 to match
    # the features_x_sec.csv files
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for x in range(20):
        all_features.append(features[x].mean())
        all_features.append(features[x].var())
    return np.array(all_features)


def pick_random_song(model: Sequential,
                     standard_scaler: StandardScaler,
                     labels: list) -> None:
    """ This function will pick a random sample song and use the 'model' to
    predict its genre, extract the song's musical features using the
    'feature_extraction' function, transform the features with the provided
    'standard_scaler', show the real and predicted genre, and generate a
    streamlit's Audio control so we can listen to the sample song.

    Args:
        model (Sequential): Our keras Sequential genre prediction model.
        standard_scaler (StandardScaler): Our sklearn's standard scaler.
        labels (list): A list with the sample songs' genres.
    """
    filepaths = []
    for dirpath, dirnames, filenames in os.walk('data/genres_original'):
        for filename in filenames:
            if filename != '.DS_Store' and filename != 'desktop.ini':
                filepaths.append(
                    [os.path.join(
                        dirpath, filename), filename[:filename.index('.')]])

    test_index = random.randint(0, len(filepaths) - 1)
    features = feature_extraction(filepaths[test_index][0]).reshape(1, -1)
    scaled_features = standard_scaler.transform(features)
    prediction = model.predict(scaled_features, batch_size=128)
    st.write('File:', filepaths[test_index][0])
    st.write('Song\'s genre:', filepaths[test_index][1])
    st.write('Predicted genre:', labels[np.argmax(prediction)])
    st.audio(filepaths[test_index][0])
