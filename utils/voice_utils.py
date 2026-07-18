import os
import librosa
import numpy as np
import pandas as pd

def extract_voice_features(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found:\n{audio_path}")

    extension = os.path.splitext(audio_path)[1].lower()

    if extension != ".wav":
        raise ValueError("Only WAV (.wav) files are supported.")

    y, sr = librosa.load(audio_path, sr=22050)

    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    for i in range(13):
        features.append(np.mean(mfcc[i]))
        features.append(np.std(mfcc[i]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    for feature in [centroid, bandwidth, rolloff, zcr, rms]:
        features.append(np.mean(feature))
        features.append(np.std(feature))

    columns = []

    for i in range(13):
        columns.append(f"mfcc_{i+1}_mean")
        columns.append(f"mfcc_{i+1}_std")

    columns.extend([
        "centroid_mean",
        "centroid_std",
        "bandwidth_mean",
        "bandwidth_std",
        "rolloff_mean",
        "rolloff_std",
        "zcr_mean",
        "zcr_std",
        "rms_mean",
        "rms_std"
    ])

    return pd.DataFrame([features], columns=columns)