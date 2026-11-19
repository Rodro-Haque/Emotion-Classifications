import os
import pickle
import tempfile
import subprocess
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

TARGET_SR = 16000
N_MFCC = 40

def has_audio_stream(video_path):
    """
    Check whether the video file contains at least one audio stream.
    Returns True/False.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return "audio" in result.stdout.strip().lower()
    except Exception:
        return False

def extract_audio_to_wav(video_path, wav_path, sr=TARGET_SR):
    """
    Use ffmpeg to extract mono WAV audio from the input video.
    """
    cmd = [
        "ffmpeg",
        "-y",                 # overwrite temp file if exists
        "-i", video_path,     # input
        "-vn",                # no video
        "-ac", "1",           # mono
        "-ar", str(sr),       # sampling rate
        "-f", "wav",
        wav_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )

    return result.returncode == 0

def extract_audio_feature(video_path, sr=TARGET_SR, n_mfcc=N_MFCC):
    """
    Extract MFCC-based features from a video's audio track.
    Returns a fixed-size vector:
      40 MFCC mean + 40 MFCC std + 40 delta mean + 40 delta std = 160 dims
    """
    if not os.path.exists(video_path):
        print(f"[Missing file] {video_path}")
        return np.zeros(160, dtype=np.float32)

    if not has_audio_stream(video_path):
        print(f"[No audio stream] {video_path}")
        return np.zeros(160, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        ok = extract_audio_to_wav(video_path, tmp_wav.name, sr=sr)
        if not ok:
            print(f"[FFmpeg failed] {video_path}")
            return np.zeros(160, dtype=np.float32)

        try:
            y, sr_loaded = librosa.load(tmp_wav.name, sr=sr)
            if y is None or len(y) == 0:
                print(f"[Empty audio] {video_path}")
                return np.zeros(160, dtype=np.float32)

            mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfcc)

            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            delta_mean = delta.mean(axis=1)
            delta_std = delta.std(axis=1)

            feat = np.concatenate(
                [mfcc_mean, mfcc_std, delta_mean, delta_std],
                axis=0
            ).astype(np.float32)

            return feat

        except Exception as e:
            print(f"[Feature extraction error] {video_path} | {e}")
            return np.zeros(160, dtype=np.float32)

def build_features(csv_paths, output_path):
    features = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path}"):
            sample_id = str(row["id"])
            video_path = row["video_path"]
            features[sample_id] = extract_audio_feature(video_path)

    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved audio features to {output_path}")

if __name__ == "__main__":
    build_features(
        [
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv"
        ],
        "data/processed/audio_features.pkl"
    )