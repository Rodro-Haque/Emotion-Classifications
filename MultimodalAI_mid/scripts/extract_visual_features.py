import pickle
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def sample_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

def extract_visual_feature(video_path):
    frames = sample_frames(video_path, num_frames=8)
    if len(frames) == 0:
        return np.zeros(512, dtype=np.float32)

    features = []
    with torch.no_grad():
        for frame in frames:
            img = Image.fromarray(frame)
            x = transform(img).unsqueeze(0).to(DEVICE)
            feat = resnet(x).squeeze(0).cpu().numpy()
            features.append(feat)

    features = np.stack(features, axis=0)
    return features.mean(axis=0).astype(np.float32)

def build_features(csv_paths, output_path):
    features = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sample_id = str(row["id"])
            video_path = row["video_path"]
            features[sample_id] = extract_visual_feature(video_path)

    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved visual features to {output_path}")

build_features(
    [
        "data/processed/train.csv",
        "data/processed/val.csv",
        "data/processed/test.csv"
    ],
    "data/processed/visual_features.pkl"
)