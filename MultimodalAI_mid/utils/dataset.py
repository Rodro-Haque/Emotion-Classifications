import torch
from torch.utils.data import Dataset
import pandas as pd

class MultimodalEmotionDataset(Dataset):
    def __init__(self, csv_path, text_feats, audio_feats, visual_feats):
        self.df = pd.read_csv(csv_path)
        self.text_feats = text_feats
        self.audio_feats = audio_feats
        self.visual_feats = visual_feats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])

        text = torch.tensor(self.text_feats[sample_id], dtype=torch.float32)
        audio = torch.tensor(self.audio_feats[sample_id], dtype=torch.float32)
        visual = torch.tensor(self.visual_feats[sample_id], dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "text": text,
            "audio": audio,
            "visual": visual,
            "label": label
        }