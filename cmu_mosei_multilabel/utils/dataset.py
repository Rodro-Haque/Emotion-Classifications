import torch
from torch.utils.data import Dataset
import pandas as pd
import config


class MultimodalEmotionDataset(Dataset):
    """
    Multi-label dataset for CMU-MOSEI.

    Each row in train/val/test CSV must contain:
      id, happy, sad, anger, surprise, disgust, fear

    Each feature dictionary must use the same id keys.
    """
    def __init__(self, csv_path, text_feats, audio_feats, visual_feats):
        self.df = pd.read_csv(csv_path)
        self.text_feats = text_feats
        self.audio_feats = audio_feats
        self.visual_feats = visual_feats
        self.label_cols = config.EMOTION_NAMES

        missing_cols = [c for c in ["id"] + self.label_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])

        if sample_id not in self.text_feats:
            raise KeyError(f"Missing text feature for sample id: {sample_id}")
        if sample_id not in self.audio_feats:
            raise KeyError(f"Missing audio feature for sample id: {sample_id}")
        if sample_id not in self.visual_feats:
            raise KeyError(f"Missing visual feature for sample id: {sample_id}")

        text = torch.tensor(self.text_feats[sample_id], dtype=torch.float32)
        audio = torch.tensor(self.audio_feats[sample_id], dtype=torch.float32)
        visual = torch.tensor(self.visual_feats[sample_id], dtype=torch.float32)

        # Multi-label target vector, e.g. [1, 0, 0, 1, 0, 0]
        label_values = row[self.label_cols].values.astype("float32")
        labels = torch.tensor(label_values, dtype=torch.float32)

        return {
            "id": sample_id,
            "text": text,
            "audio": audio,
            "visual": visual,
            "labels": labels,
        }
