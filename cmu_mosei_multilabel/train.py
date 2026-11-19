import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils.dataset import MultimodalEmotionDataset
from utils.train_utils import train_one_epoch, evaluate
from utils.metrics import compute_multilabel_metrics
from models.fusion_model import EarlyFusionModel


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_pos_weight(train_df, label_cols, device):
    """
    pos_weight helps BCEWithLogitsLoss learn minority emotions.
    Formula: negative_count / positive_count per emotion.
    """
    y = train_df[label_cols].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = neg / (pos + 1e-6)
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Emotion labels:", config.EMOTION_NAMES)

    text_feats = load_pickle("data/processed/text_features.pkl")
    audio_feats = load_pickle("data/processed/audio_features.pkl")
    visual_feats = load_pickle("data/processed/visual_features.pkl")

    train_dataset = MultimodalEmotionDataset(
        "data/processed/train.csv",
        text_feats, audio_feats, visual_feats
    )
    val_dataset = MultimodalEmotionDataset(
        "data/processed/val.csv",
        text_feats, audio_feats, visual_feats
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = EarlyFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    pos_weight = compute_pos_weight(train_dataset.df, config.EMOTION_NAMES, device)
    print("Positive class weights:", pos_weight.detach().cpu().numpy())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_macro_f1 = 0.0

    for epoch in range(config.EPOCHS):
        train_loss, train_true, train_logits = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_metrics, _, _ = compute_multilabel_metrics(
            train_true, train_logits, threshold=config.THRESHOLD
        )

        val_loss, val_true, val_logits = evaluate(
            model, val_loader, criterion, device
        )
        val_metrics, _, _ = compute_multilabel_metrics(
            val_true, val_logits, threshold=config.THRESHOLD
        )

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Micro F1: {train_metrics['micro_f1']:.4f} | Train Macro F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Micro F1: {val_metrics['micro_f1']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model.")


if __name__ == "__main__":
    main()
