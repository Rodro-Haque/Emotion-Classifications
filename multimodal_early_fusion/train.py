import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils.dataset import MultimodalEmotionDataset
from utils.train_utils import train_one_epoch, evaluate
from utils.metrics import compute_metrics
from models.fusion_model import EarlyFusionModel

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

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
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(config.EPOCHS):
        train_loss, train_true, train_pred = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_metrics = compute_metrics(train_true, train_pred)

        val_loss, val_true, val_pred = evaluate(
            model, val_loader, criterion, device
        )
        val_metrics = compute_metrics(val_true, val_pred)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model.")

if __name__ == "__main__":
    main()