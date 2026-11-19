import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import config
from utils.dataset import MultimodalEmotionDataset
from utils.train_utils import evaluate
from utils.metrics import compute_metrics
from models.fusion_model import EarlyFusionModel

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_feats = load_pickle("data/processed/text_features.pkl")
    audio_feats = load_pickle("data/processed/audio_features.pkl")
    visual_feats = load_pickle("data/processed/visual_features.pkl")

    test_dataset = MultimodalEmotionDataset(
        "data/processed/test.csv",
        text_feats, audio_feats, visual_feats
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = EarlyFusionModel().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {metrics['macro_f1']:.4f}")
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()