import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import config
from utils.dataset import MultimodalEmotionDataset
from utils.train_utils import evaluate
from utils.metrics import compute_multilabel_metrics
from models.fusion_model import EarlyFusionModel


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Emotion labels:", config.EMOTION_NAMES)

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

    criterion = nn.BCEWithLogitsLoss()
    test_loss, y_true, y_logits = evaluate(model, test_loader, criterion, device)
    metrics, y_prob, y_pred = compute_multilabel_metrics(
        y_true, y_logits, threshold=config.THRESHOLD
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Threshold: {config.THRESHOLD}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=config.EMOTION_NAMES,
            zero_division=0
        )
    )

    print("\nMultilabel Confusion Matrix per class:")
    print("Each matrix is [[TN, FP], [FN, TP]]")
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    for name, mat in zip(config.EMOTION_NAMES, matrices):
        print(f"\n{name}:")
        print(mat)


if __name__ == "__main__":
    main()
