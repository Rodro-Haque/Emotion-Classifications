import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_multilabel_metrics(y_true, y_logits, threshold=0.5):
    y_prob = sigmoid(y_logits)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        # Exact-match accuracy: all labels must be correct for a sample.
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }, y_prob, y_pred
