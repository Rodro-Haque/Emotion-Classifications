import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def encode_text(text):
    text = clean_text(text)

    # fallback for empty transcript
    if text == "":
        return torch.zeros(768).numpy()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        embedding = hidden.mean(dim=1).squeeze(0)

    return embedding.cpu().numpy()

def build_features(csv_paths, output_path):
    features = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        if "id" not in df.columns:
            raise ValueError(f"'id' column not found in {csv_path}")
        if "text" not in df.columns:
            raise ValueError(f"'text' column not found in {csv_path}")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            sample_id = str(row["id"])
            text = row["text"]
            features[sample_id] = encode_text(text)

    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved text features to {output_path}")

build_features(
    [
        "data/processed/train.csv",
        "data/processed/val.csv",
        "data/processed/test.csv"
    ],
    "data/processed/text_features.pkl"
)