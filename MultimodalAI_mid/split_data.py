import pandas as pd
from sklearn.model_selection import train_test_split

LABEL2ID = {
    "neutral": 0,
    "happy": 1,
    "anger": 2,
    "disgust": 3,
    "fear": 4,
    "sad": 5,
    "surprise": 6
}

df = pd.read_csv("data/labels.csv")
df["label"] = df["label"].map(LABEL2ID)

if df["label"].isna().sum() > 0:
    raise ValueError("Some labels were not found in LABEL2ID. Check folder names.")

train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Saved:")
print("train:", len(train_df))
print("val:", len(val_df))
print("test:", len(test_df))