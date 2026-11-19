import os
import pandas as pd

CSV_PATH = "data/labels.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    print("Columns:", list(df.columns))
    print("Number of samples:", len(df))
    print("\nFirst 5 rows:")
    print(df.head())

    missing_files = []
    for _, row in df.iterrows():
        if not os.path.exists(row["video_path"]):
            missing_files.append(row["video_path"])

    print("\nMissing video files:", len(missing_files))
    if missing_files:
        print(missing_files[:10])

    print("\nLabel counts:")
    print(df["label"].value_counts())

    duplicate_ids = df["id"].duplicated().sum()
    print("\nDuplicate IDs:", duplicate_ids)

if __name__ == "__main__":
    main()