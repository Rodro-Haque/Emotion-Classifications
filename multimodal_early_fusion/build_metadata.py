import os
import pandas as pd

label_map = {
    "angry": "anger",
    "surprised": "surprise",
    "fearful": "fear",
    "disgusted": "disgust",
    "happiness": "happy",
    "sadness": "sad"
}

ROOT_DIR = "data/"
OUTPUT_FILE = "data/labels.csv"

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def main():
    rows = []

    for label in sorted(os.listdir(ROOT_DIR)):
        label_dir = os.path.join(ROOT_DIR, label)

        if not os.path.isdir(label_dir):
            continue

        clean_label = label.lower().strip()
        clean_label = label_map.get(clean_label, clean_label)

        for filename in sorted(os.listdir(label_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            base_name = os.path.splitext(filename)[0]
            sample_id = f"{clean_label}_{base_name}"
            video_path = os.path.join(label_dir, filename)

            rows.append({
                "id": sample_id,
                "video_path": video_path,
                "label": clean_label,
                "text": ""
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved metadata to {OUTPUT_FILE}")
    print(df.head())
    print("\nLabel counts:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()