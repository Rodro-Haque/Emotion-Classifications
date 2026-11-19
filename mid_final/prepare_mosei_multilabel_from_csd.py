
"""
Robust CMU-MOSEI .csd preparation for multi-label emotion classification.
This version does NOT require mmsdk and does NOT require raw videos.
It recursively searches the .csd/HDF5 file for utterance groups containing a
'features' dataset, so it works with .csd files that are nested as:
  /<sequence_name>/data/<utterance_id>/features
or
  /data/<utterance_id>/features
or similar variants.
"""

import argparse
import json
import os
import pickle
import random
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def first_existing_file(csd_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        path = os.path.join(csd_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these files were found in {csd_dir}: {candidates}")


def debug_h5_structure(path: str, max_items: int = 80) -> None:
    print(f"\nHDF5 structure preview for: {path}")
    shown = 0
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            nonlocal shown
            if shown >= max_items:
                return
            kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
            shape = getattr(obj, "shape", "")
            print(f"  {kind}: /{name} {shape}")
            shown += 1
        f.visititems(visitor)
    print("End of structure preview.\n")


def collect_feature_groups(path: str) -> List[Tuple[str, str]]:
    """
    Return list of (sample_id, hdf5_group_path) for all groups that contain
    a dataset named 'features'.
    """
    pairs = []
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Group) and "features" in obj:
                # For /sequence/data/utterance_id, the sample id is the final path part.
                sample_id = name.split("/")[-1]
                pairs.append((str(sample_id), name))
        f.visititems(visitor)
    return pairs


def read_group_features(path: str, group_path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.array(f[group_path]["features"])


def pool_sequence_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim == 0:
        return np.array([float(x)], dtype=np.float32)
    if x.ndim == 1:
        return x.astype(np.float32)
    if x.ndim >= 2:
        # Flatten all time-like dimensions, keep the final feature dimension.
        x = x.reshape(-1, x.shape[-1])
        if x.shape[0] == 0:
            return np.zeros((x.shape[-1],), dtype=np.float32)
        return x.mean(axis=0).astype(np.float32)

    raise ValueError(f"Unsupported feature shape: {x.shape}")


def load_pooled_csd_features(path: str, name: str) -> Dict[str, np.ndarray]:
    print(f"Loading {name}: {path}")
    groups = collect_feature_groups(path)
    print(f"Found {len(groups)} groups containing 'features' in {name}")

    if len(groups) == 0:
        debug_h5_structure(path)
        raise RuntimeError(f"No groups with a 'features' dataset were found in {path}")

    output = {}
    for sample_id, group_path in tqdm(groups, desc=f"Reading {name}"):
        try:
            feat = read_group_features(path, group_path)
            pooled = pool_sequence_features(feat)
            if pooled.size > 0:
                output[sample_id] = pooled
        except Exception as e:
            print(f"[Skipped] {sample_id} at {group_path}: {e}")

    print(f"Loaded {len(output)} usable samples from {name}")
    return output


def load_multilabels(path: str) -> pd.DataFrame:
    """
    CMU-MOSEI emotion label order is commonly:
      sentiment, happy, sad, anger, surprise, disgust, fear
    Some files may contain only:
      happy, sad, anger, surprise, disgust, fear
    This converts emotion intensities to binary labels using > 0.
    """
    print(f"Loading labels: {path}")
    groups = collect_feature_groups(path)
    print(f"Found {len(groups)} groups containing 'features' in labels")

    if len(groups) == 0:
        debug_h5_structure(path)
        raise RuntimeError(f"No label feature groups found in {path}")

    rows = []
    skipped_shapes = {}

    for sample_id, group_path in tqdm(groups, desc="Reading labels"):
        try:
            feat = read_group_features(path, group_path)
            feat = np.asarray(feat, dtype=np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            feat = feat.reshape(-1, feat.shape[-1]).mean(axis=0)
        except Exception as e:
            print(f"[Skipped label] {sample_id} at {group_path}: {e}")
            continue

        if feat.shape[0] >= 7:
            emotion_values = feat[1:7]  # remove sentiment column
        elif feat.shape[0] == 6:
            emotion_values = feat[:6]
        else:
            skipped_shapes[str(tuple(feat.shape))] = skipped_shapes.get(str(tuple(feat.shape)), 0) + 1
            continue

        binary = (emotion_values > 0).astype(np.int64)
        row = {"id": str(sample_id)}
        for emotion_name, value in zip(EMOTION_NAMES, binary):
            row[emotion_name] = int(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} label rows")

    if df.empty:
        print("No valid rows were created. Skipped label shapes:", skipped_shapes)
        debug_h5_structure(path)
        raise RuntimeError(
            "Label rows are empty. Your label .csd may not contain the 6 emotion labels, "
            "or its feature column order is different. Check the structure preview above."
        )

    print("Positive label counts:")
    print(df[EMOTION_NAMES].sum().sort_values(ascending=False))
    return df


def normalize_feature_dict(feat_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keys = list(feat_dict.keys())
    X = np.stack([feat_dict[k] for k in keys]).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return {k: ((feat_dict[k].astype(np.float32) - mean) / std).astype(np.float32) for k in keys}


def filter_by_ids(feat_dict: Dict[str, np.ndarray], valid_ids) -> Dict[str, np.ndarray]:
    valid_ids = set(valid_ids)
    return {k: v for k, v in feat_dict.items() if k in valid_ids}


def split_dataframe(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    ids = list(df["id"].values)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csd_dir", default="data/csd")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect_only", action="store_true", help="Only print .csd structures, then exit.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    label_path = first_existing_file(args.csd_dir, [
        "CMU_MOSEI_Labels.csd", "CMU_MOSEI_Emotion_Labels.csd", "CMU_MOSEI_Labels.csd.csd"
    ])
    text_path = first_existing_file(args.csd_dir, [
        "CMU_MOSEI_TimestampedWordVectors.csd", "CMU_MOSEI_TimestampedWordVectors.csd.csd"
    ])
    audio_path = first_existing_file(args.csd_dir, [
        "CMU_MOSEI_COVAREP.csd", "CMU_MOSEI_COVAREP.csd.csd"
    ])
    visual_path = first_existing_file(args.csd_dir, [
        "CMU_MOSEI_VisualOpenFace2.csd", "CMU_MOSEI_OpenFace_2.csd", "CMU_MOSEI_OpenFace.csd"
    ])

    if args.inspect_only:
        for p in [label_path, text_path, audio_path, visual_path]:
            debug_h5_structure(p)
            print("feature groups:", len(collect_feature_groups(p)))
        return

    labels_df = load_multilabels(label_path)
    text_features = load_pooled_csd_features(text_path, "text word vectors")
    audio_features = load_pooled_csd_features(audio_path, "audio COVAREP")
    visual_features = load_pooled_csd_features(visual_path, "visual OpenFace")

    common_ids = set(labels_df["id"]) & set(text_features) & set(audio_features) & set(visual_features)
    print(f"Common samples across labels/text/audio/visual: {len(common_ids)}")

    if len(common_ids) == 0:
        print("Example label IDs:", list(labels_df["id"].head(5)))
        print("Example text IDs:", list(text_features.keys())[:5])
        print("Example audio IDs:", list(audio_features.keys())[:5])
        print("Example visual IDs:", list(visual_features.keys())[:5])
        raise RuntimeError("No common IDs found. Check whether all .csd files come from the same CMU-MOSEI release.")

    labels_df = labels_df[labels_df["id"].isin(common_ids)].reset_index(drop=True)
    text_features = normalize_feature_dict(filter_by_ids(text_features, common_ids))
    audio_features = normalize_feature_dict(filter_by_ids(audio_features, common_ids))
    visual_features = normalize_feature_dict(filter_by_ids(visual_features, common_ids))

    train_df, val_df, test_df = split_dataframe(labels_df, seed=args.seed)

    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    save_pickle(text_features, os.path.join(args.out_dir, "text_features.pkl"))
    save_pickle(audio_features, os.path.join(args.out_dir, "audio_features.pkl"))
    save_pickle(visual_features, os.path.join(args.out_dir, "visual_features.pkl"))

    first_id = next(iter(common_ids))
    dims = {
        "NUM_CLASSES": len(EMOTION_NAMES),
        "EMOTION_NAMES": EMOTION_NAMES,
        "TEXT_DIM": int(text_features[first_id].shape[0]),
        "AUDIO_DIM": int(audio_features[first_id].shape[0]),
        "VISUAL_DIM": int(visual_features[first_id].shape[0]),
        "num_samples": int(len(labels_df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
    }

    with open(os.path.join(args.out_dir, "feature_dims.json"), "w", encoding="utf-8") as f:
        json.dump(dims, f, indent=2)

    print("\nSaved processed files to:", args.out_dir)
    print(json.dumps(dims, indent=2))
    print("\nNext: copy TEXT_DIM, AUDIO_DIM, VISUAL_DIM, NUM_CLASSES into config.py, then run train.py")


if __name__ == "__main__":
    main()
