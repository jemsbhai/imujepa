"""
Download and preprocess UCI-HAR dataset.

Source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- 30 subjects, 6 activities, waist-mounted smartphone (Samsung Galaxy S II)
- 50Hz sampling, 6 channels (3-axis acc + 3-axis gyro)
- Pre-segmented into 2.56s windows (128 samples) with 50% overlap
- We use the raw inertial signals, NOT the 561-feature vector

Activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
Labels: 1-6 (mapped to 0-5 for zero-indexed)
"""

import os
import zipfile
import urllib.request
import numpy as np
from pathlib import Path


URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATASET_NAME = "uci_har"

# Activity label mapping (1-indexed in raw data -> 0-indexed)
ACTIVITIES = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


def download(raw_dir: Path):
    """Download and extract UCI-HAR zip."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "uci_har.zip"

    if (raw_dir / "UCI HAR Dataset").exists():
        print("UCI-HAR already extracted, skipping download.")
        return

    print(f"Downloading UCI-HAR from {URL}...")
    urllib.request.urlretrieve(URL, zip_path)
    print(f"Downloaded to {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    # The zip contains a nested zip in some versions; handle both cases
    inner_zip = raw_dir / "UCI HAR Dataset.zip"
    if inner_zip.exists():
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(raw_dir)

    zip_path.unlink()
    if inner_zip.exists():
        inner_zip.unlink()
    print("Extraction complete.")


def load_inertial_signals(base_dir: Path, split: str):
    """
    Load raw inertial signals for a given split (train/test).
    Returns array of shape (N, 128, 9) -- 128 timesteps, 9 channels.
    UCI-HAR provides 9 signals: total_acc (3), body_acc (3), body_gyro (3).
    We use body_acc (3) + body_gyro (3) = 6 channels for consistency with
    other HAR datasets. total_acc includes gravity; body_acc has it removed.
    """
    signal_dir = base_dir / split / "Inertial Signals"

    # 6-channel selection: body acceleration + body gyroscope
    signal_files = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
    ]

    signals = []
    for sig_name in signal_files:
        fname = f"{sig_name}_{split}.txt"
        filepath = signal_dir / fname
        data = np.loadtxt(filepath)  # shape: (N, 128)
        signals.append(data)

    # Stack: (N, 128, 6) -- samples x timesteps x channels
    X = np.stack(signals, axis=-1).astype(np.float32)
    return X


def load_labels(base_dir: Path, split: str):
    """Load activity labels for a split. Convert to 0-indexed."""
    label_file = base_dir / split / f"y_{split}.txt"
    labels = np.loadtxt(label_file, dtype=int) - 1  # 1-indexed -> 0-indexed
    return labels


def load_subjects(base_dir: Path, split: str):
    """Load subject IDs for a split."""
    subj_file = base_dir / split / f"subject_{split}.txt"
    subjects = np.loadtxt(subj_file, dtype=int)
    return subjects


def preprocess(raw_dir: Path, processed_dir: Path):
    """
    Load raw UCI-HAR, combine train+test, and save as numpy arrays.

    Output files in processed_dir/uci_har/:
        X.npy       - (N, 128, 6) float32 -- all windows
        y.npy       - (N,) int64 -- activity labels (0-5)
        subjects.npy - (N,) int32 -- subject IDs (1-30)
        split.npy   - (N,) int8 -- 0=train, 1=test (original split)
        meta.npz    - metadata (n_classes, activities, sample_rate, channels, etc.)
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_dir = processed_dir / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "X.npy").exists():
        print(f"UCI-HAR already preprocessed at {out_dir}, skipping.")
        return

    base_dir = raw_dir / "UCI HAR Dataset"
    if not base_dir.exists():
        raise FileNotFoundError(f"Expected {base_dir}. Run download first.")

    print("Loading train split...")
    X_train = load_inertial_signals(base_dir, "train")
    y_train = load_labels(base_dir, "train")
    s_train = load_subjects(base_dir, "train")

    print("Loading test split...")
    X_test = load_inertial_signals(base_dir, "test")
    y_test = load_labels(base_dir, "test")
    s_test = load_subjects(base_dir, "test")

    # Combine
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    subjects = np.concatenate([s_train, s_test], axis=0)
    split = np.concatenate([
        np.zeros(len(X_train), dtype=np.int8),
        np.ones(len(X_test), dtype=np.int8),
    ])

    # Z-score normalize per subject
    unique_subjects = np.unique(subjects)
    for subj_id in unique_subjects:
        mask = subjects == subj_id
        subj_data = X[mask]
        mean = subj_data.mean(axis=(0, 1), keepdims=True)
        std = subj_data.std(axis=(0, 1), keepdims=True) + 1e-8
        X[mask] = (subj_data - mean) / std

    # Save
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y.astype(np.int64))
    np.save(out_dir / "subjects.npy", subjects.astype(np.int32))
    np.save(out_dir / "split.npy", split)
    np.savez(
        out_dir / "meta.npz",
        n_classes=6,
        sample_rate=50,
        window_length=128,
        channels=6,
        channel_names=["body_acc_x", "body_acc_y", "body_acc_z",
                        "body_gyro_x", "body_gyro_y", "body_gyro_z"],
        activity_names=list(ACTIVITIES.values()),
        n_subjects=len(unique_subjects),
        dataset_name=DATASET_NAME,
    )

    print(f"Saved to {out_dir}:")
    print(f"  X.npy:        {X.shape} ({X.dtype})")
    print(f"  y.npy:        {y.shape} ({y.dtype})")
    print(f"  subjects.npy: {subjects.shape}")
    print(f"  split.npy:    {split.shape} (train={int((split==0).sum())}, test={int((split==1).sum())})")
    print(f"  Activities:   {list(ACTIVITIES.values())}")
    print(f"  Subjects:     {sorted(unique_subjects.tolist())}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / DATASET_NAME
    processed_dir = project_root / "data" / "processed"

    download(raw_dir)
    preprocess(raw_dir, processed_dir)
    print("\nUCI-HAR ready.")


if __name__ == "__main__":
    main()
