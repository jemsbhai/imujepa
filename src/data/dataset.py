"""
PyTorch Dataset for preprocessed HAR benchmark data.

Loads the numpy arrays saved by the download scripts and provides
windows as (C, T) tensors (channels-first, matching conv1d convention).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class HARDataset(Dataset):
    """
    Generic HAR dataset loader for preprocessed numpy arrays.

    Expects the following files in data_dir:
        X.npy        - (N, T, C) float32
        y.npy        - (N,) int64
        subjects.npy - (N,) int32
        split.npy    - (N,) int8 (0=train, 1=test)
        meta.npz     - metadata

    Args:
        data_dir: Path to processed dataset folder (e.g., data/processed/uci_har)
        split: "train", "test", "all", or None.
               If "train"/"test", uses the original split.npy.
               If "all" or None, returns everything.
        subjects: Optional list of subject IDs to include (for subject-disjoint splits).
                  Overrides the split argument if provided.
        transform: Optional callable applied to each (C, T) tensor.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Optional[str] = None,
        subjects: Optional[list[int]] = None,
        transform=None,
    ):
        data_dir = Path(data_dir)

        X = np.load(data_dir / "X.npy")              # (N, T, C)
        y = np.load(data_dir / "y.npy")               # (N,)
        subj = np.load(data_dir / "subjects.npy")      # (N,)
        split_arr = np.load(data_dir / "split.npy")    # (N,)
        meta = dict(np.load(data_dir / "meta.npz", allow_pickle=True))

        # Filter by subjects if specified
        if subjects is not None:
            mask = np.isin(subj, subjects)
            X, y, subj, split_arr = X[mask], y[mask], subj[mask], split_arr[mask]
        elif split == "train":
            mask = split_arr == 0
            X, y, subj = X[mask], y[mask], subj[mask]
        elif split == "test":
            mask = split_arr == 1
            X, y, subj = X[mask], y[mask], subj[mask]
        # else: use all data

        # Transpose to channels-first: (N, T, C) -> (N, C, T)
        self.X = torch.from_numpy(X.transpose(0, 2, 1))  # (N, C, T)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.subjects = subj
        self.meta = meta
        self.transform = transform

        self.n_channels = self.X.shape[1]
        self.seq_length = self.X.shape[2]
        self.n_classes = int(meta["n_classes"])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (C, T)
        label = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, label

    def __repr__(self):
        name = self.meta.get("dataset_name", "unknown")
        return (
            f"HARDataset({name}, n={len(self)}, "
            f"channels={self.n_channels}, seq_len={self.seq_length}, "
            f"classes={self.n_classes})"
        )


def get_har_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    subjects_train: Optional[list[int]] = None,
    subjects_test: Optional[list[int]] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for a HAR dataset.

    If subjects_train/subjects_test are provided, uses subject-disjoint split.
    Otherwise, uses the original train/test split from split.npy.
    """
    if subjects_train is not None and subjects_test is not None:
        train_ds = HARDataset(data_dir, subjects=subjects_train)
        test_ds = HARDataset(data_dir, subjects=subjects_test)
    else:
        train_ds = HARDataset(data_dir, split="train")
        test_ds = HARDataset(data_dir, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_project_root() -> Path:
    """Return project root (3 levels up from src/data/dataset.py)."""
    return Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":
    # Quick smoke test
    data_dir = get_project_root() / "data" / "processed" / "uci_har"

    print("--- Full dataset ---")
    ds = HARDataset(data_dir, split=None)
    print(ds)
    x, label = ds[0]
    print(f"Sample: x={list(x.shape)}, label={label.item()}")

    print("\n--- Train split ---")
    train_ds = HARDataset(data_dir, split="train")
    print(train_ds)

    print("\n--- Test split ---")
    test_ds = HARDataset(data_dir, split="test")
    print(test_ds)

    print("\n--- DataLoaders ---")
    train_loader, test_loader = get_har_dataloaders(data_dir, batch_size=32)
    batch_x, batch_y = next(iter(train_loader))
    print(f"Train batch: x={list(batch_x.shape)}, y={list(batch_y.shape)}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Class distribution
    print("\n--- Class distribution ---")
    for i in range(ds.n_classes):
        count = (ds.y == i).sum().item()
        name = ds.meta["activity_names"][i]
        print(f"  {i}: {name:25s} {count:5d} ({100*count/len(ds):.1f}%)")
