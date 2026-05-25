"""
Fix processing for PAMAP2 and USC-HAD (data already on disk, just parser issues).

Usage:
    python -m scripts.fix_datasets
"""

import numpy as np
from scipy.signal import resample
from scipy.io import loadmat
from pathlib import Path


def resample_signal(data, orig_hz, target_hz):
    if orig_hz == target_hz:
        return data
    n_target = int(len(data) * target_hz / orig_hz)
    return resample(data, n_target, axis=0).astype(np.float32)


def segment_windows(data, labels, window_len=128):
    windows, win_labels = [], []
    for start in range(0, len(data) - window_len + 1, window_len):
        w = data[start:start + window_len]
        if np.isnan(w).any():
            continue
        lbl = labels[start:start + window_len]
        vals, counts = np.unique(lbl[~np.isnan(lbl)].astype(int), return_counts=True)
        if len(vals) == 0:
            continue
        win_labels.append(vals[np.argmax(counts)])
        windows.append(w)
    if not windows:
        return np.zeros((0, window_len, data.shape[1])), np.zeros(0, dtype=np.int64)
    return np.stack(windows).astype(np.float32), np.array(win_labels, dtype=np.int64)


def normalize_per_subject(X, subjects):
    X = X.copy()
    for sid in np.unique(subjects):
        mask = subjects == sid
        d = X[mask]
        X[mask] = (d - d.mean(axis=(0,1), keepdims=True)) / (d.std(axis=(0,1), keepdims=True) + 1e-8)
    return X


def save_dataset(out_dir, X, y, subjects, meta_dict, train_ratio=0.8):
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_subj = np.unique(subjects)
    rng = np.random.RandomState(42)
    perm = rng.permutation(unique_subj)
    n_train = max(1, int(len(perm) * train_ratio))
    train_set = set(perm[:n_train].tolist())
    split = np.array([0 if s in train_set else 1 for s in subjects], dtype=np.int8)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subjects.npy", subjects.astype(np.int32))
    np.save(out_dir / "split.npy", split)
    np.savez(out_dir / "meta.npz", **meta_dict)
    print(f"  Saved: X={X.shape}, y={y.shape}, "
          f"subjects={len(unique_subj)}, classes={int(y.max()+1)}, "
          f"train={(split==0).sum()}, test={(split==1).sum()}")


def process_pamap2(raw_dir, processed_dir):
    """PAMAP2: 9 subjects, 100Hz IMU, hand sensor acc+gyro."""
    name = "pamap2"
    out_dir = processed_dir / name
    proto_dir = raw_dir / name / "PAMAP2_Dataset" / "Protocol"

    print(f"[{name}] Processing from {proto_dir}...")

    # PAMAP2 column layout (0-indexed):
    #  0: timestamp
    #  1: activityID
    #  2: heart rate
    #  3: (temp hand)
    #  4-6: hand acc 16g (x,y,z)
    #  7-9: hand acc 6g (x,y,z)
    # 10-12: hand gyro (x,y,z)
    # 13-15: hand mag (x,y,z)
    # 16-19: hand orientation (quaternion)
    # 20: (temp chest) ... etc.

    all_X, all_y, all_subj = [], [], []

    for subj_id in range(1, 10):
        filepath = proto_dir / f"subject10{subj_id}.dat"
        if not filepath.exists():
            print(f"  Skipping subject {subj_id} (file not found)")
            continue

        # Read with space delimiter, handle NaN
        data = np.genfromtxt(filepath, dtype=np.float64)
        print(f"  Subject {subj_id}: {data.shape[0]} samples, {data.shape[1]} columns")

        labels = data[:, 1].astype(float)
        # Hand IMU: acc_16g (cols 4-6) + gyro (cols 10-12)
        imu = data[:, [4, 5, 6, 10, 11, 12]].astype(np.float32)

        # Remove transient/null activity (label 0) and NaN rows
        valid = (labels > 0) & ~np.isnan(labels) & ~np.isnan(imu).any(axis=1)
        imu = imu[valid]
        labels = labels[valid].astype(int)

        if len(imu) == 0:
            continue

        # Resample 100Hz -> 50Hz
        imu_50 = resample_signal(imu, 100, 50)
        labels_50 = resample_signal(
            labels.astype(float).reshape(-1, 1), 100, 50
        ).round().astype(int).flatten()

        X_subj, y_subj = segment_windows(imu_50, labels_50, window_len=128)
        if len(X_subj) == 0:
            continue

        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subj.append(np.full(len(X_subj), subj_id, dtype=np.int32))

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subj)

    # Remap labels to contiguous 0-indexed
    unique_labels = sorted(np.unique(y).tolist())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y], dtype=np.int64)

    X = normalize_per_subject(X, subjects)
    save_dataset(out_dir, X, y, subjects, {
        "n_classes": len(unique_labels),
        "sample_rate": 50, "window_length": 128, "channels": 6,
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "dataset_name": name, "n_subjects": len(np.unique(subjects)),
        "original_label_map": str(label_map),
    })


def process_usc_had(raw_dir, processed_dir):
    """USC-HAD: 14 subjects, 12 activities, 100Hz, hip-mounted.
    Files: data/raw/usc_had/USC-HAD/Subject{K}/a{N}t{M}.mat
    Each .mat has 'sensor_readings' with 6 columns: acc(3) + gyro(3).
    """
    name = "usc_had"
    out_dir = processed_dir / name
    data_root = raw_dir / name / "USC-HAD"

    print(f"[{name}] Processing from {data_root}...")

    all_X, all_y, all_subj = [], [], []

    for subj_dir in sorted(data_root.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("Subject"):
            continue

        subj_id = int(subj_dir.name.replace("Subject", ""))

        for mat_file in sorted(subj_dir.glob("a*t*.mat")):
            # Parse activity from filename: a{N}t{M}.mat
            fname = mat_file.stem  # e.g., "a1t1"
            try:
                parts = fname.split("t")
                act_id = int(parts[0][1:]) - 1  # a1 -> 0, a12 -> 11
            except (ValueError, IndexError):
                continue

            try:
                mat = loadmat(str(mat_file))
            except Exception as e:
                print(f"  Warning: failed to load {mat_file}: {e}")
                continue

            # sensor_readings: (T, 6) — acc_x,y,z, gyro_x,y,z
            if "sensor_readings" in mat:
                imu = mat["sensor_readings"].astype(np.float32)
            else:
                # Try alternate key names
                keys = [k for k in mat.keys() if not k.startswith("_")]
                if not keys:
                    continue
                imu = mat[keys[0]].astype(np.float32)

            if imu.shape[1] < 6:
                continue

            imu = imu[:, :6]  # first 6 columns: acc + gyro

            # Resample 100Hz -> 50Hz
            imu = resample_signal(imu, 100, 50)
            labels = np.full(len(imu), act_id, dtype=np.int64)

            X_seg, y_seg = segment_windows(imu, labels, window_len=128)
            if len(X_seg) > 0:
                all_X.append(X_seg)
                all_y.append(y_seg)
                all_subj.append(np.full(len(X_seg), subj_id, dtype=np.int32))

    if not all_X:
        print(f"[{name}] ERROR: No data loaded.")
        return

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subj)

    # Remap labels to contiguous
    unique_labels = sorted(np.unique(y).tolist())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y], dtype=np.int64)

    X = normalize_per_subject(X, subjects)
    save_dataset(out_dir, X, y, subjects, {
        "n_classes": len(unique_labels),
        "sample_rate": 50, "window_length": 128, "channels": 6,
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "dataset_name": name, "n_subjects": len(np.unique(subjects)),
    })


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("Fixing PAMAP2 and USC-HAD Processing")
    print("=" * 60)
    print()

    process_pamap2(raw_dir, processed_dir)
    print()
    process_usc_had(raw_dir, processed_dir)

    # Summary of all datasets
    print("\n" + "=" * 60)
    print("All Datasets Status")
    print("=" * 60)
    for ds in ["uci_har", "pamap2", "dsads", "mhealth", "usc_had", "realworld"]:
        ds_dir = processed_dir / ds
        if (ds_dir / "X.npy").exists():
            X = np.load(ds_dir / "X.npy")
            y = np.load(ds_dir / "y.npy")
            subj = np.load(ds_dir / "subjects.npy")
            print(f"  {ds:12s}: {X.shape[0]:6d} windows, "
                  f"{len(np.unique(y)):2d} classes, "
                  f"{len(np.unique(subj)):2d} subjects, "
                  f"shape={X.shape[1:]}")
        else:
            print(f"  {ds:12s}: NOT AVAILABLE")


if __name__ == "__main__":
    main()
