"""
Download and preprocess remaining HAR benchmark datasets.
Standardizes all to: 50Hz, 128-sample windows, 6-channel (acc+gyro), z-score per subject.

Datasets: PAMAP2, DSADS, MHealth, USC-HAD, Realworld

Usage:
    python -m scripts.download_har_benchmarks
"""

import os
import zipfile
import urllib.request
import numpy as np
from scipy.signal import resample
from pathlib import Path
from io import BytesIO


def resample_signal(data: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    """Resample (T, C) signal from orig_hz to target_hz."""
    if orig_hz == target_hz:
        return data
    n_target = int(len(data) * target_hz / orig_hz)
    return resample(data, n_target, axis=0).astype(np.float32)


def segment_windows(data: np.ndarray, labels: np.ndarray, window_len: int, stride: int = None):
    """Segment (T, C) data into (N, window_len, C) windows with majority-vote labels."""
    if stride is None:
        stride = window_len
    T = len(data)
    windows, win_labels = [], []
    for start in range(0, T - window_len + 1, stride):
        end = start + window_len
        w = data[start:end]
        # Skip windows with NaN
        if np.isnan(w).any():
            continue
        lbl_segment = labels[start:end]
        # Majority vote
        vals, counts = np.unique(lbl_segment[~np.isnan(lbl_segment)].astype(int), return_counts=True)
        if len(vals) == 0:
            continue
        win_labels.append(vals[np.argmax(counts)])
        windows.append(w)
    if len(windows) == 0:
        return np.zeros((0, window_len, data.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.stack(windows).astype(np.float32), np.array(win_labels, dtype=np.int64)


def normalize_per_subject(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Z-score normalize per subject."""
    X = X.copy()
    for subj_id in np.unique(subjects):
        mask = subjects == subj_id
        subj_data = X[mask]
        mean = subj_data.mean(axis=(0, 1), keepdims=True)
        std = subj_data.std(axis=(0, 1), keepdims=True) + 1e-8
        X[mask] = (subj_data - mean) / std
    return X


def save_dataset(out_dir, X, y, subjects, meta_dict, train_ratio=0.8):
    """Save preprocessed dataset in standard format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create train/test split by subjects
    unique_subj = np.unique(subjects)
    np.random.seed(42)
    np.random.shuffle(unique_subj)
    n_train = max(1, int(len(unique_subj) * train_ratio))
    train_subj = set(unique_subj[:n_train].tolist())

    split = np.array([0 if s in train_subj else 1 for s in subjects], dtype=np.int8)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subjects.npy", subjects.astype(np.int32))
    np.save(out_dir / "split.npy", split)
    np.savez(out_dir / "meta.npz", **meta_dict)

    n_train_samples = (split == 0).sum()
    n_test_samples = (split == 1).sum()
    print(f"  Saved: X={X.shape}, y={y.shape}, "
          f"subjects={len(unique_subj)}, "
          f"train={n_train_samples}, test={n_test_samples}")


# ============================================================
# PAMAP2
# ============================================================
def download_pamap2(raw_dir: Path, processed_dir: Path):
    """PAMAP2: 9 subjects, 18 activities, 100Hz, wrist+chest+ankle IMU."""
    name = "pamap2"
    out_dir = processed_dir / name
    if (out_dir / "X.npy").exists():
        print(f"[{name}] Already preprocessed, skipping.")
        return

    url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    dl_dir = raw_dir / name
    dl_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dl_dir / "pamap2.zip"
    if not (dl_dir / "PAMAP2_Dataset").exists():
        print(f"[{name}] Downloading...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dl_dir)
        zip_path.unlink(missing_ok=True)

    # Find the Protocol directory
    proto_dir = None
    for candidate in [
        dl_dir / "PAMAP2_Dataset" / "Protocol",
        dl_dir / "Protocol",
    ]:
        if candidate.exists():
            proto_dir = candidate
            break

    if proto_dir is None:
        # Search recursively
        for p in dl_dir.rglob("subject10*.dat"):
            proto_dir = p.parent
            break

    if proto_dir is None:
        print(f"[{name}] ERROR: Could not find Protocol directory. Check {dl_dir}")
        return

    print(f"[{name}] Processing from {proto_dir}...")

    # PAMAP2 columns: timestamp, activityID, heartrate, then 3x IMU (hand, chest, ankle)
    # Each IMU block: 17 columns (temp, 3acc_16g, 3acc_6g, 3gyro, 3mag, 4orient)
    # We use hand IMU (columns 4-20): acc_16g (cols 4-6) + gyro (cols 10-12)
    # Column indices (0-based): hand_acc=4,5,6  hand_gyro=10,11,12

    all_X, all_y, all_subj = [], [], []
    activity_set = set()

    for subj_id in range(1, 10):  # subjects 1-9
        filepath = proto_dir / f"subject10{subj_id}.dat"
        if not filepath.exists():
            continue

        data = np.loadtxt(filepath)
        labels = data[:, 1].astype(int)

        # 6-channel: hand acc (16g) + hand gyro
        imu_data = np.column_stack([
            data[:, 4:7],    # hand acc x,y,z
            data[:, 10:13],  # hand gyro x,y,z
        ])

        # Remove transient activity (id=0)
        valid = labels > 0
        imu_data = imu_data[valid]
        labels = labels[valid]

        activity_set.update(labels.tolist())

        # Resample 100Hz -> 50Hz
        imu_data = resample_signal(imu_data, 100, 50)
        labels_resampled = resample_signal(
            labels.astype(float).reshape(-1, 1), 100, 50
        ).round().astype(int).flatten()

        # Segment
        X_subj, y_subj = segment_windows(imu_data, labels_resampled, window_len=128)
        if len(X_subj) == 0:
            continue

        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subj.append(np.full(len(X_subj), subj_id, dtype=np.int32))

    if not all_X:
        print(f"[{name}] ERROR: No data loaded.")
        return

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
        "sample_rate": 50,
        "window_length": 128,
        "channels": 6,
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "dataset_name": name,
        "original_label_map": str(label_map),
        "n_subjects": len(np.unique(subjects)),
    })
    print(f"[{name}] Done. Classes: {len(unique_labels)}")


# ============================================================
# DSADS
# ============================================================
def download_dsads(raw_dir: Path, processed_dir: Path):
    """DSADS: 8 subjects, 19 activities, 25Hz, 5 body positions."""
    name = "dsads"
    out_dir = processed_dir / name
    if (out_dir / "X.npy").exists():
        print(f"[{name}] Already preprocessed, skipping.")
        return

    url = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"
    dl_dir = raw_dir / name
    dl_dir.mkdir(parents=True, exist_ok=True)

    if not list(dl_dir.rglob("*.txt")):
        print(f"[{name}] Downloading...")
        zip_path = dl_dir / "dsads.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dl_dir)
        zip_path.unlink(missing_ok=True)

    print(f"[{name}] Processing...")

    # DSADS structure: activities a01-a19, persons p1-p8, segments s01-s60
    # Each segment file: 125 rows (5s at 25Hz), 45 columns
    # 5 sensor units x 9 channels (acc, gyro, mag)
    # Torso unit = columns 0-8: acc(0-2), gyro(3-5), mag(6-8)
    # We use torso: acc(0-2) + gyro(3-5) = 6 channels

    all_X, all_y, all_subj = [], [], []

    # Find the data directory
    data_root = None
    for candidate in dl_dir.rglob("a01"):
        data_root = candidate.parent
        break

    if data_root is None:
        print(f"[{name}] ERROR: Could not find activity directories.")
        return

    for act_id in range(1, 20):  # 19 activities
        act_dir = data_root / f"a{act_id:02d}"
        if not act_dir.exists():
            continue

        for person_id in range(1, 9):  # 8 subjects
            person_dir = act_dir / f"p{person_id}"
            if not person_dir.exists():
                continue

            for seg_file in sorted(person_dir.glob("s*.txt")):
                data = np.loadtxt(seg_file, delimiter=",")
                # Torso acc + gyro (first 6 of 45 columns)
                imu = data[:, :6].astype(np.float32)

                # Resample 25Hz -> 50Hz
                imu = resample_signal(imu, 25, 50)

                # Each segment is 5s -> after resampling: 250 samples
                # Segment into 128-sample windows
                for start in range(0, len(imu) - 128 + 1, 128):
                    w = imu[start:start + 128]
                    if np.isnan(w).any():
                        continue
                    all_X.append(w)
                    all_y.append(act_id - 1)  # 0-indexed
                    all_subj.append(person_id)

    X = np.stack(all_X).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    subjects = np.array(all_subj, dtype=np.int32)

    X = normalize_per_subject(X, subjects)

    save_dataset(out_dir, X, y, subjects, {
        "n_classes": 19,
        "sample_rate": 50,
        "window_length": 128,
        "channels": 6,
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "dataset_name": name,
        "n_subjects": len(np.unique(subjects)),
    })
    print(f"[{name}] Done.")


# ============================================================
# MHealth
# ============================================================
def download_mhealth(raw_dir: Path, processed_dir: Path):
    """MHealth: 10 subjects, 12 activities, 50Hz, chest+wrist+ankle."""
    name = "mhealth"
    out_dir = processed_dir / name
    if (out_dir / "X.npy").exists():
        print(f"[{name}] Already preprocessed, skipping.")
        return

    url = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
    dl_dir = raw_dir / name
    dl_dir.mkdir(parents=True, exist_ok=True)

    if not list(dl_dir.rglob("*.log")):
        print(f"[{name}] Downloading...")
        zip_path = dl_dir / "mhealth.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dl_dir)
        zip_path.unlink(missing_ok=True)

    print(f"[{name}] Processing...")

    # MHealth columns (0-indexed):
    # 0-2: chest acc, 3-4: ECG, 5-7: left-ankle acc, 8-10: left-ankle gyro,
    # 11-13: left-ankle mag, 14-16: right-wrist acc, 17-19: right-wrist gyro,
    # 20-22: right-wrist mag, 23: label
    # We use right-wrist: acc(14-16) + gyro(17-19) = 6 channels

    all_X, all_y, all_subj = [], [], []

    for subj_file in sorted(dl_dir.rglob("mHealth_subject*.log")):
        subj_id = int(subj_file.stem.split("subject")[-1])
        data = np.loadtxt(subj_file)

        labels = data[:, 23].astype(int)
        imu = data[:, 14:20].astype(np.float32)  # right-wrist acc+gyro

        # Remove null class (label 0)
        valid = labels > 0
        imu = imu[valid]
        labels = labels[valid] - 1  # 0-indexed

        # Already 50Hz, segment into 128-sample windows
        X_subj, y_subj = segment_windows(imu, labels, window_len=128)
        if len(X_subj) == 0:
            continue
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subj.append(np.full(len(X_subj), subj_id, dtype=np.int32))

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subj)

    X = normalize_per_subject(X, subjects)

    save_dataset(out_dir, X, y, subjects, {
        "n_classes": int(y.max() + 1),
        "sample_rate": 50,
        "window_length": 128,
        "channels": 6,
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "dataset_name": name,
        "n_subjects": len(np.unique(subjects)),
    })
    print(f"[{name}] Done.")


# ============================================================
# USC-HAD
# ============================================================
def download_usc_had(raw_dir: Path, processed_dir: Path):
    """USC-HAD: 14 subjects, 12 activities, 100Hz, hip-mounted."""
    name = "usc_had"
    out_dir = processed_dir / name
    if (out_dir / "X.npy").exists():
        print(f"[{name}] Already preprocessed, skipping.")
        return

    # USC-HAD requires manual download from https://sipi.usc.edu/had/
    # Check if already downloaded
    dl_dir = raw_dir / name
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Try direct download (may not work due to server restrictions)
    url = "https://sipi.usc.edu/had/USC-HAD.zip"
    zip_path = dl_dir / "USC-HAD.zip"

    if not list(dl_dir.rglob("*.mat")):
        print(f"[{name}] Downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dl_dir)
            zip_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[{name}] Auto-download failed: {e}")
            print(f"[{name}] Please manually download from https://sipi.usc.edu/had/")
            print(f"[{name}] Extract to {dl_dir}")
            return

    print(f"[{name}] Processing...")

    # USC-HAD uses .mat files
    try:
        from scipy.io import loadmat
    except ImportError:
        print(f"[{name}] scipy.io.loadmat required. pip install scipy")
        return

    all_X, all_y, all_subj = [], [], []

    for mat_file in sorted(dl_dir.rglob("*.mat")):
        try:
            mat = loadmat(str(mat_file))
        except Exception:
            continue

        # Check for expected keys
        if "sensor_readings" not in mat and "acc" not in mat:
            continue

        # Try to extract acc and gyro
        if "sensor_readings" in mat:
            readings = mat["sensor_readings"]
            if readings.shape[1] >= 6:
                imu = readings[:, :6].astype(np.float32)
            else:
                continue
        elif "acc" in mat and "gyro" in mat:
            acc = mat["acc"].astype(np.float32)
            gyro = mat["gyro"].astype(np.float32)
            imu = np.column_stack([acc, gyro])
        else:
            continue

        # Extract label and subject from filename
        # Typical: Subject1_a1_t1.mat
        fname = mat_file.stem
        parts = fname.split("_")
        try:
            subj_id = int("".join(filter(str.isdigit, parts[0])))
            act_id = int("".join(filter(str.isdigit, parts[1]))) - 1
        except (ValueError, IndexError):
            continue

        # Resample 100Hz -> 50Hz
        imu = resample_signal(imu, 100, 50)
        labels = np.full(len(imu), act_id, dtype=np.int64)

        X_seg, y_seg = segment_windows(imu, labels, window_len=128)
        if len(X_seg) == 0:
            continue

        all_X.append(X_seg)
        all_y.append(y_seg)
        all_subj.append(np.full(len(X_seg), subj_id, dtype=np.int32))

    if not all_X:
        print(f"[{name}] No data loaded. Check directory structure.")
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
        "sample_rate": 50,
        "window_length": 128,
        "channels": 6,
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "dataset_name": name,
        "n_subjects": len(np.unique(subjects)),
    })
    print(f"[{name}] Done.")


# ============================================================
# Realworld
# ============================================================
def download_realworld(raw_dir: Path, processed_dir: Path):
    """Realworld HAR: 15 subjects, 8 activities, 50Hz, 7 body positions."""
    name = "realworld"
    out_dir = processed_dir / name
    if (out_dir / "X.npy").exists():
        print(f"[{name}] Already preprocessed, skipping.")
        return

    # Realworld dataset is distributed as per-subject zip files
    base_url = "https://sensor.informatik.uni-mannheim.de/realworld2016/dataset"
    dl_dir = raw_dir / name
    dl_dir.mkdir(parents=True, exist_ok=True)

    activities = ["climbing_down", "climbing_up", "jumping", "lying",
                  "running", "sitting", "standing", "walking"]
    sensor_positions = ["waist"]  # Use waist for consistency

    print(f"[{name}] Downloading (per-subject zips)...")
    all_X, all_y, all_subj = [], [], []

    for subj_id in range(1, 16):  # 15 subjects
        subj_dir = dl_dir / f"proband{subj_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        for act_idx, activity in enumerate(activities):
            for sensor_pos in sensor_positions:
                # Try to download acc and gyro files
                for sensor_type in ["acc", "Gyroscope"]:
                    fname = f"{sensor_type}_{activity}_{subj_id}.csv"
                    local_path = subj_dir / fname

                    if not local_path.exists():
                        file_url = f"{base_url}/{fname}"
                        # Realworld uses a different URL pattern per subject
                        # Try the zip approach
                        zip_url = (f"{base_url}/realworld2016_dataset_"
                                   f"proband{subj_id}.zip")
                        # This dataset often requires manual download
                        pass

        # If zip files exist, extract
        for zf_path in dl_dir.glob(f"*proband{subj_id}*.zip"):
            try:
                with zipfile.ZipFile(zf_path, "r") as zf:
                    zf.extractall(subj_dir)
            except Exception:
                pass

    # Try to process whatever we have
    for subj_id in range(1, 16):
        subj_dir = dl_dir / f"proband{subj_id}"
        if not subj_dir.exists():
            continue

        for act_idx, activity in enumerate(activities):
            # Look for acc and gyro CSVs for waist
            acc_files = list(subj_dir.rglob(f"*acc*{activity}*waist*"))
            gyro_files = list(subj_dir.rglob(f"*yro*{activity}*waist*"))

            if not acc_files or not gyro_files:
                continue

            try:
                acc_data = np.loadtxt(acc_files[0], delimiter=",", skiprows=1,
                                      usecols=[1, 2, 3])
                gyro_data = np.loadtxt(gyro_files[0], delimiter=",", skiprows=1,
                                       usecols=[1, 2, 3])

                # Align lengths
                min_len = min(len(acc_data), len(gyro_data))
                imu = np.column_stack([
                    acc_data[:min_len],
                    gyro_data[:min_len],
                ]).astype(np.float32)

                labels = np.full(len(imu), act_idx, dtype=np.int64)

                # Already 50Hz
                X_seg, y_seg = segment_windows(imu, labels, window_len=128)
                if len(X_seg) > 0:
                    all_X.append(X_seg)
                    all_y.append(y_seg)
                    all_subj.append(np.full(len(X_seg), subj_id, dtype=np.int32))
            except Exception as e:
                continue

    if not all_X:
        print(f"[{name}] No data loaded. This dataset often requires manual download.")
        print(f"[{name}] Download from: https://sensor.informatik.uni-mannheim.de/")
        print(f"[{name}] Skipping for now — will proceed without it.")
        return

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subj)

    X = normalize_per_subject(X, subjects)

    save_dataset(out_dir, X, y, subjects, {
        "n_classes": int(y.max() + 1),
        "sample_rate": 50,
        "window_length": 128,
        "channels": 6,
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "dataset_name": name,
        "n_subjects": len(np.unique(subjects)),
    })
    print(f"[{name}] Done.")


# ============================================================
# Main
# ============================================================
def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("Downloading & Preprocessing HAR Benchmarks")
    print("=" * 60)
    print(f"Target format: 50Hz, 128 samples, 6ch (acc+gyro), z-score/subject\n")

    download_pamap2(raw_dir, processed_dir)
    print()
    download_dsads(raw_dir, processed_dir)
    print()
    download_mhealth(raw_dir, processed_dir)
    print()
    download_usc_had(raw_dir, processed_dir)
    print()
    download_realworld(raw_dir, processed_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for ds_name in ["uci_har", "pamap2", "dsads", "mhealth", "usc_had", "realworld"]:
        ds_dir = processed_dir / ds_name
        if (ds_dir / "X.npy").exists():
            X = np.load(ds_dir / "X.npy")
            y = np.load(ds_dir / "y.npy")
            subj = np.load(ds_dir / "subjects.npy")
            meta = dict(np.load(ds_dir / "meta.npz", allow_pickle=True))
            print(f"  {ds_name:12s}: {X.shape[0]:6d} windows, "
                  f"{len(np.unique(y)):2d} classes, "
                  f"{len(np.unique(subj)):2d} subjects, "
                  f"shape={X.shape[1:]}")
        else:
            print(f"  {ds_name:12s}: NOT AVAILABLE")


if __name__ == "__main__":
    main()
