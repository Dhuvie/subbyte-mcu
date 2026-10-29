"""
UCI HAR (Human Activity Recognition) Dataset Loader.
Loads 9-channel raw inertial signals (128 samples per window) across 6 activities.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from subbyte.datasets import resolve_data_dir

class HARDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def _load_signals(base_dir: Path, split: str = "train") -> np.ndarray:
    signal_names = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_"
    ]
    signals = []
    signals_dir = base_dir / split / "Inertial Signals"
    for name in signal_names:
        fname = signals_dir / f"{name}{split}.txt"
        data = np.loadtxt(fname)
        signals.append(data)
    return np.stack(signals, axis=1).astype(np.float32)

def get_har_loaders(
    val_size: int = 1500,
    calib_size: int = 500,
    batch_size: int = 64,
    seed: int = 42
) -> Dict[str, Any]:
    data_dir = resolve_data_dir()
    cache_f = data_dir / "har_cached.npz"
    if cache_f.exists():
        cached = np.load(cache_f)
        all_train_x = cached['train_x']
        all_train_y = cached['train_y']
        test_x = cached['test_x']
        test_y = cached['test_y']
    else:
        har_dir = data_dir / "uci_har" / "UCI HAR Dataset"
        if not har_dir.exists():
            har_dir = data_dir / "UCI HAR Dataset"
        all_train_x = _load_signals(har_dir, "train")
        all_train_y = np.loadtxt(har_dir / "train" / "y_train.txt").astype(np.int64) - 1
        test_x = _load_signals(har_dir, "test")
        test_y = np.loadtxt(har_dir / "test" / "y_test.txt").astype(np.int64) - 1
        np.savez_compressed(cache_f, train_x=all_train_x, train_y=all_train_y, test_x=test_x, test_y=test_y)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(all_train_x, all_train_y))
    
    train_x, train_y = all_train_x[train_idx], all_train_y[train_idx]
    val_x, val_y = all_train_x[val_idx], all_train_y[val_idx]
    
    sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed)
    _, cal_idx = next(sss_cal.split(train_x, train_y))
    calib_x, calib_y = train_x[cal_idx], train_y[cal_idx]
    
    return {
        'train_loader': DataLoader(HARDataset(train_x, train_y), batch_size=batch_size, shuffle=True, pin_memory=True),
        'val_loader': DataLoader(HARDataset(val_x, val_y), batch_size=batch_size, shuffle=False, pin_memory=True),
        'test_loader': DataLoader(HARDataset(test_x, test_y), batch_size=batch_size, shuffle=False, pin_memory=True),
        'calib_loader': DataLoader(HARDataset(calib_x, calib_y), batch_size=min(batch_size, calib_size), shuffle=False),
        'num_classes': 6,
        'input_shape': (9, 128),
        'augment': False
    }
