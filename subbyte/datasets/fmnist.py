"""
Fashion-MNIST Dataset Loader.
28x28 grayscale image dataset across 10 clothing categories.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from subbyte.datasets import resolve_data_dir

class FMNISTTensorDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx].float() / 255.0, self.labels[idx]

def get_fmnist_loaders(
    val_size: int = 5000,
    calib_size: int = 500,
    batch_size: int = 128,
    seed: int = 42
) -> Dict[str, Any]:
    data_dir = resolve_data_dir()
    cache_f = data_dir / "fmnist_tensors.npz"
    if not cache_f.exists():
        raise FileNotFoundError(f"Fashion-MNIST tensor cache not found at {cache_f}")
        
    data = np.load(cache_f)
    all_train_x, all_train_y = data['x_train'], data['y_train']
    test_x, test_y = data['x_test'], data['y_test']
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(all_train_x, all_train_y))
    
    train_x, train_y = all_train_x[train_idx], all_train_y[train_idx]
    val_x, val_y = all_train_x[val_idx], all_train_y[val_idx]
    
    sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed)
    _, cal_idx = next(sss_cal.split(train_x, train_y))
    calib_x, calib_y = train_x[cal_idx], train_y[cal_idx]
    
    return {
        'train_loader': DataLoader(FMNISTTensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, pin_memory=True),
        'val_loader': DataLoader(FMNISTTensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False, pin_memory=True),
        'test_loader': DataLoader(FMNISTTensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, pin_memory=True),
        'calib_loader': DataLoader(FMNISTTensorDataset(calib_x, calib_y), batch_size=min(batch_size, calib_size), shuffle=False),
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'augment': False
    }
