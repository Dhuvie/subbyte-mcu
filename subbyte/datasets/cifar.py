"""
CIFAR-10 and CIFAR-100 Dataset Loaders.
Loads from cached numpy tensors with fast GPU batch augmentation.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from subbyte.datasets import resolve_data_dir

def apply_gpu_augmentation(x: torch.Tensor) -> torch.Tensor:
    """Fast vectorized GPU batch augmentation: horizontal flip + random 4px crop."""
    b, c, h, w = x.shape
    flip_mask = (torch.rand(b, 1, 1, 1, device=x.device) > 0.5)
    x = torch.where(flip_mask, torch.flip(x, [-1]), x)
    padded = F.pad(x, (4, 4, 4, 4), mode='reflect')
    crop_h = torch.randint(0, 9, (1,), device=x.device).item()
    crop_w = torch.randint(0, 9, (1,), device=x.device).item()
    return padded[:, :, crop_h:crop_h+32, crop_w:crop_w+32]

def get_cifar10_loaders(
    val_size: int = 5000,
    calib_size: int = 500,
    batch_size: int = 128,
    seed: int = 42,
    augment: bool = True
) -> Dict[str, Any]:
    data_dir = resolve_data_dir()
    cache_f = data_dir / "cifar10_tensors.npz"
    if not cache_f.exists():
        raise FileNotFoundError(f"CIFAR-10 tensor cache not found at {cache_f}")
        
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
    
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long())
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y).long())
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).long())
    calib_ds = TensorDataset(torch.from_numpy(calib_x), torch.from_numpy(calib_y).long())
    
    return {
        'train_loader': DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        'val_loader': DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'test_loader': DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'calib_loader': DataLoader(calib_ds, batch_size=min(batch_size, calib_size), shuffle=False),
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'augment': augment
    }

def get_cifar100_loaders(
    val_size: int = 5000,
    calib_size: int = 500,
    batch_size: int = 128,
    seed: int = 42,
    augment: bool = True
) -> Dict[str, Any]:
    data_dir = resolve_data_dir()
    cache_f = data_dir / "cifar100_tensors.npz"
    if not cache_f.exists():
        raise FileNotFoundError(f"CIFAR-100 tensor cache not found at {cache_f}")
        
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
    
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long())
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y).long())
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).long())
    calib_ds = TensorDataset(torch.from_numpy(calib_x), torch.from_numpy(calib_y).long())
    
    return {
        'train_loader': DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        'val_loader': DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'test_loader': DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'calib_loader': DataLoader(calib_ds, batch_size=min(batch_size, calib_size), shuffle=False),
        'num_classes': 100,
        'input_shape': (3, 32, 32),
        'augment': augment
    }
