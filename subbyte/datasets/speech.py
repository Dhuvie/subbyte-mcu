"""
Google Speech Commands v2 Keyword Spotting Loader.
12 classes: 10 target keywords, plus silence and unknown.
Loads 49x10 pre-computed MFCC representations.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from subbyte.datasets import resolve_data_dir

TARGET_WORDS = ['silence', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown']
WORD_TO_IDX = {w: i for i, w in enumerate(TARGET_WORDS)}

class SpeechCommandsDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # features: [N, 49, 10] -> [N, 1, 49, 10]
        self.features = torch.from_numpy(features).float().unsqueeze(1)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_speech_loaders(
    val_size: int = 1500,
    calib_size: int = 500,
    batch_size: int = 64,
    seed: int = 42
) -> Dict[str, Any]:
    data_dir = resolve_data_dir()
    cache_f = data_dir / "speech_mfcc_cached.npz"
    if not cache_f.exists():
        raise FileNotFoundError(f"Speech Commands MFCC cache not found at {cache_f}")
        
    data = np.load(cache_f)
    features, labels = data['features'], data['labels']
    
    # Stratified 80/10/10 train/val/test split
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(features, labels))
    
    tv_features, tv_labels = features[train_val_idx], labels[train_val_idx]
    test_features, test_labels = features[test_idx], labels[test_idx]
    
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss_val.split(tv_features, tv_labels))
    
    train_features, train_labels = tv_features[train_idx], tv_labels[train_idx]
    val_features, val_labels = tv_features[val_idx], tv_labels[val_idx]
    
    sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed)
    _, cal_idx = next(sss_cal.split(train_features, train_labels))
    calib_features, calib_labels = train_features[cal_idx], train_labels[cal_idx]
    
    return {
        'train_loader': DataLoader(SpeechCommandsDataset(train_features, train_labels), batch_size=batch_size, shuffle=True, pin_memory=True),
        'val_loader': DataLoader(SpeechCommandsDataset(val_features, val_labels), batch_size=batch_size, shuffle=False, pin_memory=True),
        'test_loader': DataLoader(SpeechCommandsDataset(test_features, test_labels), batch_size=batch_size, shuffle=False, pin_memory=True),
        'calib_loader': DataLoader(SpeechCommandsDataset(calib_features, calib_labels), batch_size=min(batch_size, calib_size), shuffle=False),
        'num_classes': 12,
        'input_shape': (1, 49, 10),
        'augment': False
    }
