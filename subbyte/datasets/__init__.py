"""
Dataset Loaders Registry for TinyML Benchmarks.
Provides unified loaders for CIFAR-10, CIFAR-100, Fashion-MNIST, UCI HAR, and Google Speech Commands.
"""

import os
from pathlib import Path
from typing import Dict, Any

def resolve_data_dir() -> Path:
    if "SUBBYTE_DATA_DIR" in os.environ:
        p = Path(os.environ["SUBBYTE_DATA_DIR"])
        if p.exists():
            return p
    # Standard local project data dir
    local_data = Path(__file__).resolve().parents[2] / "data"
    if (local_data / "speech_mfcc_cached.npz").exists() or (local_data / "har_cached.npz").exists():
        return local_data
    # Project root data directory
    project_root_data = Path(__file__).resolve().parents[2].parent / "data" / "raw"
    if project_root_data.exists():
        return project_root_data
    return local_data

def get_data(dataset_name: str, seed: int = 42, batch_size: int = 128) -> Dict[str, Any]:
    d = dataset_name.lower()
    if d == 'cifar10':
        from subbyte.datasets.cifar import get_cifar10_loaders
        return get_cifar10_loaders(seed=seed, batch_size=batch_size)
    elif d == 'cifar100':
        from subbyte.datasets.cifar import get_cifar100_loaders
        return get_cifar100_loaders(seed=seed, batch_size=batch_size)
    elif d == 'fmnist' or d == 'fashion_mnist':
        from subbyte.datasets.fmnist import get_fmnist_loaders
        return get_fmnist_loaders(seed=seed, batch_size=batch_size)
    elif d == 'har' or d == 'uci_har':
        from subbyte.datasets.har import get_har_loaders
        return get_har_loaders(seed=seed, batch_size=batch_size)
    elif d == 'speech' or d == 'speech_commands':
        from subbyte.datasets.speech import get_speech_loaders
        return get_speech_loaders(seed=seed, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
