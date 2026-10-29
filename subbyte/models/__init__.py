"""
Model Registry for TinyML Benchmarks.
"""

import torch.nn as nn
from subbyte.models.resnet import ResNet8
from subbyte.models.separable_cnn import SeparableCNN
try:
    from subbyte.models.mobilenet import MobileNetV1Small
except ImportError:
    MobileNetV1Small = None
from subbyte.models.dscnn import DSCNNS
from subbyte.models.cnn1d import CNN1D
from subbyte.models.fmnist_cnn import FashionCNN

def get_model(dataset: str, model_name: str, num_classes: int = 10) -> nn.Module:
    d = dataset.lower()
    m = model_name.lower()
    
    if 'resnet8_2x' in m:
        return ResNet8(num_classes=num_classes, width_mult=2.0)
    elif 'resnet8' in m:
        return ResNet8(num_classes=num_classes, width_mult=1.0)
    elif 'papercnn' in m or 'separable' in m or 'dsconv' in m:
        return SeparableCNN(num_classes=num_classes)
    elif 'mobilenet' in m:
        return MobileNetV1Small(num_classes=num_classes)
    elif 'dscnn' in m:
        return DSCNNS(num_classes=num_classes)
    elif 'cnn1d' in m or '1d' in m:
        return CNN1D(num_classes=num_classes)
    elif 'fmnist' in m:
        return FashionCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}' for dataset '{dataset}'")
