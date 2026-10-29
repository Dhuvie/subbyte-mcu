"""
DS-CNN-S (Depthwise Separable CNN Small) for Keyword Spotting.
Benchmark keyword spotting model for Google Speech Commands (~24K params).
Input: 49x10 (Time x Frequency MFCC features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DSBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.dw(x)))
        out = F.relu(self.bn2(self.pw(out)))
        return out

class DSCNNS(nn.Module):
    def __init__(self, num_classes: int = 12, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(10, 4), stride=(2, 2), padding=(5, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.ds1 = DSBlock(64)
        self.ds2 = DSBlock(64)
        self.ds3 = DSBlock(64)
        self.ds4 = DSBlock(64)
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ds1(out)
        out = self.ds2(out)
        out = self.ds3(out)
        out = self.ds4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        out = self.fc(out)
        return out
