"""
1D-CNN Model Architecture for Sensor Time-Series Data (UCI HAR).
Input: 9-axis IMU accelerometer/gyroscope channels across 128 timesteps (~40K params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 5, stride: int = 1, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_c)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = self.pool(out)
        return out

class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = 6):
        super().__init__()
        self.block1 = ConvBlock1D(in_channels, 64, kernel_size=5, pool=True)
        self.block2 = ConvBlock1D(64, 128, kernel_size=5, pool=True)
        self.block3 = ConvBlock1D(128, 128, kernel_size=5, pool=True)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > x.shape[2] and x.shape[2] in (1, 9):
            x = x.transpose(1, 2)
            
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = F.adaptive_avg_pool1d(out, 1).flatten(1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
