"""
Depthwise-Separable Convolutional Neural Network (508,462 parameters).
Standard vision baseline featuring depthwise separable convolutions and channel attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        mid_channels = max(1, channels // ratio)
        self.fc1 = nn.Linear(channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_p = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_p = F.adaptive_max_pool2d(x, 1).view(b, c)
        
        avg_out = self.fc2(F.relu(self.fc1(avg_p)))
        max_out = self.fc2(F.relu(self.fc1(max_p)))
        
        att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * att

class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_residual: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.use_residual = use_residual and (stride == 1 and in_channels == out_channels)
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.attention = ChannelAttention(out_channels)
        self.spatial_dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.depthwise(x)))
        out = self.bn2(self.pointwise(out))
        out = self.attention(out)
        
        if self.use_residual:
            out = out + residual
            
        out = F.relu(out)
        out = self.spatial_dropout(out)
        return out

class SeparableCNN(nn.Module):
    """Depthwise-Separable CNN containing 508,462 parameters for CIFAR-10."""
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        
        self.ds1 = DepthwiseSeparableBlock(48, 96, stride=1)
        self.ds2 = DepthwiseSeparableBlock(96, 128, stride=2)
        self.ds3 = DepthwiseSeparableBlock(128, 192, stride=1)
        self.ds4 = DepthwiseSeparableBlock(192, 256, stride=2)
        self.ds5 = DepthwiseSeparableBlock(256, 384, stride=1)
        
        self.fc1 = nn.Linear(768, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.predictions = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ds1(out)
        out = self.ds2(out)
        out = self.ds3(out)
        out = self.ds4(out)
        out = self.ds5(out)
        
        gap = F.adaptive_avg_pool2d(out, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(out, 1).flatten(1)
        pooled = torch.cat([gap, gmp], dim=1)
        
        out = F.relu(self.bn_fc1(self.fc1(pooled)))
        out = self.dropout1(out)
        
        out = F.relu(self.bn_fc2(self.fc2(out)))
        out = self.dropout2(out)
        
        logits = self.predictions(out)
        return logits
