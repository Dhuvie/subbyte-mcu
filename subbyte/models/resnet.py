"""
ResNet-8 Model Architecture.
Standard compact residual network for embedded vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, in_channels: int = 3):
        super().__init__()
        self.in_planes = int(16 * width_mult)
        
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        c1 = int(16 * width_mult)
        c2 = int(32 * width_mult)
        c3 = int(64 * width_mult)
        
        self.layer1 = self._make_layer(c1, stride=1)
        self.layer2 = self._make_layer(c2, stride=2)
        self.layer3 = self._make_layer(c3, stride=2)
        
        self.fc = nn.Linear(c3, num_classes)

    def _make_layer(self, planes: int, stride: int) -> nn.Module:
        layer = ResNetBasicBlock(self.in_planes, planes, stride)
        self.in_planes = planes
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        out = self.fc(out)
        return out
