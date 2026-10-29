"""
Small 3-Layer CNN for Fashion-MNIST (28x28 grayscale, ~80K parameters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        out = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        out = F.max_pool2d(F.relu(self.bn2(self.conv2(out))), 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
