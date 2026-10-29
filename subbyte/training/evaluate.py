"""
Evaluation Routines for PyTorch Models.
"""

from typing import Any
import torch
import torch.nn as nn

def evaluate_loader(model: nn.Module, loader: Any, device: torch.device, dataset_name: str = "") -> float:
    """Evaluates top-1 classification accuracy on a dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
            elif x.dtype != torch.float32:
                x = x.float()
            if y.dtype != torch.long:
                y = y.long()
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return (correct / float(total)) * 100.0 if total > 0 else 0.0
