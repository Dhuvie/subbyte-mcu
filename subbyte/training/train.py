"""
Model Training and Fine-Tuning Routines.
Provides standard training routines for FP32 baselines and brief fine-tuning for quantized models.
"""

import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from subbyte.datasets.cifar import apply_gpu_augmentation
from subbyte.training.evaluate import evaluate_loader

def train_fp32_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    epochs: int = 40,
    lr: float = 0.05,
    is_cifar: bool = True
) -> Dict[str, Any]:
    """Trains an FP32 model with SGD, momentum 0.9, and cosine annealing learning rate schedule."""
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    start_time = time.time()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
            elif x.dtype != torch.float32:
                x = x.float()
            if y.dtype != torch.long:
                y = y.long()
            if is_cifar:
                x = apply_gpu_augmentation(x)
                
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        scheduler.step()
        val_acc = evaluate_loader(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    training_time = time.time() - start_time
    return {
        'best_val_acc': best_val_acc,
        'state_dict': best_state,
        'training_time_s': training_time,
        'epoch_losses': epoch_losses
    }

def finetune_quantized_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    is_cifar: bool = True
) -> float:
    """Brief fine-tuning schedule for quantized representations using Adam."""
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
            elif x.dtype != torch.float32:
                x = x.float()
            if y.dtype != torch.long:
                y = y.long()
            if is_cifar:
                x = apply_gpu_augmentation(x)
                
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return evaluate_loader(model, val_loader, device)
