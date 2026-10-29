"""
Empirical Quantization Sensitivity Analysis and Hessian Trace Estimation.
Measures single-layer accuracy drop and Hutchinson Hessian trace across layers.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from subbyte.quant.affine import compute_quant_params, quantize_affine, dequantize_affine

def compute_hutchinson_hessian_trace(
    model: nn.Module,
    val_loader: Any,
    criterion: Any,
    device: torch.device,
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Computes per-layer Hessian trace using Hutchinson's stochastic trace estimator:
    Tr(H_l) = E [ v^T H_l v ] = E [ v^T grad(grad(L) * v) ]
    with Rademacher random vectors v in {-1, +1}^N.
    """
    model.eval()
    traces = {}
    named_params = [
        (name, p) for name, p in model.named_parameters()
        if p.requires_grad and ('weight' in name) and p.ndim in (2, 3, 4)
    ]
    for name, _ in named_params:
        traces[name] = 0.0
        
    total_samples = 0
    for batch_x, batch_y in val_loader:
        if total_samples >= num_samples:
            break
            
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if batch_x.dtype == torch.uint8:
            batch_x = batch_x.float().div_(255.0)
        elif batch_x.dtype != torch.float32:
            batch_x = batch_x.float()
        if batch_y.dtype != torch.long:
            batch_y = batch_y.long()
            
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        params = [p for _, p in named_params]
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        v_list = [torch.randint(0, 2, p.shape, device=device).float() * 2.0 - 1.0 for p in params]
        grad_v = sum((g * v).sum() for g, v in zip(grads, v_list))
        hv_list = torch.autograd.grad(grad_v, params, retain_graph=False)
        
        for (name, p), v, hv in zip(named_params, v_list, hv_list):
            traces[name] += float((v * hv).sum().item())
            
        total_samples += batch_x.shape[0]
        
    for name in traces:
        traces[name] /= max(1, total_samples)
        
    return traces

def measure_layer_sensitivity(
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
    bits_list: List[int] = [2, 3, 4, 6, 8]
) -> pd.DataFrame:
    """
    Measures empirical single-layer accuracy degradation.
    For each layer l and bitwidth b:
      Quantizes ONLY layer l (others remain FP32),
      measures validation accuracy drop.
    """
    model.eval()
    correct_fp32 = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
            elif x.dtype != torch.float32:
                x = x.float()
            if y.dtype != torch.long:
                y = y.long()
            out = model(x)
            pred = out.argmax(dim=-1)
            correct_fp32 += (pred == y).sum().item()
            total += y.size(0)
            
    fp32_acc = (correct_fp32 / float(total)) * 100.0 if total > 0 else 0.0
    
    quantizable_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            quantizable_layers.append((name, mod))
            
    records = []
    for layer_idx, (layer_name, mod) in enumerate(quantizable_layers):
        orig_weight = mod.weight.data.clone()
        for b in bits_list:
            with torch.no_grad():
                w = orig_weight
                w_min = torch.min(w)
                w_max = torch.max(w)
                scale, zp = compute_quant_params(w_min, w_max, bits=b, symmetric=False)
                q = quantize_affine(w, scale, zp, bits=b)
                dq = dequantize_affine(q, scale, zp)
                mod.weight.data.copy_(dq)
                
                correct_q = 0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    if x.dtype == torch.uint8:
                        x = x.float().div_(255.0)
                    elif x.dtype != torch.float32:
                        x = x.float()
                    if y.dtype != torch.long:
                        y = y.long()
                    out = model(x)
                    pred = out.argmax(dim=-1)
                    correct_q += (pred == y).sum().item()
                    
                q_acc = (correct_q / float(total)) * 100.0 if total > 0 else 0.0
                drop = max(0.0, fp32_acc - q_acc)
                records.append({
                    'layer_idx': layer_idx,
                    'layer_name': layer_name,
                    'bits': b,
                    'acc_drop': drop,
                    'q_acc': q_acc,
                    'fp32_acc': fp32_acc
                })
                mod.weight.data.copy_(orig_weight)
                
    return pd.DataFrame(records)
