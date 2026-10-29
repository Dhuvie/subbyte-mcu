"""
Representative Formula-Based Layer-Importance Heuristic.
Implements a multi-term statistical heuristic combining position, magnitude,
variance, and size terms to score layer importance for mixed-precision allocation.
"""

import math
from typing import Dict, List, Tuple, Any
import torch

def compute_layer_statistics(w: torch.Tensor) -> Dict[str, float]:
    """Computes summary statistics for a weight tensor."""
    w_flat = w.detach().cpu().flatten().float()
    mu = torch.mean(w_flat).item()
    sigma = torch.std(w_flat).item()
    w_min = torch.min(w_flat).item()
    w_max = torch.max(w_flat).item()
    w_range = max(w_max - w_min, 1e-6)
    sparsity = torch.mean((torch.abs(w_flat) < 1e-6).float()).item()
    mean_abs = torch.mean(torch.abs(w_flat)).item()
    
    return {
        'mu': mu,
        'sigma': sigma,
        'range': w_range,
        'sparsity': sparsity,
        'mean_abs': mean_abs,
        'num_elements': w_flat.numel()
    }

def compute_position_factor(layer_idx: int, total_layers: int, is_critical: bool) -> float:
    """Smooth position factor: 1.0 at endpoints, smoothly dipping toward 0.5 at midpoint."""
    if is_critical or total_layers <= 1:
        return 1.0
    r = float(layer_idx) / float(total_layers - 1)
    return 1.0 - 0.5 * math.sin(math.pi * r)

def compute_size_factor(num_elements: int) -> float:
    """Logarithmic size damping factor."""
    if num_elements <= 0:
        return 1.0
    log_size = math.log10(max(1.0, float(num_elements)))
    return 1.0 - min(0.95, log_size / 7.0)

def compute_importance_score(
    stats: Dict[str, float],
    pos_factor: float,
    weights: Tuple[float, float, float, float, float] = (0.4, 0.3, 0.15, 0.1, 0.05),
    eps: float = 1e-6
) -> float:
    """
    Computes heuristic importance score:
    Score = w1 * f_pos + w2 * (mean(|W|) / (sigma + eps))
          + w3 * (1 - xi) * (1 - |mu|)
          + w4 * (sigma^2 / (range + eps))
          + w5 * f_size(N)
    """
    w1, w2, w3, w4, w5 = weights
    term_pos = w1 * pos_factor
    term_mag = w2 * (stats['mean_abs'] / (stats['sigma'] + eps))
    term_dist = w3 * (1.0 - stats['sparsity']) * (1.0 - abs(stats['mu']))
    term_var = w4 * ((stats['sigma'] ** 2) / (stats['range'] + eps))
    term_size = w5 * compute_size_factor(stats['num_elements'])
    
    score = term_pos + term_mag + term_dist + term_var + term_size
    return max(0.0, min(1.0, float(score)))

def assign_bits_from_thresholds(
    score: float,
    is_critical: bool,
    thresholds: Tuple[float, float, float, float, float] = (0.85, 0.78, 0.72, 0.68, 0.65),
    min_bits: int = 3,
    deployable: bool = False
) -> int:
    """Assigns bitwidth based on threshold intervals."""
    if is_critical:
        return 8
        
    t_high, t6, t5, t4, t3 = thresholds
    
    if deployable:
        if score > t_high:
            raw_bits = 8
        elif score > t4:
            raw_bits = 4
        else:
            raw_bits = 2
        return max(min_bits, raw_bits)

    if score > t_high:
        raw_bits = 8
    elif score > t6:
        raw_bits = 7
    elif score > t5:
        raw_bits = 6
    elif score > t4:
        raw_bits = 5
    elif score > t3:
        raw_bits = 4
    else:
        raw_bits = 3
    return max(min_bits, raw_bits)

def compute_model_scores(named_weights: List[Tuple[str, torch.Tensor, bool]]) -> List[Dict[str, Any]]:
    """Scores all layers in a network."""
    total_layers = len(named_weights)
    results = []
    for idx, (name, w, is_crit) in enumerate(named_weights):
        stats = compute_layer_statistics(w)
        pos = compute_position_factor(idx, total_layers, is_crit)
        score = compute_importance_score(stats, pos)
        results.append({
            'name': name,
            'is_critical': is_crit,
            'stats': stats,
            'pos_factor': pos,
            'score': score
        })
    return results
