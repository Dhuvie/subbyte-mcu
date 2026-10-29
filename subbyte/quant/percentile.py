"""
Percentile Range Selection for Layer Quantization.
Critical layers: [0.05, 99.95] percentiles.
Standard layers: [min, 99.9] percentiles.
"""

import torch
from typing import Tuple

def compute_percentile_range(
    weights: torch.Tensor,
    is_critical: bool,
    per_channel: bool = True,
    channel_dim: int = 0,
    p_low_critical: float = 0.05,
    p_high_critical: float = 99.95,
    p_high_standard: float = 99.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes w_min and w_max per channel or globally based on layer criticality."""
    if not per_channel or weights.ndim <= 1:
        w_flat = weights.flatten()
        if is_critical:
            q_low = torch.quantile(w_flat, p_low_critical / 100.0)
            q_high = torch.quantile(w_flat, p_high_critical / 100.0)
            return q_low, q_high
        else:
            w_min = torch.min(w_flat)
            q_high = torch.quantile(w_flat, p_high_standard / 100.0)
            return w_min, q_high

    num_channels = weights.shape[channel_dim]
    mins = []
    maxs = []
    
    permuted = weights.transpose(0, channel_dim).contiguous()
    flat_channels = permuted.view(num_channels, -1)
    
    for c in range(num_channels):
        w_c = flat_channels[c]
        if is_critical:
            c_min = torch.quantile(w_c, p_low_critical / 100.0)
            c_max = torch.quantile(w_c, p_high_critical / 100.0)
        else:
            c_min = torch.min(w_c)
            c_max = torch.quantile(w_c, p_high_standard / 100.0)
        mins.append(c_min)
        maxs.append(c_max)
        
    return torch.stack(mins), torch.stack(maxs)
