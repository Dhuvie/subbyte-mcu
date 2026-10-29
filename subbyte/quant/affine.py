"""
Affine Quantization Functions (Per-Tensor and Per-Channel).
Supports asymmetric and symmetric integer quantization for arbitrary bitwidths 2..8.
"""

import torch
from typing import Tuple

def compute_quant_params(
    w_min: torch.Tensor,
    w_max: torch.Tensor,
    bits: int,
    symmetric: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scale and zero-point for given min/max range and bitwidth.
    For asymmetric: q in [0, 2^bits - 1]
    For symmetric: q in [-(2^(bits-1)), 2^(bits-1) - 1] or unsigned symmetric [0, 2^bits - 1]
    """
    qmin = 0
    qmax = (1 << bits) - 1
    diff = torch.clamp(w_max - w_min, min=1e-8)

    if symmetric:
        max_abs = torch.max(torch.abs(w_min), torch.abs(w_max))
        max_abs = torch.clamp(max_abs, min=1e-8)
        scale = (2.0 * max_abs) / float(qmax - qmin)
        zero_point = torch.full_like(scale, (qmax + qmin) / 2.0)
    else:
        scale = diff / float(qmax - qmin)
        zero_point = torch.round(-w_min / scale)
        zero_point = torch.clamp(zero_point, float(qmin), float(qmax))

    return scale, zero_point

def quantize_affine(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int
) -> torch.Tensor:
    """Quantizes floating point tensor to integer grid in [0, 2^bits - 1]."""
    qmin = 0
    qmax = (1 << bits) - 1
    q = torch.round(tensor / scale + zero_point)
    return torch.clamp(q, float(qmin), float(qmax))

def dequantize_affine(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantizes integer grid back to floating point."""
    return scale * (q - zero_point)
