"""
PyTorch Fake-Quantization Modules and Straight-Through Estimator (STE).
Implements weight and activation fake-quantization with per-channel calibration.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from subbyte.quant.affine import compute_quant_params, quantize_affine, dequantize_affine
from subbyte.quant.percentile import compute_percentile_range
from subbyte.quant.noise import apply_weight_noise

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, bits):
        q = quantize_affine(x, scale, zero_point, bits)
        dq = dequantize_affine(q, scale, zero_point)
        return dq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

def apply_error_correction(
    orig_w: torch.Tensor,
    quant_w: torch.Tensor,
    is_conv: bool = False,
    channel_dim: int = 0
) -> torch.Tensor:
    """Mean quantization residual compensation."""
    w_corrected = quant_w.clone()
    std_global = torch.std(orig_w)
    eps_global = torch.mean(orig_w - quant_w)
    
    if torch.abs(eps_global) > 0.2 * std_global:
        w_corrected = w_corrected + eps_global
        
    if is_conv and orig_w.ndim >= 3:
        num_channels = orig_w.shape[channel_dim]
        for c in range(num_channels):
            orig_c = orig_w.select(channel_dim, c)
            quant_c = w_corrected.select(channel_dim, c)
            std_c = torch.std(orig_c)
            eps_c = torch.mean(orig_c - quant_c)
            if std_c > 1e-8 and torch.abs(eps_c) > 0.2 * std_c:
                w_corrected.select(channel_dim, c).add_(eps_c)
                
    return w_corrected

class FakeQuantizeWeight(nn.Module):
    def __init__(
        self,
        bits: int = 8,
        is_critical: bool = False,
        per_channel: bool = True,
        channel_dim: int = 0,
        enable_noise: bool = True,
        noise_gamma: float = 0.01,
        enable_error_correction: bool = True
    ):
        super().__init__()
        self.bits = bits
        self.is_critical = is_critical
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.enable_noise = enable_noise
        self.noise_gamma = noise_gamma
        self.enable_error_correction = enable_error_correction
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.enable_noise and self.bits <= 4:
            reg_weight = apply_weight_noise(weight, self.bits, self.noise_gamma)
        else:
            reg_weight = weight

        w_min, w_max = compute_percentile_range(
            reg_weight,
            is_critical=self.is_critical,
            per_channel=self.per_channel,
            channel_dim=self.channel_dim
        )

        scale, zero_point = compute_quant_params(w_min, w_max, self.bits, symmetric=False)
        self.scale = scale
        self.zero_point = zero_point

        if self.per_channel and weight.ndim > 1:
            shape = [1] * weight.ndim
            shape[self.channel_dim] = -1
            scale_b = scale.view(shape)
            zero_point_b = zero_point.view(shape)
        else:
            scale_b = scale
            zero_point_b = zero_point

        q_weight = STEQuantize.apply(reg_weight, scale_b, zero_point_b, self.bits)

        if self.enable_error_correction and self.bits < 8:
            is_conv = (weight.ndim == 4)
            q_weight = apply_error_correction(weight, q_weight, is_conv=is_conv, channel_dim=self.channel_dim)

        return q_weight

class FakeQuantizeActivation(nn.Module):
    """Calibrated INT8 activation fake-quantization with STE."""
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.register_buffer("running_min", torch.tensor(-1.0))
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.calibrated = False

    def update_range(self, x: torch.Tensor):
        x_min = x.min()
        x_max = x.max()
        scale, zero_point = compute_quant_params(x_min, x_max, self.bits, symmetric=False)
        self.running_min.copy_(x_min)
        self.running_max.copy_(x_max)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        self.calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or not self.calibrated:
            self.update_range(x)
        return STEQuantize.apply(x, self.scale, self.zero_point, self.bits)
