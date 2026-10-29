"""
Noise Regularization for Low-bit Quantization.
Adds Gaussian noise N(0, (gamma * sigma)^2) to layers with bitwidth <= 4.
"""

import torch

def apply_weight_noise(
    weights: torch.Tensor,
    bits: int,
    gamma: float = 0.01,
    seed: int = None
) -> torch.Tensor:
    """
    Applies Gaussian noise regularization if bits <= 4.
    If bits > 4 or gamma <= 0, returns unchanged copy.
    """
    if bits > 4 or gamma <= 0.0:
        return weights.clone()
    
    sigma = torch.std(weights)
    if sigma < 1e-8:
        return weights.clone()
        
    generator = None
    if seed is not None:
        generator = torch.Generator(device=weights.device).manual_seed(seed)
        
    noise = torch.randn(weights.shape, generator=generator, device=weights.device, dtype=weights.dtype)
    noise = noise * (gamma * sigma)
    return weights + noise
