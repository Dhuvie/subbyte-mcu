"""
Baseline Allocation Strategies.
1. Uniform Quantization (W2, W3, W4, W6, W8, with optional 8-bit endpoint protection)
2. Size-Only Allocation (larger layers assigned fewer bits)
3. Random Allocation (constrained random draws meeting budget within +/- 1%)
4. Inverted Score (negative control)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from subbyte.quant.budget import compute_allocation_storage, solve_budget_bisection
from subbyte.allocation.formula_heuristic import assign_bits_from_thresholds

def get_uniform_allocation(
    num_layers: int,
    bits: int,
    first_last_8: bool = False
) -> List[int]:
    """Generates uniform bit allocation."""
    alloc = [bits] * num_layers
    if first_last_8 and num_layers >= 2:
        alloc[0] = 8
        alloc[-1] = 8
    return alloc

def get_inverted_score_allocation(
    scores: List[float],
    critical_flags: List[bool],
    layer_metas: List[Dict[str, Any]],
    target_budget_bytes: int,
    deployable: bool = False,
    min_bits: int = 3
) -> Tuple[List[int], int]:
    """Negative control: higher importance receives lower bitwidth."""
    inv_scores = [1.0 - s for s in scores]
    
    def score_to_alloc(delta: float) -> List[int]:
        base_thresh = (0.85 + delta, 0.78 + delta, 0.72 + delta, 0.68 + delta, 0.65 + delta)
        return [
            assign_bits_from_thresholds(s, is_crit, thresholds=base_thresh, min_bits=min_bits, deployable=deployable)
            for s, is_crit in zip(inv_scores, critical_flags)
        ]
        
    alloc, actual_bytes, _ = solve_budget_bisection(
        score_fn=score_to_alloc,
        layer_metas=layer_metas,
        target_budget_bytes=target_budget_bytes
    )
    return alloc, actual_bytes

def get_size_only_allocation(
    layer_metas: List[Dict[str, Any]],
    critical_flags: List[bool],
    target_budget_bytes: int,
    deployable: bool = False,
    min_bits: int = 3
) -> Tuple[List[int], int]:
    """Size-only heuristic: bigger layers receive fewer bits."""
    sizes = [m['num_elements'] for m in layer_metas]
    max_s = max(sizes) if sizes else 1
    size_scores = [1.0 - (float(s) / float(max_s)) for s in sizes]
    
    def score_to_alloc(delta: float) -> List[int]:
        base_thresh = (0.85 + delta, 0.78 + delta, 0.72 + delta, 0.68 + delta, 0.65 + delta)
        return [
            assign_bits_from_thresholds(s, is_crit, thresholds=base_thresh, min_bits=min_bits, deployable=deployable)
            for s, is_crit in zip(size_scores, critical_flags)
        ]
        
    alloc, actual_bytes, _ = solve_budget_bisection(
        score_fn=score_to_alloc,
        layer_metas=layer_metas,
        target_budget_bytes=target_budget_bytes
    )
    return alloc, actual_bytes

def get_random_allocations(
    layer_metas: List[Dict[str, Any]],
    critical_flags: List[bool],
    target_budget_bytes: int,
    num_samples: int = 10,
    deployable: bool = False,
    min_bits: int = 3,
    seed: int = 42
) -> List[Tuple[List[int], int]]:
    """Draws random bit allocations hitting target budget within +/- 1%."""
    rng = np.random.RandomState(seed)
    valid_allocs = []
    bit_choices = [2, 4, 8] if deployable else [3, 4, 5, 6, 7, 8]
    if min_bits == 2 and not deployable:
        bit_choices = [2, 3, 4, 5, 6, 7, 8]
        
    attempts = 0
    max_attempts = 50000
    
    while len(valid_allocs) < num_samples and attempts < max_attempts:
        attempts += 1
        alloc = []
        for is_crit in critical_flags:
            if is_crit:
                alloc.append(8)
            else:
                b = int(rng.choice(bit_choices))
                alloc.append(max(min_bits, b))
                
        b_bytes = compute_allocation_storage(layer_metas, alloc)
        err = abs(b_bytes - target_budget_bytes) / float(target_budget_bytes)
        if err <= 0.01:
            valid_allocs.append((alloc, b_bytes))
            
    if len(valid_allocs) < num_samples:
        candidates = []
        for _ in range(500):
            alloc = [8 if is_crit else int(rng.choice(bit_choices)) for is_crit in critical_flags]
            b_bytes = compute_allocation_storage(layer_metas, alloc)
            candidates.append((alloc, b_bytes, abs(b_bytes - target_budget_bytes)))
        candidates.sort(key=lambda x: x[2])
        for a, b, _ in candidates[:num_samples]:
            valid_allocs.append((a, b))
            if len(valid_allocs) >= num_samples:
                break
                
    return valid_allocs[:num_samples]
