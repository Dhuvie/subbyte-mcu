"""
Budget Solver and Storage Accounting.
Ensures mixed-precision allocations hit the specified target memory budgets
{25%, 35%, 50%, 75%, 100%} of real INT8 per-channel storage.
"""

from typing import List, Dict, Tuple, Any, Callable
from subbyte.quant.qpack import compute_qpack_size

def compute_allocation_storage(
    layer_metas: List[Dict[str, Any]],
    bits_allocation: List[int],
    header_overhead: int = 256
) -> int:
    """Computes real .qpack storage for a given bit allocation."""
    return compute_qpack_size(layer_metas, bits_allocation, header_overhead=header_overhead)

def get_int8_baseline_storage(
    layer_metas: List[Dict[str, Any]],
    header_overhead: int = 256
) -> int:
    """Computes exact INT8 per-channel reference storage in bytes."""
    return compute_allocation_storage(
        layer_metas,
        bits_allocation=[8] * len(layer_metas),
        header_overhead=header_overhead
    )

def solve_budget_bisection(
    score_fn: Callable[[float], List[int]],
    layer_metas: List[Dict[str, Any]],
    target_budget_bytes: int,
    delta_low: float = -1.0,
    delta_high: float = 1.0,
    max_iters: int = 50,
    tolerance: float = 0.01,
    header_overhead: int = 256
) -> Tuple[List[int], int, float]:
    """Finds threshold offset delta via monotone bisection to match target byte budget."""
    best_alloc = None
    best_bytes = float('inf')
    best_delta = 0.0
    
    for _ in range(max_iters):
        delta_mid = (delta_low + delta_high) / 2.0
        alloc_mid = score_fn(delta_mid)
        bytes_mid = compute_allocation_storage(layer_metas, alloc_mid, header_overhead)
        
        ratio = bytes_mid / float(target_budget_bytes)
        if abs(ratio - 1.0) < abs(best_bytes / float(target_budget_bytes) - 1.0):
            best_alloc = alloc_mid
            best_bytes = bytes_mid
            best_delta = delta_mid
            
        if abs(ratio - 1.0) <= tolerance:
            return alloc_mid, bytes_mid, delta_mid
            
        if bytes_mid > target_budget_bytes:
            delta_low = delta_mid
        else:
            delta_high = delta_mid
            
    return best_alloc, best_bytes, best_delta
