"""
Optimization Solvers for Mixed-Precision Quantization.
Implements:
1. Exact Dynamic Programming (DP) Knapsack Solver
2. HAWQ Hessian eigenvalue trace allocation
3. Sensitivity-Greedy allocation
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from subbyte.quant.budget import compute_allocation_storage

def solve_dp_knapsack(
    layer_sensitivities: List[Dict[int, float]],
    layer_costs: List[Dict[int, float]],
    cost_budget: float,
    allowed_bits: Optional[List[int]] = None,
    critical_indices: Optional[List[int]] = None,
    critical_bits: int = 8,
    cost_scale: float = 1.0
) -> Tuple[List[int], float, float]:
    """
    Exact DP solver for Bounded Multi-Choice Knapsack:
    Minimizes total predicted accuracy drop sum(delta_acc_l)
    subject to sum(cost_l) <= cost_budget.
    """
    if allowed_bits is None:
        allowed_bits = [2, 3, 4, 5, 6, 7, 8]
    if critical_indices is None:
        critical_indices = []
        
    num_layers = len(layer_sensitivities)
    max_budget_int = int(cost_budget * cost_scale)
    dp = {0: (0.0, [])}
    
    for l in range(num_layers):
        next_dp = {}
        sens_dict = layer_sensitivities[l]
        cost_dict = layer_costs[l]
        
        if l in critical_indices:
            choices = [critical_bits] if critical_bits in allowed_bits else [max(allowed_bits)]
        else:
            choices = [b for b in allowed_bits if b in sens_dict and b in cost_dict]
            
        for current_cost, (current_drop, current_alloc) in dp.items():
            for b in choices:
                drop = sens_dict.get(b, 0.0)
                cost_int = int(cost_dict.get(b, 0.0) * cost_scale)
                total_c = current_cost + cost_int
                
                if total_c <= max_budget_int:
                    total_d = current_drop + drop
                    if total_c not in next_dp or total_d < next_dp[total_c][0]:
                        next_dp[total_c] = (total_d, current_alloc + [b])
                        
        if not next_dp:
            min_c_prev = min(dp.keys())
            prev_drop, prev_alloc = dp[min_c_prev]
            min_c = min(int(cost_dict.get(b, 0.0) * cost_scale) for b in choices)
            best_b = min(choices)
            next_dp = {min_c_prev + min_c: (prev_drop + sens_dict.get(best_b, 0.0), prev_alloc + [best_b])}
            
        # Pruning: preserve Pareto frontier at each layer
        sorted_costs = sorted(next_dp.keys())
        pruned_dp = {}
        min_so_far = float('inf')
        for c in sorted_costs:
            d, alloc = next_dp[c]
            if d <= min_so_far + 1e-9:
                pruned_dp[c] = (d, alloc)
                min_so_far = d
        dp = pruned_dp
        
    best_c = max(dp.keys())
    best_d, best_alloc = dp[best_c]
    actual_cost = best_c / cost_scale
    
    return best_alloc, actual_cost, best_d

def solve_hawq_allocation(
    hessian_traces: List[float],
    layer_metas: List[Dict[str, Any]],
    target_budget_bytes: int,
    allowed_bits: Optional[List[int]] = None,
    critical_indices: Optional[List[int]] = None,
    critical_bits: int = 8
) -> List[int]:
    """
    HAWQ-style second-order allocation:
    Layers with larger Hessian trace Tr(H_l) receive higher bitwidth.
    """
    if allowed_bits is None:
        allowed_bits = [2, 3, 4, 5, 6, 7, 8]
    if critical_indices is None:
        critical_indices = []
        
    eps = 1e-12
    log_traces = [np.log(max(eps, t)) for t in hessian_traces]
    min_lt, max_lt = min(log_traces), max(log_traces)
    norm_traces = [(lt - min_lt) / (max_lt - min_lt + eps) for lt in log_traces]
    
    def trace_to_alloc(delta: float) -> List[int]:
        alloc = []
        for l, (nt, meta) in enumerate(zip(norm_traces, layer_metas)):
            if l in critical_indices:
                alloc.append(critical_bits)
            else:
                score = nt + delta
                if score > 0.85:
                    b = 8
                elif score > 0.72:
                    b = 6
                elif score > 0.55:
                    b = 4
                elif score > 0.35:
                    b = 3
                else:
                    b = 2
                b_valid = min(allowed_bits, key=lambda x: abs(x - b))
                alloc.append(b_valid)
        return alloc
        
    low, high = -2.0, 2.0
    best_alloc = trace_to_alloc(0.0)
    best_diff = float('inf')
    
    for _ in range(30):
        mid = (low + high) / 2.0
        alloc = trace_to_alloc(mid)
        cost = compute_allocation_storage(layer_metas, alloc)
        diff = abs(cost - target_budget_bytes)
        if diff < best_diff:
            best_diff = diff
            best_alloc = alloc
        if cost < target_budget_bytes:
            low = mid
        else:
            high = mid
            
    return best_alloc

def solve_sensitivity_greedy(
    layer_sensitivities: List[Dict[int, float]],
    layer_metas: List[Dict[str, Any]],
    target_budget_bytes: int,
    allowed_bits: Optional[List[int]] = None,
    critical_indices: Optional[List[int]] = None,
    critical_bits: int = 8
) -> List[int]:
    """
    Sensitivity-Greedy:
    Starts at minimal precision and iteratively promotes the layer with highest
    marginal accuracy benefit per byte until target budget is reached.
    """
    if allowed_bits is None:
        allowed_bits = [2, 3, 4, 5, 6, 7, 8]
    if critical_indices is None:
        critical_indices = []
        
    num_layers = len(layer_sensitivities)
    sorted_bits = sorted(allowed_bits)
    min_b = sorted_bits[0]
    
    alloc = [critical_bits if l in critical_indices else min_b for l in range(num_layers)]
    current_cost = compute_allocation_storage(layer_metas, alloc)
    
    while current_cost < target_budget_bytes:
        best_layer = None
        best_ratio = -1.0
        best_next_b = None
        
        for l in range(num_layers):
            if l in critical_indices:
                continue
            cur_b = alloc[l]
            idx = sorted_bits.index(cur_b)
            if idx + 1 < len(sorted_bits):
                next_b = sorted_bits[idx + 1]
                drop_cur = layer_sensitivities[l].get(cur_b, 0.0)
                drop_next = layer_sensitivities[l].get(next_b, 0.0)
                acc_gain = max(0.0, drop_cur - drop_next)
                
                temp_alloc = list(alloc)
                temp_alloc[l] = next_b
                new_cost = compute_allocation_storage(layer_metas, temp_alloc)
                cost_diff = new_cost - current_cost
                
                if cost_diff > 0 and (current_cost + cost_diff) <= target_budget_bytes * 1.01:
                    ratio = acc_gain / float(cost_diff)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_layer = l
                        best_next_b = next_b
                        
        if best_layer is None or best_ratio <= 0:
            break
            
        alloc[best_layer] = best_next_b
        current_cost = compute_allocation_storage(layer_metas, alloc)
        
    return alloc
