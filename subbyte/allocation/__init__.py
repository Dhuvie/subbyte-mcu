"""
Bit Allocation Algorithms and Optimization Solvers.
"""

from subbyte.allocation.formula_heuristic import (
    compute_layer_statistics,
    compute_position_factor,
    compute_size_factor,
    compute_importance_score,
    assign_bits_from_thresholds,
    compute_model_scores
)
from subbyte.allocation.baselines import (
    get_uniform_allocation,
    get_inverted_score_allocation,
    get_size_only_allocation,
    get_random_allocations
)
from subbyte.allocation.sensitivity import (
    compute_hutchinson_hessian_trace,
    measure_layer_sensitivity
)
from subbyte.allocation.knapsack import (
    solve_dp_knapsack,
    solve_hawq_allocation,
    solve_sensitivity_greedy
)
