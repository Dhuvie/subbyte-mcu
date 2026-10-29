"""
Quantization and Bit-Packing Package.
"""

from subbyte.quant.affine import compute_quant_params, quantize_affine, dequantize_affine
from subbyte.quant.bitpack import pack_bits, unpack_bits
from subbyte.quant.qpack import save_qpack, load_qpack, compute_qpack_size
from subbyte.quant.percentile import compute_percentile_range
from subbyte.quant.noise import apply_weight_noise
from subbyte.quant.fakequant import STEQuantize, FakeQuantizeWeight, FakeQuantizeActivation
from subbyte.quant.budget import compute_allocation_storage, get_int8_baseline_storage, solve_budget_bisection
