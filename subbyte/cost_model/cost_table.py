"""
ARM Cortex-M4 Hardware Instruction Cost Model.
Provides documented cycle penalties and hardware metrics grounded in ARM DDI 0439D Table 3-1.
"""

from typing import Dict, Any

# Instruction-level cycle latencies per MAC operation
# Derived from Cortex-M4 Technical Reference Manual (TRM DDI 0439D Table 3-1)
CYCLES_PER_MAC = {
    # INT8 variants
    (8, 'baseline'): 2.00, # 8 cycles for 4 MACs (LDR + SXTB16 + SMLAD + loop overhead)
    (8, 'simd_opt'): 1.50, # 12 cycles for 8 MACs (pipelined dual-LDR + unrolled SMLAD)
    # INT4 variants
    (4, 'scalar'):   2.88, # 23 cycles for 8 MACs (LDR + shifts + AND + SBFX + SMULBB)
    (4, 'simd_opt'): 1.75, # 14 cycles for 8 MACs (LDR + BFX + SXTB16 + SMLAD)
    # INT2 variants
    (2, 'scalar'):   2.69, # 43 cycles for 16 MACs
    (2, 'simd_opt'): 1.56, # 25 cycles for 16 MACs
    # Unaligned odd and non-power-of-two bitwidths
    (3, 'scalar'):   4.00, # 32 cycles for 8 MACs (2.00x vs INT8 baseline)
    (3, 'simd_opt'): 3.00, # 24 cycles for 8 MACs (1.50x vs INT8 baseline, 24-bit aligned)
    (5, 'scalar'):   4.25, # 34 cycles for 8 MACs (2.12x vs INT8 baseline)
    (5, 'simd_opt'): 4.25, # No SIMD kernel exists; falls back to scalar (2.12x)
    (6, 'scalar'):   4.25, # 34 cycles for 8 MACs (2.12x vs INT8 baseline)
    (6, 'simd_opt'): 4.25, # No SIMD kernel exists; falls back to scalar (2.12x)
    (7, 'scalar'):   4.50, # 36 cycles for 8 MACs (2.25x vs INT8 baseline)
    (7, 'simd_opt'): 4.50, # No SIMD kernel exists; falls back to scalar (2.25x)
}

def get_unpack_penalty(bits: int, path: str = 'simd_opt') -> float:
    """Returns latency ratio relative to baseline INT8 (2.00 cyc/MAC)."""
    base_int8 = CYCLES_PER_MAC[(8, 'baseline')]
    if (bits, path) in CYCLES_PER_MAC:
        cyc = CYCLES_PER_MAC[(bits, path)]
    elif (bits, 'scalar') in CYCLES_PER_MAC:
        cyc = CYCLES_PER_MAC[(bits, 'scalar')]
    else:
        cyc = 4.25
    return cyc / base_int8
