#!/usr/bin/env python3
"""
Instruction-Trace Hardware Cost Model for ARM Cortex-M4 (Thumb-2).

1. Parses freestanding Cortex-M4 Thumb-2 ELF object (kernels_m4.o).
2. Emulates execution of concrete C reference kernels on test vectors using Unicorn Engine.
3. Disassembles executed instructions via Capstone to extract exact instruction traces.
4. Verifies arithmetic output against double-precision NumPy references.
5. Computes modelled cycles per MAC by applying documented ARM Cortex-M4 TRM
   (ARM DDI 0439D Table 3-1) instruction timings, parameterized by Flash wait states,
   load latency, and branch penalties.
6. Exports timing table and hardware trace costs to results/ for Phase 2 validation.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    import unicorn
    from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_MEM_READ, UC_MEM_WRITE, UC_HOOK_CODE
    from unicorn.arm_const import (
        UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
        UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7,
        UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
        UC_ARM_REG_R12, UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC,
        UC_ARM_REG_CPSR
    )
except ImportError:
    Uc = None

try:
    import capstone
    from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB
except ImportError:
    Cs = None

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
except ImportError:
    ELFFile = None

ROOT = Path(__file__).resolve().parents[2]
ELF_OBJ = ROOT / "subbyte" / "cost_model" / "kernels_m4.o"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Documented Cortex-M4 TRM DDI 0439D Table 3-1 Instruction Timings
# Parameterized by wait states, load latency, and branch penalty
DEFAULT_TIMING_PARAMS = {
    "flash_wait_states": 0,    # 0 for zero-wait-state memory; up to 5 on high-clock MCUs
    "load_latency": 1,         # 1 cycle sequential/pipelined, 2 cycles non-pipelined
    "taken_branch_penalty": 2, # 1-3 cycles for pipeline refill on taken branch
    "untaken_branch_penalty": 1,
}


def load_elf_symbols(elf_path: Path) -> Tuple[bytes, int, Dict[str, Tuple[int, int]]]:
    """Parses ELF object and extracts .text section machine code and symbol offsets."""
    if not elf_path.exists():
        raise FileNotFoundError(f"Missing ELF object: {elf_path}")

    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        text_sec = elf.get_section_by_name(".text")
        if text_sec is None:
            raise ValueError("No .text section found in ELF object")
        code_bytes = text_sec.data()
        text_addr = text_sec['sh_addr']

        symbols = {}
        for sec in elf.iter_sections():
            if isinstance(sec, SymbolTableSection):
                for sym in sec.iter_symbols():
                    if sym['st_info']['type'] == 'STT_FUNC' and sym['st_size'] > 0:
                        symbols[sym.name] = (sym['st_value'], sym['st_size'])
    return code_bytes, text_addr, symbols


def classify_instruction(mnemonic: str) -> str:
    """Classifies Thumb-2 instruction into execution latency category (TRM DDI 0439D Table 3-1)."""
    m = mnemonic.lower()
    if m in ('ldr', 'ldrb', 'ldrh', 'ldrsb', 'ldrsh'):
        return 'load'
    elif m in ('str', 'strb', 'strh'):
        return 'store'
    elif m in ('ldm', 'ldmia', 'pop'):
        return 'load_multi'
    elif m in ('stm', 'stmia', 'push'):
        return 'store_multi'
    elif m in ('b', 'beq', 'bne', 'bcs', 'bcc', 'bmi', 'bpl', 'bvs', 'bvc', 'bhi', 'bls', 'bge', 'blt', 'bgt', 'ble', 'cbz', 'cbnz'):
        return 'branch'
    elif m in ('bl', 'blx', 'bx'):
        return 'branch_link'
    elif m in ('mul', 'mla', 'mls', 'smulbb', 'smultb', 'smulbt', 'smultt', 'smull', 'smlal', 'smlad', 'smlsd', 'smmla'):
        return 'mac'
    elif m in ('sbfx', 'ubfx', 'bfi', 'bfc'):
        return 'bitfield'
    elif m in ('sxtb', 'sxth', 'sxtb16', 'uxtb', 'uxth', 'uxtb16'):
        return 'extend'
    elif m in ('lsl', 'lsr', 'asr', 'ror', 'rrx'):
        return 'shift'
    else:
        return 'alu'


def calculate_instruction_cycles(category: str, params: Dict[str, int]) -> int:
    """Calculates cycle latency for an instruction category under timing parameters."""
    ws = params.get("flash_wait_states", 0)
    ll = params.get("load_latency", 1)
    bp = params.get("taken_branch_penalty", 2)

    if category == 'load':
        return 1 + ll + ws
    elif category == 'store':
        return 1 + ws
    elif category == 'branch':
        return bp
    elif category in ('mac', 'bitfield', 'extend', 'shift', 'alu'):
        return 1
    elif category in ('load_multi', 'store_multi'):
        return 2 + ll + ws
    else:
        return 1


def trace_kernel_execution(
    code_bytes: bytes,
    func_offset: int,
    func_size: int,
    packed_weights: bytes,
    activations: np.ndarray,
    n_macs: int,
    timing_params: Dict[str, int]
) -> Dict[str, Any]:
    """
    Executes a compiled Thumb-2 function on test vectors using Unicorn emulator.
    Records every executed instruction and computes cycle costs.
    """
    if Uc is None or Cs is None:
        raise RuntimeError("Unicorn and Capstone are required for instruction tracing")

    BASE_CODE = 0x08000000
    BASE_SRAM = 0x20000000
    ADDR_WEIGHTS = 0x20001000
    ADDR_ACTS = 0x20002000
    ADDR_STACK = 0x20008000
    TRAP_LR = 0x08FF0000

    # 1. Initialize Emulator
    mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
    cs = Cs(CS_ARCH_ARM, CS_MODE_THUMB)

    # 2. Map Memory (Code + SRAM)
    mu.mem_map(BASE_CODE, 0x00100000)  # 1 MB Flash
    mu.mem_map(BASE_SRAM, 0x00040000)  # 256 KB SRAM
    mu.mem_map(TRAP_LR & ~0xFFF, 0x1000) # Trap page for return

    # Write trap instruction (BKPT 0: 0xBE00) at TRAP_LR
    mu.mem_write(TRAP_LR, b"\x00\xbe")

    # 3. Write code to Flash
    mu.mem_write(BASE_CODE, code_bytes)

    # 4. Write data buffers to SRAM
    mu.mem_write(ADDR_WEIGHTS, packed_weights)
    act_bytes = activations.astype(np.int8).tobytes()
    mu.mem_write(ADDR_ACTS, act_bytes)

    # 5. Setup CPU Registers (AAPCS Calling Convention)
    start_pc = (BASE_CODE + func_offset) | 1  # Thumb mode requires bit 0 = 1
    mu.reg_write(UC_ARM_REG_PC, start_pc)
    mu.reg_write(UC_ARM_REG_SP, ADDR_STACK)
    mu.reg_write(UC_ARM_REG_LR, TRAP_LR | 1) # Return trap
    mu.reg_write(UC_ARM_REG_R0, ADDR_WEIGHTS)
    mu.reg_write(UC_ARM_REG_R1, ADDR_ACTS)
    mu.reg_write(UC_ARM_REG_R2, n_macs)

    # 6. Instruction Tracing Hook
    executed_instructions = []
    category_counts = {}

    def hook_code(uc, address, size, user_data):
        if address == TRAP_LR:
            uc.emu_stop()
            return
        instr_bytes = uc.mem_read(address, size)
        disasm = list(cs.disasm(instr_bytes, address))
        if disasm:
            ins = disasm[0]
            cat = classify_instruction(ins.mnemonic)
            executed_instructions.append((address, ins.mnemonic, ins.op_str, cat))
            category_counts[cat] = category_counts.get(cat, 0) + 1

    hook = mu.hook_add(UC_HOOK_CODE, hook_code)

    # 7. Execute Emulation
    try:
        mu.emu_start(start_pc, TRAP_LR, timeout=5000000) # 5s timeout
    except Exception as e:
        _ = e  # Emulation stopped at trap address

    mu.hook_del(hook)

    # 8. Read Result
    res_acc = mu.reg_read(UC_ARM_REG_R0)
    # Sign extend 32-bit integer
    if res_acc >= 0x80000000:
        res_acc -= 0x100000000

    # 9. Compute Cycle Latencies
    total_cycles = 0
    for addr, mnem, op_str, cat in executed_instructions:
        cyc = calculate_instruction_cycles(cat, timing_params)
        total_cycles += cyc

    cycles_per_mac = total_cycles / float(n_macs) if n_macs > 0 else 0.0

    return {
        "function_name": "",
        "return_val": int(res_acc),
        "total_instructions": len(executed_instructions),
        "category_counts": category_counts,
        "total_cycles": total_cycles,
        "cycles_per_mac": cycles_per_mac,
        "n_macs": n_macs
    }


def pack_weights_numpy(weights: np.ndarray, bitwidth: int) -> bytes:
    """Packs signed integer weights into tight bitstream bytes."""
    n = len(weights)
    if bitwidth == 8:
        return weights.astype(np.int8).tobytes()
    elif bitwidth == 4:
        packed = bytearray((n + 1) // 2)
        for i in range(n):
            val = int(weights[i]) & 0x0F
            if i % 2 == 0:
                packed[i // 2] |= val
            else:
                packed[i // 2] |= (val << 4)
        return bytes(packed)
    elif bitwidth == 2:
        packed = bytearray((n + 3) // 4)
        for i in range(n):
            val = int(weights[i]) & 0x03
            shift = (i % 4) * 2
            packed[i // 4] |= (val << shift)
        return bytes(packed)
    else:
        # Arbitrary bitwidths: 3, 5, 6, 7
        total_bits = n * bitwidth
        total_bytes = (total_bits + 7) // 8 + 4
        buf = bytearray(total_bytes)
        mask = (1 << bitwidth) - 1
        for i, w in enumerate(weights):
            raw_val = int(w) & mask
            bit_offset = i * bitwidth
            for b in range(bitwidth):
                bit = (raw_val >> b) & 1
                cur_bit = bit_offset + b
                byte_idx = cur_bit >> 3
                bit_idx = cur_bit & 7
                if bit:
                    buf[byte_idx] |= (1 << bit_idx)
        return bytes(buf)


def run_instruction_trace_suite(timing_params: Dict[str, int] = None) -> Dict[str, Any]:
    """Runs complete trace-based evaluation across all compiled C kernels."""
    if timing_params is None:
        timing_params = DEFAULT_TIMING_PARAMS.copy()

    code_bytes, text_addr, symbols = load_elf_symbols(ELF_OBJ)
    print(f"Loaded {len(symbols)} functions from ELF {ELF_OBJ.name} (.text size: {len(code_bytes)} bytes)")

    test_cases = [
        ("dot_product_int8", 8, "scalar"),
        ("dot_product_int8_opt", 8, "simd_opt"),
        ("dot_product_int4_naive", 4, "scalar"),
        ("dot_product_int4_opt", 4, "simd_opt"),
        ("dot_product_int2_naive", 2, "scalar"),
        ("dot_product_int2_opt", 2, "simd_opt"),
        ("dot_product_int3", 3, "scalar"),
        ("dot_product_int3_opt", 3, "simd_opt"),
        ("dot_product_int5", 5, "scalar"),
        ("dot_product_int6", 6, "scalar"),
        ("dot_product_int7", 7, "scalar"),
    ]

    np.random.seed(42)
    N_MACS = 128
    results = {}

    print("\n" + "="*85)
    print(f"{'Kernel Function':<24} | {'Width':<5} | {'Path':<8} | {'Cycles/MAC':<11} | {'Instr/MAC':<10} | {'Status'}")
    print("="*85)

    base_int8_cyc = None

    for func_name, width, path_type in test_cases:
        if func_name not in symbols:
            print(f"Warning: Symbol {func_name} not found in ELF object")
            continue

        offset, size = symbols[func_name]

        # Generate test vectors
        min_val = -(2**(width-1))
        max_val = 2**(width-1) - 1
        weights = np.random.randint(min_val, max_val + 1, size=N_MACS).astype(np.int32)
        acts = np.random.randint(-128, 127, size=N_MACS).astype(np.int32)
        expected_ref = int(np.sum(weights * acts))

        packed_weights = pack_weights_numpy(weights, width)

        trace_res = trace_kernel_execution(
            code_bytes, offset, size, packed_weights, acts, N_MACS, timing_params
        )
        trace_res["function_name"] = func_name
        trace_res["bitwidth"] = width
        trace_res["path_type"] = path_type

        # Verify against NumPy reference
        match = (trace_res["return_val"] == expected_ref)
        status = "VERIFIED" if match else "MISMATCH"

        instr_per_mac = trace_res["total_instructions"] / float(N_MACS)
        print(f"{func_name:<24} | {width:<5} | {path_type:<8} | {trace_res['cycles_per_mac']:<11.2f} | {instr_per_mac:<10.2f} | {status}")

        if func_name == "dot_product_int8":
            base_int8_cyc = trace_res["cycles_per_mac"]

        results[func_name] = trace_res

    # Compute penalty ratios relative to baseline INT8
    print("="*85 + "\n")
    for func_name, r in results.items():
        if base_int8_cyc and base_int8_cyc > 0:
            r["penalty_vs_int8"] = r["cycles_per_mac"] / base_int8_cyc

    out_file = RESULTS_DIR / "hardware_trace_costs.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved instruction trace cost results to {out_file}")

    # Also save timing parameter definitions
    timing_file = RESULTS_DIR / "timing_table.json"
    with open(timing_file, "w", encoding="utf-8") as f:
        json.dump({
            "trm_reference": "ARM Cortex-M4 Technical Reference Manual (TRM DDI 0439D Table 3-1)",
            "timing_parameters": timing_params,
            "instruction_categories": {
                "alu": "1 cycle",
                "mac": "1 cycle (MUL, MLA, SMLAD, SMULBB)",
                "load": "1 + load_latency + flash_wait_states cycles (LDR, LDRB, LDRH)",
                "store": "1 + flash_wait_states cycles (STR, STRB, STRH)",
                "branch_taken": "taken_branch_penalty cycles",
                "bitfield": "1 cycle (SBFX, UBFX)",
                "extend": "1 cycle (SXTB, SXTB16)"
            }
        }, f, indent=2)
    print(f"Saved timing table reference to {timing_file}")

    return results


if __name__ == "__main__":
    run_instruction_trace_suite()
