#!/usr/bin/env python3
"""
Regenerates and verifies the Instruction Cycle Summary table in README.md
from results/hardware_trace_costs.json.
Ensures every measured cycle count and instruction count matches the paper's macros
and results file, with analytical model cycle counts labeled in a separate column.
"""

import sys
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "results" / "hardware_trace_costs.json"
README_FILE = ROOT / "README.md"

# Table schema definition mapping results key to presentation metadata
# Format: (trace_key, name, path_label, model_cyc_str, penalty_str, flash_savings_str)
ROWS_METADATA = [
    ("dot_product_int8_opt", "INT8 Baseline (CMSIS-NN Style)", "SIMD", "1.50-2.00", "1.00x (SIMD Baseline)", "0.0%"),
    ("dot_product_int8", "INT8 Scalar Baseline", "Scalar", "2.00", "1.00x (Scalar Baseline)", "0.0%"),
    ("dot_product_int4_opt", "INT4 Optimised SIMD (SXTB16)", "SIMD", "1.75", "1.56x (vs INT8 SIMD)", "50.0%"),
    ("dot_product_int4_naive", "INT4 Naive Scalar (SBFX)", "Scalar", "2.88", "1.16x (vs INT8 Scalar)", "50.0%"),
    ("dot_product_int2_opt", "INT2 Optimised SIMD (SXTB16)", "SIMD", "1.56", "1.33x (vs INT8 SIMD)", "75.0%"),
    ("dot_product_int2_naive", "INT2 Naive Scalar (SBFX)", "Scalar", "2.69", "0.98x (vs INT8 Scalar)", "75.0%"),
    ("dot_product_int3_opt", "INT3 Aligned Unpack + SMLAD", "SIMD", "3.00", "2.12x (vs INT8 SIMD)", "62.5%"),
    ("dot_product_int3", "INT3 Scalar Bitfield Extract", "Scalar", "4.00", "3.15x (vs INT8 Scalar)", "62.5%"),
    ("dot_product_int5", "INT5 Sub-Byte Unaligned Scalar", "Scalar", "4.25", "3.41x (vs INT8 Scalar)", "37.5%"),
    ("dot_product_int6", "INT6 Semi-Aligned Scalar", "Scalar", "4.25", "3.24x (vs INT8 Scalar)", "25.0%"),
    ("dot_product_int7", "INT7 Sub-Byte Unaligned Scalar", "Scalar", "4.50", "3.41x (vs INT8 Scalar)", "12.5%"),
]

def build_markdown_table(trace_data: dict) -> str:
    lines = [
        "### 3. Instruction Cycle Summary (Measured Traces & Analytical Model)",
        "",
        "Instruction-level execution costs grounded in measured Unicorn CPU emulator traces (`results/hardware_trace_costs.json`) and analytical cycle counts from ARM Cortex-M4 TRM DDI 0439D Table 3-1:",
        "",
        "| Kernel Variant | Path | Trace Inst / 128 MAC | Trace Cyc / MAC | Model Cyc / MAC (ARM TRM Table 3-1) | Penalty vs INT8 (Like-for-Like) | Flash Savings |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for key, name, path_label, model_cyc, penalty, flash in ROWS_METADATA:
        if key not in trace_data:
            raise KeyError(f"Key {key} not found in {RESULTS_FILE}")
        entry = trace_data[key]
        inst = entry["total_instructions"]
        cyc = f"{entry['cycles_per_mac']:.2f}"
        lines.append(f"| **{name}** | {path_label} | {inst} | **{cyc}** | {model_cyc} | {penalty} | {flash} |")
    return "\n".join(lines)

def update_readme() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_FILE}")
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    new_table_str = build_markdown_table(trace_data)

    content = README_FILE.read_text(encoding="utf-8")
    pattern = re.compile(
        r"### 3\. Instruction Cycle Summary.*?(?=\n---\n)",
        re.DOTALL
    )

    if not pattern.search(content):
        raise ValueError("Could not find '### 3. Instruction Cycle Summary' section in README.md")

    updated_content = pattern.sub(new_table_str + "\n", content)
    README_FILE.write_text(updated_content, encoding="utf-8")
    print(f"Successfully updated README cycle table from {RESULTS_FILE.name}.")

def verify_readme_cycles(trace_data: dict, readme_text: str) -> list[str]:
    """Returns list of error messages if README table drifts from hardware_trace_costs.json."""
    errors = []
    for key, name, path_label, model_cyc, penalty, flash in ROWS_METADATA:
        if key not in trace_data:
            errors.append(f"Trace key {key} missing from hardware_trace_costs.json")
            continue
        expected_inst = trace_data[key]["total_instructions"]
        expected_cyc = f"{trace_data[key]['cycles_per_mac']:.2f}"

        # Match row for this kernel variant
        # e.g.: | **INT4 Optimised SIMD (SXTB16)** | SIMD | 608 | **5.28** | ...
        row_pat = re.compile(
            r"\|\s*\*\*" + re.escape(name) + r"\*\*\s*\|\s*" +
            re.escape(path_label) + r"\s*\|\s*(\d+)\s*\|\s*\*\*([0-9.]+)\*\*",
            re.IGNORECASE
        )
        match = row_pat.search(readme_text)
        if not match:
            errors.append(f"Row for '{name}' not found in README.md table")
            continue
        parsed_inst = int(match.group(1))
        parsed_cyc = match.group(2)
        if parsed_inst != expected_inst:
            errors.append(f"Instruction mismatch for {name}: README has {parsed_inst}, expected {expected_inst}")
        if parsed_cyc != expected_cyc:
            errors.append(f"Cycle count mismatch for {name}: README has {parsed_cyc}, expected {expected_cyc}")

    return errors

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            traces = json.load(f)
        readme_txt = README_FILE.read_text(encoding="utf-8")
        errs = verify_readme_cycles(traces, readme_txt)
        if errs:
            print("[FAIL] Cycle table verification failed:")
            for e in errs:
                print("  -", e)
            sys.exit(1)
        else:
            print("[OK] README cycle table matches hardware_trace_costs.json.")
    else:
        update_readme()
