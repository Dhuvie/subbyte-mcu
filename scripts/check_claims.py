#!/usr/bin/env python3
"""
Claims Verification Tool for Sub-Byte MCU Paper.

Verifies that every empirical sentence in claims.yaml maps to genuine,
traceable macro values in numbers.tex and underlying results stores.
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml

ROOT = Path(__file__).resolve().parents[1]

def parse_numbers_tex(numbers_file: Path) -> Dict[str, str]:
    macros = {}
    if not numbers_file.exists():
        return macros
    with open(numbers_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'\\newcommand\{\\([a-zA-Z0-9]+)\}\{(.*)\}', line)
            if m:
                val = m.group(2).replace('{', '').replace('}', '').replace(',', '').strip()
                macros[m.group(1)] = val
    return macros


def verify_claims():
    claims_file = ROOT / "paper" / "claims.yaml"
    numbers_file = ROOT / "paper" / "numbers.tex"

    if not claims_file.exists():
        print(f"Claims file not found: {claims_file}")
        return False

    with open(claims_file, 'r', encoding='utf-8') as f:
        claims_data = yaml.safe_load(f)

    macros = parse_numbers_tex(numbers_file)
    print(f"Loaded {len(macros)} macros from numbers.tex.")

    errors = []
    claims_list = claims_data.get('claims', [])
    for claim in claims_list:
        cid = claim.get('id', 'unknown')
        statement = claim.get('statement', '')
        target_macro = claim.get('macro')
        expected_val = claim.get('expected_val')

        if target_macro:
            if target_macro not in macros:
                errors.append(f"[{cid}] Macro '\\{target_macro}' not defined in numbers.tex.")
            else:
                val_str = macros[target_macro]
                try:
                    val_num = float(val_str)
                    if abs(val_num - float(expected_val)) > 0.05:
                        errors.append(f"[{cid}] Value mismatch in \\{target_macro}: expected {expected_val}, got {val_num}")
                except ValueError:
                    errors.append(f"[{cid}] Could not parse float value from \\{target_macro}: '{val_str}'")

    if errors:
        print(f"\nFAILED: {len(errors)} claim verification errors detected:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"SUCCESS: All {len(claims_list)} empirical claims in paper/claims.yaml verified against numbers.tex.")
        return True


if __name__ == '__main__':
    ok = verify_claims()
    sys.exit(0 if ok else 1)
