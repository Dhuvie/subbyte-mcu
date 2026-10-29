#!/usr/bin/env python3
"""Independent referee for subbyte-mcu.
Usage: python independent_check.py [path/to/result_store.jsonl]
Defaults to results/runs_multiseed_v3.jsonl.
Compare printed numbers with the abstract / Sec. VII / Table XI of the paper.
"""
import json, sys, collections, hashlib, statistics as st
from pathlib import Path

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def analyze_store(path_str, section_title):
    p = Path(path_str)
    if not p.exists():
        print(f"Store file not found: {p}")
        return
    file_hash = sha256_file(p)
    root = Path(__file__).resolve().parents[1]
    try:
        rel_p = p.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        rel_p = p.name
    print("=" * 80)
    print(f"{section_title}")
    print(f"File: {rel_p}  (SHA-256: {file_hash})")
    print("=" * 80)

    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    ft = [r for r in rows if r.get("regime") in ("finetune", "ft")]
    ptq = [r for r in rows if r.get("regime") == "ptq"]
    key = lambda r: (r["model"], r["budget_target_frac"], r["seed"])

    def table(rs, method):
        return {key(r): r["test_accuracy"] for r in rs if r["method"] == method}

    def diffs(rs, a, b):
        A, B = table(rs, a), table(rs, b)
        return {k: A[k] - B[k] for k in A if k in B}

    models = sorted({r["model"] for r in rows})
    budgets = sorted({r["budget_target_frac"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    print(f"rows: {len(rows)}  models: {models}  budgets: {budgets}  seeds: {seeds}")

    # Determine which comparison methods exist
    avail_methods = set(r["method"] for r in ft)
    compare_methods = []
    if "hawq" in avail_methods:
        compare_methods.append(("hawq", "HAWQ (legacy trace-rank heuristic)"))
    if "greedy" in avail_methods:
        compare_methods.append(("greedy", "Sensitivity Greedy"))
    if "hawq_omega_dp" in avail_methods:
        compare_methods.append(("hawq_omega_dp", "HAWQ-Omega (second-order Hessian trace + knapsack)"))

    for other, label in compare_methods:
        d = diffs(ft, "dp_measured", other)
        print(f"\n--- FINE-TUNED: DP minus {label} ---")
        tier = {}
        for b in budgets:
            per_model = []
            for m in models:
                v = [d[(m, b, s)] for s in seeds if (m, b, s) in d]
                if v: per_model.append(st.mean(v))
            tier[b] = per_model
            pos_models = sum(1 for x in per_model if x > 0)
            tot_models = len(per_model)
            pos_seeds = sum(1 for (m, bb, s), x in d.items() if bb == b and x > 0)
            tot_seeds = sum(1 for (m, bb, s) in d if bb == b)
            if per_model:
                print(f"budget {b:.2f}: per-model {[round(x,2) for x in per_model]}  "
                      f"mean {st.mean(per_model):+.2f}  median {st.median(per_model):+.2f}  "
                      f"positive models {pos_models}/{tot_models}  seed-level positive {pos_seeds}/{tot_seeds}")
        allm = [x for v in tier.values() for x in v]
        if allm:
            print(f"all tiers pooled per-model means: mean {st.mean(allm):+.2f} median {st.median(allm):+.2f}")

    print("\n--- PTQ accuracy by method and budget (mean over models and seeds) ---")
    for meth in sorted({r["method"] for r in ptq}):
        print(f"{meth:<22}", {b: round(st.mean([r["test_accuracy"] for r in ptq
              if r["method"] == meth and r["budget_target_frac"] == b]), 1) for b in budgets})

    print("\n--- BUDGET / SIZE CHECKS ---")
    bad = 0
    vec_size = collections.defaultdict(set)
    for r in rows:
        frac = r["real_size_kb"] / r["int8_baseline_kb"]
        if r["method"] != "uniform_w4_endpoints" and frac > r["budget_target_frac"] + 0.005:
            bad += 1
        vec_size[(r["model"], tuple(r["bit_allocation"]))].add(round(r["real_size_kb"], 1))
    print("rows above budget (non-uniform methods; floor cells expected):", bad)
    conflict = {k: v for k, v in vec_size.items() if len(v) > 1 and k[1]}
    print("same bit vector with different sizes (must be 0):", len(conflict))
    print("empty bit_allocation rows (must be 0):", sum(1 for r in rows if not r["bit_allocation"]))

    print("\n--- ACHIEVED BYTES (% of INT8), mean by method and budget ---")
    for meth in sorted({r["method"] for r in ft}):
        print(f"{meth:<22}", {b: round(100*st.mean([r["real_size_kb"]/r["int8_baseline_kb"] for r in ft
              if r["method"] == meth and r["budget_target_frac"] == b]), 1) for b in budgets})

def main():
    root = Path(__file__).resolve().parents[1]
    default_v3 = root / "results" / "runs_multiseed_v3.jsonl"
    frozen_v1 = root / "results" / "runs_multiseed.jsonl"

    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_v3

    # Section 1: Target / v3 Store
    sec1_title = "SECTION 1: PRIMARY MULTI-SEED V3 RESULT STORE (Benchmark Release)"
    if target_path.resolve() == frozen_v1.resolve():
        sec1_title = "SECTION 1: FROZEN V1 RESULT STORE"
    analyze_store(target_path, sec1_title)

    # Section 2: If primary was v3, also display frozen v1 store for historical comparison
    if target_path.resolve() != frozen_v1.resolve() and frozen_v1.exists():
        print("\n\n")
        analyze_store(frozen_v1, "SECTION 2: FROZEN V1 RESULT STORE (Historical Comparison, 800 Rows)")

if __name__ == "__main__":
    main()
