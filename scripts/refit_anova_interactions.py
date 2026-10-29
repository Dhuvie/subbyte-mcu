#!/usr/bin/env python3
"""
ANOVA Variance Decomposition with Interactions for Sub-Byte MCU Study (Phase 1, Task 1.3).

Refits the variance decomposition:
1. Full model with main effects and interactions:
   Factors:
   - Target Bit Budget (avg_bits / budget)
   - Endpoint Layer Protection (endpoint_prot)
   - Allocation Algorithm (method)
   - Method x Budget interaction (method:budget)
   - Endpoint x Budget interaction (endpoint:budget)
2. All budgets (N=120) and plateau regime budgets >= 0.50 (N=60).
3. Evaluated on CIFAR-10 / SeparableCNN and across all datasets.
4. Reports main-effect and interaction sums of squares, variance percentages, F-stats, and p-values.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

ROOT = Path(__file__).resolve().parents[1]
RUNS_FILE = ROOT / "results" / "runs.jsonl"
OUT_JSON = ROOT / "results" / "anova_interactions_h12.json"


def has_endpoint_prot(alloc):
    if not alloc or len(alloc) < 2:
        return 0.0
    return 1.0 if (alloc[0] == 8 and alloc[-1] == 8) else 0.0


def run_interaction_anova():
    records = []
    with open(RUNS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)

    df["endpoint_prot"] = df["bit_allocation"].apply(has_endpoint_prot)
    df["avg_bits"] = df["avg_bits_per_param"]
    df["budget"] = df["budget_target_frac"]

    ft = df[df["regime"] == "finetune"].copy()

    # 1. CIFAR-10 SeparableCNN specific decomposition (N=32 within CIFAR-10 or N=120 multi-task)
    # The paper's H12 was on N=120 fine-tuned runs
    print("=" * 80)
    print("ANOVA VARIANCE DECOMPOSITION WITH INTERACTIONS (FULL N=120 RUNS)")
    print("=" * 80)

    # We fit OLS with Type II ANOVA
    # Model: test_accuracy ~ C(dataset) + avg_bits + C(endpoint_prot) + C(method) + C(method):avg_bits + C(endpoint_prot):avg_bits
    formula = "test_accuracy ~ C(dataset) + avg_bits + C(endpoint_prot) + C(method) + C(method):avg_bits + C(endpoint_prot):avg_bits"
    model_all = ols(formula, data=ft).fit()
    anova_all = sm.stats.anova_lm(model_all, typ=2)

    # Within-dataset variance (removing dataset effect)
    ss_total = anova_all["sum_sq"].sum()
    ss_dataset = anova_all.loc["C(dataset)", "sum_sq"] if "C(dataset)" in anova_all.index else 0.0
    ss_within = ss_total - ss_dataset

    results_all = {}
    print(f"\nTotal Sum of Squares: {ss_total:.2f}, Within-Dataset SS: {ss_within:.2f}\n")
    print(f"{'Factor / Source':<35} | {'df':<4} | {'Sum Sq':<10} | {'Var % (Within)':<15} | {'F-stat':<8} | {'p-value'}")
    print("-" * 90)

    for term, row in anova_all.iterrows():
        if term == "C(dataset)":
            continue
        pct = (row["sum_sq"] / ss_within) * 100.0
        f_val = row["F"] if not np.isnan(row["F"]) else 0.0
        p_val = row["PR(>F)"] if not np.isnan(row["PR(>F)"]) else 1.0
        results_all[term] = {
            "df": int(row["df"]),
            "sum_sq": float(row["sum_sq"]),
            "pct_within": float(pct),
            "f_stat": float(f_val),
            "p_value": float(p_val),
        }
        print(f"{term:<35} | {int(row['df']):<4} | {row['sum_sq']:<10.2f} | {pct:<15.2f}% | {f_val:<8.2f} | {p_val:.4e}")

    # 2. Plateau Regime: Budgets >= 0.50 only (N=60)
    print("\n" + "=" * 80)
    ft_high = ft[ft["budget"] >= 0.50].copy()
    formula_high = "test_accuracy ~ C(dataset) + avg_bits + C(method) + C(method):avg_bits"
    model_high = ols(formula_high, data=ft_high).fit()
    anova_high = sm.stats.anova_lm(model_high, typ=2)

    ss_total_h = anova_high["sum_sq"].sum()
    ss_dataset_h = anova_high.loc["C(dataset)", "sum_sq"] if "C(dataset)" in anova_high.index else 0.0
    ss_within_h = ss_total_h - ss_dataset_h

    results_high = {}
    print(f"\nTotal Sum of Squares (>=50%): {ss_total_h:.2f}, Within-Dataset SS: {ss_within_h:.2f}\n")
    print(f"{'Factor / Source':<35} | {'df':<4} | {'Sum Sq':<10} | {'Var % (Within)':<15} | {'F-stat':<8} | {'p-value'}")
    print("-" * 90)

    for term, row in anova_high.iterrows():
        if term == "C(dataset)":
            continue
        pct = (row["sum_sq"] / ss_within_h) * 100.0
        f_val = row["F"] if not np.isnan(row["F"]) else 0.0
        p_val = row["PR(>F)"] if not np.isnan(row["PR(>F)"]) else 1.0
        results_high[term] = {
            "df": int(row["df"]),
            "sum_sq": float(row["sum_sq"]),
            "pct_within": float(pct),
            "f_stat": float(f_val),
            "p_value": float(p_val),
        }
        print(f"{term:<35} | {int(row['df']):<4} | {row['sum_sq']:<10.2f} | {pct:<15.2f}% | {f_val:<8.2f} | {p_val:.4e}")

    # Save to JSON
    output_data = {
        "all_budgets_n120": {
            "n_runs": len(ft),
            "formula": formula,
            "terms": results_all,
            "r2_score": float(model_all.rsquared),
        },
        "plateau_regime_n60": {
            "n_runs": len(ft_high),
            "formula": formula,
            "terms": results_high,
            "r2_score": float(model_high.rsquared),
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Saved] Interaction ANOVA results saved to {OUT_JSON}")
    return output_data


if __name__ == "__main__":
    run_interaction_anova()
