import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

runs = [json.loads(line) for line in open('results/runs_multiseed.jsonl', encoding='utf-8') if line.strip()]
df = pd.DataFrame(runs)

# Filter for SeparableCNN fine-tuned runs with the 3 allocation methods
sep_ft = df[(df['model'] == 'separable_cnn') & (df['regime'] == 'finetune') & (df['method'].isin(['dp_measured', 'hawq', 'greedy']))].copy()
print(f"SeparableCNN FT runs count: {len(sep_ft)}")

# Fit 2-way ANOVA with interaction (Type II sum of squares)
model = ols('test_accuracy ~ C(method) + C(budget_target_frac) + C(method):C(budget_target_frac)', data=sep_ft).fit()
anova = sm.stats.anova_lm(model, typ=2)
ss_total = anova['sum_sq'].sum()

print("\n--- ALL BUDGETS (N=60) ---")
print(f"{'Source':<35} | {'df':<4} | {'Sum Sq':<10} | {'Var %':<8} | {'F-stat':<8} | {'p-value'}")
print("-" * 80)
for term, row in anova.iterrows():
    pct = (row['sum_sq'] / ss_total) * 100.0
    f_str = f"{row['F']:.2f}" if not np.isnan(row['F']) else "---"
    p_str = f"{row['PR(>F)']:.4e}" if not np.isnan(row['PR(>F)']) else "---"
    print(f"{term:<35} | {int(row['df']):<4} | {row['sum_sq']:<10.2f} | {pct:<7.2f}% | {f_str:<8} | {p_str}")

# Plateau regime (B >= 0.50)
sep_plateau = sep_ft[sep_ft['budget_target_frac'] >= 0.50].copy()
print(f"\nSeparableCNN Plateau FT runs count: {len(sep_plateau)}")
model_p = ols('test_accuracy ~ C(method) + C(budget_target_frac) + C(method):C(budget_target_frac)', data=sep_plateau).fit()
anova_p = sm.stats.anova_lm(model_p, typ=2)
ss_total_p = anova_p['sum_sq'].sum()

print("\n--- PLATEAU REGIME B >= 50% (N=30) ---")
print(f"{'Source':<35} | {'df':<4} | {'Sum Sq':<10} | {'Var %':<8} | {'F-stat':<8} | {'p-value'}")
print("-" * 80)
for term, row in anova_p.iterrows():
    pct = (row['sum_sq'] / ss_total_p) * 100.0
    f_str = f"{row['F']:.2f}" if not np.isnan(row['F']) else "---"
    p_str = f"{row['PR(>F)']:.4e}" if not np.isnan(row['PR(>F)']) else "---"
    print(f"{term:<35} | {int(row['df']):<4} | {row['sum_sq']:<10.2f} | {pct:<7.2f}% | {f_str:<8} | {p_str}")
