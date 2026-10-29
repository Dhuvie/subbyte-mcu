#!/usr/bin/env python3
"""
Complete Experiment Matrix and Collapse Rate Analysis across all 5 Datasets.
Requirement 2.4:
- All five datasets: CIFAR-10 (SeparableCNN & ResNet-8), Fashion-MNIST, UCI HAR, Speech Commands.
- Budgets: {25, 35, 50, 75, 100}% of real INT8 .qpack bytes
- 5 seeds (42, 43, 44, 45, 46)
- Regimes: PTQ and Fine-Tuned
- Strategies: formula-score heuristic, size-only, random, inverted, sensitivity-greedy,
  HAWQ-style, exact DP, uniform INT8, uniform W4 + protected endpoints.
- Reports collapse rates (fraction of seeds below stated accuracy floor) separately from means.
- No cell empty: explicitly notes un-run combinations if applicable.
- Generates results/complete_matrix.json and paper/tables/tab_full_matrix.tex.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

import os
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RUNS_PATH = Path(os.environ.get("RUNS_FILE", str(RESULTS_DIR / "runs_multiseed_v3.jsonl" if (RESULTS_DIR / "runs_multiseed_v3.jsonl").exists() else (RESULTS_DIR / "runs_multiseed.jsonl" if (RESULTS_DIR / "runs_multiseed.jsonl").exists() else (RESULTS_DIR / "runs.jsonl")))))
OUT_JSON = RESULTS_DIR / "complete_matrix.json"
OUT_TEX = Path(os.environ.get("FULL_MATRIX_OUT", str(ROOT / "paper" / "tables" / "tab_full_matrix.tex")))

# Stated accuracy floors for catastrophic collapse (slightly above chance accuracy)
COLLAPSE_FLOORS = {
    'cifar10': 20.0,   # Chance = 10%
    'fmnist': 25.0,    # Chance = 10%
    'har': 30.0,       # Chance = 16.7%
    'speech': 20.0,    # Chance = 8.3%
}

STRATEGY_NAMES = [
    ('dp_measured', 'Measured-Sensitivity DP (Ours)'),
    ('hawq', 'HAWQ-Style Second-Order'),
    ('greedy', 'Sensitivity-Greedy'),
    ('uniform_w4_endpoints', 'Uniform W4 + Endpoints'),
    ('uniform_w8', 'Uniform INT8 Baseline'),
    ('formula_score', 'Formula-Score Heuristic'),
    ('size_only', 'Size-Only Heuristic'),
    ('random', 'Random Allocation'),
    ('inverted', 'Inverted Sensitivity (Adversarial)')
]

DATASET_CONFIGS = [
    ('cifar10', 'separable_cnn', 'CIFAR-10 / SeparableCNN'),
    ('cifar10', 'resnet8', 'CIFAR-10 / ResNet-8'),
    ('fmnist', 'fmnist_cnn', 'Fashion-MNIST / CNN'),
    ('har', 'cnn1d', 'UCI HAR / 1D-CNN'),
    ('speech', 'dscnn', 'Speech Commands / DS-CNN')
]

BUDGETS = [0.25, 0.35, 0.50, 0.75, 1.00]


def build_complete_matrix() -> Dict[str, Any]:
    records = []
    if RUNS_PATH.exists():
        with open(RUNS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    df = pd.DataFrame(records)
    
    matrix_output = []
    
    for ds, model, task_title in DATASET_CONFIGS:
        floor = COLLAPSE_FLOORS.get(ds, 20.0)
        task_data = {
            'dataset': ds,
            'model': model,
            'title': task_title,
            'collapse_floor': floor,
            'budgets': {}
        }
        
        for b in BUDGETS:
            b_pct = int(round(b * 100))
            task_data['budgets'][b_pct] = {}
            
            for strat_key, strat_label in STRATEGY_NAMES:
                for regime in ['ptq', 'finetune']:
                    # Allow matching both uniform_w4_endpoints and fallback uniform_w8_first_last_8
                    strat_query = [strat_key]
                    if strat_key == 'uniform_w4_endpoints':
                        strat_query.append('uniform_w8_first_last_8')
                    elif strat_key == 'uniform_w8_first_last_8':
                        strat_query.append('uniform_w4_endpoints')

                    sub = df[
                        (df['dataset'] == ds) &
                        (df['model'] == model) &
                        (df['method'].isin(strat_query)) &
                        (df['regime'] == regime) &
                        (np.isclose(df['budget_target_frac'], b, atol=0.03))
                    ]
                    
                    cell_key = f"{strat_key}_{regime}"
                    if len(sub) == 0:
                        # Handle known structural baseline conditions
                        if strat_key == 'uniform_w8' and b == 1.0:
                            # INT8 at 100%
                            sub_int8 = df[
                                (df['dataset'] == ds) &
                                (df['model'] == model) &
                                (df['method'].str.contains('uniform_w8')) &
                                (df['regime'] == regime)
                            ]
                            if len(sub_int8) > 0:
                                accs = sub_int8['test_accuracy'].tolist()
                                collapsed = sum(1 for a in accs if a < floor)
                                task_data['budgets'][b_pct][cell_key] = {
                                    'status': 'evaluated',
                                    'n_seeds': len(accs),
                                    'mean': round(float(np.mean(accs)), 2),
                                    'std': round(float(np.std(accs, ddof=1)), 2) if len(accs) > 1 else 0.0,
                                    'collapse_rate': f"{collapsed}/{len(accs)}",
                                    'collapse_pct': round((collapsed / len(accs)) * 100.0, 1)
                                }
                                continue

                        if strat_key == 'uniform_w8_first_last_8' and b == 0.50:
                            # Uniform W4+endpoints at 50%
                            # Look for uniform_w8_first_last_8 or W4 cells
                            if len(sub) > 0:
                                accs = sub['test_accuracy'].tolist()
                                collapsed = sum(1 for a in accs if a < floor)
                                task_data['budgets'][b_pct][cell_key] = {
                                    'status': 'evaluated',
                                    'n_seeds': len(accs),
                                    'mean': round(float(np.mean(accs)), 2),
                                    'std': round(float(np.std(accs, ddof=1)), 2) if len(accs) > 1 else 0.0,
                                    'collapse_rate': f"{collapsed}/{len(accs)}",
                                    'collapse_pct': round((collapsed / len(accs)) * 100.0, 1)
                                }
                                continue

                        task_data['budgets'][b_pct][cell_key] = {
                            'status': 'not_evaluated',
                            'reason': 'Ablation/fixed-budget constraint',
                            'n_seeds': 0,
                            'mean': None,
                            'std': None,
                            'collapse_rate': 'N/A'
                        }
                    else:
                        accs = sub['test_accuracy'].tolist()
                        n_tot = len(accs)
                        collapsed = sum(1 for a in accs if a < floor)
                        surviving = [a for a in accs if a >= floor]
                        mean_val = float(np.mean(accs))
                        std_val = float(np.std(accs, ddof=1)) if n_tot > 1 else 0.0
                        
                        task_data['budgets'][b_pct][cell_key] = {
                            'status': 'evaluated',
                            'n_seeds': n_tot,
                            'mean': round(mean_val, 2),
                            'std': round(std_val, 2),
                            'collapse_rate': f"{collapsed}/{n_tot}",
                            'collapse_pct': round((collapsed / n_tot) * 100.0, 1),
                            'mean_surviving': round(float(np.mean(surviving)), 2) if surviving else 0.0
                        }
                        
        matrix_output.append(task_data)
        
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(matrix_output, f, indent=2)
    print(f"Saved complete experiment matrix to {OUT_JSON}")
    
    generate_latex_full_matrix(matrix_output)
    return matrix_output


def generate_latex_full_matrix(matrix_data: List[Dict[str, Any]]):
    lines = [
        r"% Complete Multi-Task Quantization Matrix and Catastrophic Collapse Rates",
        r"% Generated automatically by scripts/generate_full_matrix.py",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Multi-domain quantization results across budgets, allocation algorithms, and regimes across 5 independently trained seeds (42--46). Values report mean test accuracy alongside catastrophic collapse rate (count of seeds falling below the task collapse floor). Cells marked [floor] or [infeas] denote budgets constrained by the 8-bit endpoint protection floor. Uniform W4+Endpoints is not budget-compliant at $\le 35\%$ budgets, occupying $\sim \MacroUniformWFourFootprintPct\%$ of INT8 bytes.}",
        r"\label{tab:full_matrix}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"\textbf{Benchmark / Method} & \multicolumn{2}{c}{\textbf{25\% Budget}} & \multicolumn{2}{c}{\textbf{35\% Budget}} & \multicolumn{2}{c}{\textbf{50\% Budget}} & \multicolumn{2}{c}{\textbf{75\% Budget}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r" & \textbf{PTQ (Coll.)} & \textbf{Fine-Tuned} & \textbf{PTQ (Coll.)} & \textbf{Fine-Tuned} & \textbf{PTQ (Coll.)} & \textbf{Fine-Tuned} & \textbf{PTQ (Coll.)} & \textbf{Fine-Tuned} \\",
        r"\midrule"
    ]
    
    # Minimum feasible size fraction with 8-bit endpoints
    FLOORS = {
        'cifar10 / separable_cnn': 25.4,
        'cifar10 / resnet8': 26.0,
        'fmnist / fmnist_cnn': 26.2,
        'har / cnn1d': 26.6,
        'speech / dscnn': 36.3
    }

    for task in matrix_data:
        title = task['title']
        t_key = f"{task['dataset']} / {task['model']}"
        min_floor = FLOORS.get(t_key, 25.0)
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{title}}} (Collapse Floor: {task['collapse_floor']:.0f}\\%, Endpoint Min Size: {min_floor:.1f}\\%)}}\\\\")
        
        for strat_key, strat_label in STRATEGY_NAMES[:4]: # Focus on primary methods
            row_cells = []
            for b in [25, 35, 50, 75]:
                b_data = task['budgets'].get(b, {})
                ptq = b_data.get(f"{strat_key}_ptq", {})
                ft = b_data.get(f"{strat_key}_finetune", {})
                
                # Check feasibility
                tag = ""
                if b < min_floor:
                    if strat_key in ['dp_measured', 'hawq', 'greedy']:
                        if b == 25 and min_floor > 30:
                            tag = " [infeas]"
                        elif b == 35 and min_floor > 35:
                            tag = " [infeas]"
                        else:
                            tag = " [floor]"
                if strat_key == 'uniform_w4_endpoints' and b < 50:
                    tag = r" [50\% B]"
                
                if ptq.get('status') == 'evaluated':
                    c_rate = ptq.get('collapse_rate', '0/5')
                    ptq_str = f"{ptq['mean']:.1f}\\% ({c_rate}){tag}"
                else:
                    ptq_str = "--"
                    
                if ft.get('status') == 'evaluated':
                    ft_str = f"{ft['mean']:.1f}\\%{tag}"
                else:
                    ft_str = "--"
                    
                row_cells.extend([ptq_str, ft_str])
                
            cells_joined = " & ".join(row_cells)
            lines.append(f"\\quad {strat_label} & {cells_joined} \\\\")
        lines.append(r"\midrule")
        
    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated LaTeX full matrix table at {OUT_TEX}")


if __name__ == "__main__":
    build_complete_matrix()
