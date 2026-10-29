#!/usr/bin/env python3
"""
Multi-Seed Experimental Matrix Execution Suite.

Executes ground-up multi-seed evaluation across:
- 5 models / datasets: CIFAR-10 SeparableCNN, CIFAR-10 ResNet-8, Fashion-MNIST CNN, Speech Commands DSCNN, HAR 1D-CNN
- 5 independent random seeds: 42, 43, 44, 45, 46
- 4 compression budget tiers: 25%, 35%, 50%, 75%
- 4 allocation methods: DP Knapsack, HAWQ, Sensitivity-Greedy, Uniform W4+Endpoints
- 2 operational regimes: PTQ (zero fine-tuning) and FT (5 epochs fine-tuning)

Produces:
- results/runs_multiseed.jsonl: Complete raw runs database
- results/multiseed_summary.json: Aggregated statistics (mean, sd, 95% CI) per cell
- results/model_level_tests.json: Honest n=5 permutation tests with model as unit
- results/ablations_seed_level.json: Sensitivity sample size & low-budget endpoint protection ablations
- results/full_row_retrain.json: From-scratch retrain of full row (Seed 99)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subbyte.datasets.cifar import get_cifar10_loaders
from subbyte.datasets.fmnist import get_fmnist_loaders
from subbyte.datasets.speech import get_speech_loaders
from subbyte.datasets.har import get_har_loaders

from subbyte.models.separable_cnn import SeparableCNN
from subbyte.models.resnet import ResNet8
from subbyte.models.fmnist_cnn import FashionCNN
from subbyte.models.dscnn import DSCNNS
from subbyte.models.cnn1d import CNN1D

from subbyte.training.train import train_fp32_model, finetune_quantized_model
from subbyte.training.evaluate import evaluate_loader
from subbyte.allocation.knapsack import solve_dp_knapsack, solve_hawq_allocation
from subbyte.allocation.sensitivity import measure_layer_sensitivity, compute_hutchinson_hessian_trace
from subbyte.quant.affine import quantize_affine, dequantize_affine, compute_quant_params

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_RUNS = RESULTS_DIR / "runs_multiseed.jsonl"
OUT_SUMMARY = RESULTS_DIR / "multiseed_summary.json"
OUT_TESTS = RESULTS_DIR / "model_level_tests.json"
OUT_ABLATIONS = RESULTS_DIR / "ablations_seed_level.json"
OUT_RETRAIN = RESULTS_DIR / "full_row_retrain.json"

TASKS = [
    {
        'id': 'har_cnn1d', 'dataset': 'har', 'model': 'cnn1d',
        'model_cls': CNN1D, 'model_kwargs': {},
        'loader_fn': get_har_loaders, 'batch_size': 64,
        'epochs': 25, 'lr': 0.01, 'is_cifar': False,
        # Endpoint protocol: Dynamically derived in run_matrix as [0, L-1] with L layers
        'critical_indices': None, 'num_layers': None
    },
    {
        'id': 'speech_dscnn', 'dataset': 'speech', 'model': 'dscnn',
        'model_cls': DSCNNS, 'model_kwargs': {},
        'loader_fn': get_speech_loaders, 'batch_size': 128,
        'epochs': 25, 'lr': 0.01, 'is_cifar': False,
        # Historically 'critical_indices': [0, 9], 'num_layers': 10
        'critical_indices': None, 'num_layers': None
    },
    {
        'id': 'fmnist_cnn', 'dataset': 'fmnist', 'model': 'fmnist_cnn',
        'model_cls': FashionCNN, 'model_kwargs': {},
        'loader_fn': get_fmnist_loaders, 'batch_size': 128,
        'epochs': 20, 'lr': 0.02, 'is_cifar': False,
        # Historically 'critical_indices': [0, 4], 'num_layers': 5
        'critical_indices': None, 'num_layers': None
    },
    {
        'id': 'cifar10_resnet8', 'dataset': 'cifar10', 'model': 'resnet8',
        'model_cls': ResNet8, 'model_kwargs': {'num_classes': 10},
        'loader_fn': get_cifar10_loaders, 'batch_size': 128,
        'epochs': 25, 'lr': 0.05, 'is_cifar': True,
        # Historically 'critical_indices': [0, 9], 'num_layers': 10
        'critical_indices': None, 'num_layers': None
    },
    {
        'id': 'cifar10_separable_cnn', 'dataset': 'cifar10', 'model': 'separable_cnn',
        'model_cls': SeparableCNN, 'model_kwargs': {'num_classes': 10},
        'loader_fn': get_cifar10_loaders, 'batch_size': 128,
        'epochs': 25, 'lr': 0.05, 'is_cifar': True,
        # Historically 'critical_indices': [0, 23], 'num_layers': 24
        'critical_indices': None, 'num_layers': None
    }
]

BUDGETS = [0.25, 0.35, 0.50, 0.75]
SEEDS = [42, 43, 44, 45, 46]


def get_layer_weights(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    quantizable = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            quantizable.append((name, mod))
    return quantizable


def apply_layer_quantization(model: nn.Module, alloc: List[int]) -> nn.Module:
    layers = get_layer_weights(model)
    assert len(layers) == len(alloc), f"Mismatch: {len(layers)} layers vs {len(alloc)} bits"
    with torch.no_grad():
        for (name, mod), b in zip(layers, alloc):
            w = mod.weight.data
            scale, zero_pt = compute_quant_params(w.min(), w.max(), bits=b, symmetric=True)
            q = quantize_affine(w, scale, zero_pt, bits=b)
            deq = dequantize_affine(q, scale, zero_pt)
            mod.weight.data.copy_(deq)
    return model


def compute_layer_costs(model: nn.Module, bits_list: List[int]) -> List[Dict[int, float]]:
    layers = get_layer_weights(model)
    costs = []
    for name, mod in layers:
        num_params = mod.weight.numel()
        c_dict = {}
        for b in bits_list:
            c_dict[b] = float((num_params * b + 7) // 8)
        costs.append(c_dict)
    return costs


def compute_total_int8_cost(model: nn.Module) -> float:
    layers = get_layer_weights(model)
    return sum(float(mod.weight.numel()) for _, mod in layers)


def solve_greedy_allocation(
    sensitivities: List[Dict[int, float]],
    costs: List[Dict[int, float]],
    budget_bytes: float,
    critical_indices: List[int]
) -> List[int]:
    n_layers = len(sensitivities)
    alloc = [8 if i in critical_indices else 2 for i in range(n_layers)]
    cur_cost = sum(costs[i][alloc[i]] for i in range(n_layers))
    
    # Greedily upgrade layers with highest accuracy gain per byte
    improved = True
    while improved:
        improved = False
        best_gain_per_byte = -1.0
        best_layer = -1
        best_next_b = -1
        
        for i in range(n_layers):
            if i in critical_indices:
                continue
            cur_b = alloc[i]
            if cur_b < 8:
                next_b = cur_b + 1
                cost_diff = costs[i][next_b] - costs[i][cur_b]
                if cur_cost + cost_diff <= budget_bytes and cost_diff > 0:
                    # Drop reduction is positive gain
                    gain = sensitivities[i].get(cur_b, 0.0) - sensitivities[i].get(next_b, 0.0)
                    gain_per_byte = gain / cost_diff
                    if gain_per_byte > best_gain_per_byte:
                        best_gain_per_byte = gain_per_byte
                        best_layer = i
                        best_next_b = next_b
                        
        if best_layer != -1 and best_gain_per_byte >= 0:
            cur_cost += (costs[best_layer][best_next_b] - costs[best_layer][alloc[best_layer]])
            alloc[best_layer] = best_next_b
            improved = True
            
    return alloc


def validate_allocation_endpoints(alloc: List[int], num_layers: int, method_name: str) -> List[int]:
    """
    Endpoint Validation Assertion:
    Raises if alloc length != L or if b[0] != 8 or b[-1] != 8 for endpoint-protected methods.
    """
    assert len(alloc) == num_layers, f"Allocation length mismatch for {method_name}: expected {num_layers}, got {len(alloc)}"
    if method_name in ('dp_measured', 'greedy', 'hawq', 'uniform_w4_endpoints'):
        assert alloc[0] == 8, f"Endpoint violation in {method_name}: b[0] = {alloc[0]} != 8"
        assert alloc[-1] == 8, f"Endpoint violation in {method_name}: b[-1] = {alloc[-1]} != 8"
    return alloc


def run_matrix():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Multi-Seed Evaluation Matrix on {device} ===")
    
    fp32_results = []
    runs_records = []
    
    if OUT_RUNS.exists():
        with open(OUT_RUNS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    runs_records.append(json.loads(line))
        print(f"Loaded {len(runs_records)} existing runs from {OUT_RUNS}")
        
    fp32_f = RESULTS_DIR / "fp32_results.json"
    if fp32_f.exists():
        with open(fp32_f, "r", encoding="utf-8") as f:
            fp32_results = json.load(f)
        print(f"Loaded {len(fp32_results)} existing FP32 baseline records from {fp32_f}")
    
    start_total_time = time.time()
    
    for task in TASKS:
        task_id = task['id']
        ds_name = task['dataset']
        model_name = task['model']
        model_cls = task['model_cls']
        loader_fn = task['loader_fn']
        b_size = task['batch_size']
        crit_idx = task['critical_indices']
        
        print(f"\n=======================================================")
        print(f"Processing Task: {task_id} ({ds_name} / {model_name})")
        print(f"=======================================================")
        
        task_seed_runs = []
        
        for seed in SEEDS:
            existing_for_seed = [r for r in runs_records if r['dataset'] == ds_name and r['model'] == model_name and r['seed'] == seed]
            if len(existing_for_seed) >= 32:
                print(f"Skipping already completed task {task_id} seed {seed} ({len(existing_for_seed)} runs)")
                continue
                
            print(f"\n--- [Task: {task_id}] Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            loaders = loader_fn(batch_size=b_size, seed=seed)
            train_loader = loaders['train_loader']
            val_loader = loaders['val_loader']
            test_loader = loaders['test_loader']
            
            # Checkpoint path per seed
            ckpt_path = CHECKPOINTS_DIR / f"fp32_{ds_name}_{model_name}_seed{seed}.pt"
            model_fp32 = model_cls(**task['model_kwargs']).to(device)
            
            if ckpt_path.exists():
                print(f"Loading cached FP32 checkpoint: {ckpt_path.name}")
                ckpt = torch.load(ckpt_path, map_location=device)
                if isinstance(ckpt, dict):
                    if 'state_dict' in ckpt:
                        sd = ckpt['state_dict']
                    elif 'model_state_dict' in ckpt:
                        sd = ckpt['model_state_dict']
                    else:
                        sd = ckpt
                else:
                    sd = ckpt
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                model_fp32.load_state_dict(sd)
                train_time = ckpt.get('training_time_s', 0.0) if isinstance(ckpt, dict) else 0.0
            else:
                print(f"Training FP32 baseline ({task['epochs']} epochs, lr={task['lr']})...")
                res = train_fp32_model(
                    model_fp32, train_loader, val_loader, device=device,
                    epochs=task['epochs'], lr=task['lr'], is_cifar=task['is_cifar']
                )
                train_time = res['training_time_s']
                torch.save({'state_dict': res['state_dict'], 'training_time_s': train_time}, ckpt_path)
                model_fp32.load_state_dict(res['state_dict'])
                
            fp32_test_acc = evaluate_loader(model_fp32, test_loader, device=device)
            print(f"FP32 Test Accuracy: {fp32_test_acc:.2f}% (train time: {train_time:.1f}s)")
            
            fp32_record = {
                'dataset': ds_name, 'model': model_name, 'seed': seed,
                'fp32_test_acc': fp32_test_acc, 'train_time_s': train_time
            }
            fp32_results.append(fp32_record)
            
            # 1. Compute layer costs and total INT8 size
            bits_list = [2, 3, 4, 5, 6, 7, 8]
            layer_costs = compute_layer_costs(model_fp32, bits_list)
            total_int8_bytes = compute_total_int8_cost(model_fp32)
            
            # 2. Empirical sensitivity profiling on validation split
            print("Profiling layer sensitivities on validation split...")
            layers = get_layer_weights(model_fp32)
            # Dynamically derive critical indices as [0, L-1], never a hardcoded list
            crit_idx = [0, len(layers) - 1]
            sens_table = []
            fp32_val_acc = evaluate_loader(model_fp32, val_loader, device=device)
            
            with torch.no_grad():
                for name, mod in layers:
                    w_orig = mod.weight.data.clone()
                    s_dict = {}
                    for b in bits_list:
                        scale, zp = compute_quant_params(w_orig.min(), w_orig.max(), bits=b, symmetric=True)
                        q = quantize_affine(w_orig, scale, zp, bits=b)
                        deq = dequantize_affine(q, scale, zp)
                        mod.weight.data.copy_(deq)
                        v_acc = evaluate_loader(model_fp32, val_loader, device=device)
                        s_dict[b] = max(0.0, fp32_val_acc - v_acc)
                    mod.weight.data.copy_(w_orig)
                    sens_table.append(s_dict)
                    
            # 3. Hessian traces for HAWQ
            print("Estimating Hessian traces via Hutchinson estimator...")
            criterion = nn.CrossEntropyLoss()
            traces_dict = compute_hutchinson_hessian_trace(
                model_fp32, val_loader, criterion, device=device, num_samples=100
            )
            hessian_traces = [traces_dict.get(f"{name}.weight", 1.0) for name, _ in layers]
            
            # 4. Evaluate across 4 budgets
            for b_frac in BUDGETS:
                target_budget_bytes = total_int8_bytes * b_frac
                
                # A. DP Knapsack Allocation
                alloc_dp, cost_dp, _ = solve_dp_knapsack(
                    sens_table, layer_costs, cost_budget=target_budget_bytes,
                    allowed_bits=bits_list, critical_indices=crit_idx, critical_bits=8
                )
                
                # B. HAWQ Allocation
                # Sort layers by trace: higher trace -> higher bitwidth
                log_tr = [np.log(max(1e-12, t)) for t in hessian_traces]
                min_t, max_t = min(log_tr), max(log_tr)
                norm_tr = [(t - min_t) / (max_t - min_t + 1e-12) for t in log_tr]
                alloc_hawq = []
                for i, nt in enumerate(norm_tr):
                    if i in crit_idx:
                        alloc_hawq.append(8)
                    else:
                        # map norm trace to bit range [2, 8] scaled by budget
                        b_val = int(2 + round(nt * (6.0 * b_frac / 0.75)))
                        b_val = max(2, min(8, b_val))
                        alloc_hawq.append(b_val)
                # Budget adjust HAWQ
                cur_h_cost = sum(layer_costs[i][alloc_hawq[i]] for i in range(len(alloc_hawq)))
                while cur_h_cost > target_budget_bytes:
                    reducible = [i for i in range(len(alloc_hawq)) if i not in crit_idx and alloc_hawq[i] > 2]
                    if not reducible:
                        break
                    # pick layer with lowest trace to reduce
                    best_i = min(reducible, key=lambda idx: norm_tr[idx])
                    alloc_hawq[best_i] -= 1
                    cur_h_cost = sum(layer_costs[i][alloc_hawq[i]] for i in range(len(alloc_hawq)))
                    
                # C. Sensitivity-Greedy Allocation
                alloc_greedy = solve_greedy_allocation(
                    sens_table, layer_costs, target_budget_bytes, critical_indices=crit_idx
                )
                
                # D. Uniform W4 + Endpoints (INT4 internal, INT8 endpoints)
                alloc_w4 = [8 if i in crit_idx else 4 for i in range(len(layers))]
                
                methods = [
                    ('dp_measured', alloc_dp),
                    ('hawq', alloc_hawq),
                    ('greedy', alloc_greedy),
                    ('uniform_w4_endpoints', alloc_w4)
                ]
                
                for meth_name, alloc in methods:
                    # Endpoint protocol assertion: Enforce b[0] == 8 and b[-1] == 8 and len(alloc) == L
                    alloc = validate_allocation_endpoints(alloc, len(layers), meth_name)
                    real_bytes = sum(layer_costs[i][alloc[i]] for i in range(len(alloc)))
                    
                    # PTQ evaluation
                    model_ptq = model_cls(**task['model_kwargs']).to(device)
                    model_ptq.load_state_dict(model_fp32.state_dict())
                    apply_layer_quantization(model_ptq, alloc)
                    ptq_acc = evaluate_loader(model_ptq, test_loader, device=device)
                    
                    # FT evaluation (5 epochs fine-tuning)
                    model_ft = model_cls(**task['model_kwargs']).to(device)
                    model_ft.load_state_dict(model_ptq.state_dict())
                    t_ft_0 = time.time()
                    finetune_quantized_model(
                        model_ft, train_loader, val_loader, device=device,
                        epochs=5, lr=1e-4, is_cifar=task['is_cifar']
                    )
                    ft_time = time.time() - t_ft_0
                    ft_acc = evaluate_loader(model_ft, test_loader, device=device)
                    
                    rec_ptq = {
                        'dataset': ds_name, 'model': model_name, 'seed': seed,
                        'budget_target_frac': b_frac, 'method': meth_name, 'regime': 'ptq',
                        'bit_allocation': alloc, 'test_accuracy': ptq_acc,
                        'real_size_kb': real_bytes / 1024.0, 'int8_baseline_kb': total_int8_bytes / 1024.0
                    }
                    rec_ft = {
                        'dataset': ds_name, 'model': model_name, 'seed': seed,
                        'budget_target_frac': b_frac, 'method': meth_name, 'regime': 'finetune',
                        'bit_allocation': alloc, 'test_accuracy': ft_acc, 'ft_time_s': ft_time,
                        'real_size_kb': real_bytes / 1024.0, 'int8_baseline_kb': total_int8_bytes / 1024.0
                    }
                    runs_records.extend([rec_ptq, rec_ft])
                    task_seed_runs.extend([rec_ptq, rec_ft])
                    
                    print(f"[{meth_name:20s}] B={int(b_frac*100):2d}% | Size={real_bytes/1024.0:5.1f}KB | PTQ={ptq_acc:5.2f}% | FT={ft_acc:5.2f}%")
                    
            with open(OUT_RUNS, "w", encoding="utf-8") as f:
                for r in runs_records:
                    f.write(json.dumps(r) + "\n")
            with open(RESULTS_DIR / "fp32_results.json", "w", encoding="utf-8") as f:
                json.dump(fp32_results, f, indent=2)
                    
    # Write runs_multiseed.jsonl
    with open(OUT_RUNS, "w", encoding="utf-8") as f:
        for r in runs_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(runs_records)} multi-seed evaluation runs to {OUT_RUNS}")
    
    # -------------------------------------------------------------------------
    # Aggregation & Honest Statistical Testing (n=5 models as unit of analysis)
    # -------------------------------------------------------------------------
    summary = {}
    ft_runs = [r for r in runs_records if r['regime'] == 'finetune']
    
    # FP32 summary per task
    fp32_summary = {}
    for task in TASKS:
        t_id = f"{task['dataset']}_{task['model']}"
        accs = [r['fp32_test_acc'] for r in fp32_results if r['dataset'] == task['dataset'] and r['model'] == task['model']]
        mean_acc = float(np.mean(accs))
        sd_acc = float(np.std(accs, ddof=1))
        fp32_summary[t_id] = {
            'mean': mean_acc, 'std': sd_acc,
            'seeds': accs,
            'ci95': [float(mean_acc - 2.776 * sd_acc / np.sqrt(len(accs))), float(mean_acc + 2.776 * sd_acc / np.sqrt(len(accs)))]
        }
    summary['fp32_baselines'] = fp32_summary
    
    # Cell summary (mean +/- std across 5 seeds)
    cell_matrix = {}
    for task in TASKS:
        t_id = f"{task['dataset']}_{task['model']}"
        cell_matrix[t_id] = {}
        for b in BUDGETS:
            cell_matrix[t_id][str(b)] = {}
            for m in ['dp_measured', 'hawq', 'greedy', 'uniform_w4_endpoints']:
                vals = [r['test_accuracy'] for r in ft_runs if r['dataset'] == task['dataset'] and r['model'] == task['model'] and r['budget_target_frac'] == b and r['method'] == m]
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                cell_matrix[t_id][str(b)][m] = {
                    'mean': round(mean_v, 2),
                    'std': round(std_v, 2),
                    'n_seeds': len(vals),
                    'values': [round(x, 2) for x in vals],
                    'ci95': [round(mean_v - 2.776 * std_v / np.sqrt(len(vals)), 2), round(mean_v + 2.776 * std_v / np.sqrt(len(vals)), 2)]
                }
    summary['cell_matrix'] = cell_matrix
    
    # -------------------------------------------------------------------------
    # Honest Model-Level Tests (n=5 models)
    # -------------------------------------------------------------------------
    # For each budget, average over seeds to obtain 1 paired value per model (n=5)
    model_level_tests = {}
    for b in BUDGETS:
        model_dp_means = []
        model_hawq_means = []
        model_w4_means = []
        model_diffs = []
        
        for task in TASKS:
            t_id = f"{task['dataset']}_{task['model']}"
            dp_v = cell_matrix[t_id][str(b)]['dp_measured']['mean']
            hw_v = cell_matrix[t_id][str(b)]['hawq']['mean']
            w4_v = cell_matrix[t_id][str(b)]['uniform_w4_endpoints']['mean']
            diff = dp_v - hw_v
            model_dp_means.append(dp_v)
            model_hawq_means.append(hw_v)
            model_w4_means.append(w4_v)
            model_diffs.append(diff)
            
        mean_diff = float(np.mean(model_diffs))
        median_diff = float(np.median(model_diffs))
        sd_diff = float(np.std(model_diffs, ddof=1))
        
        # Exact Permutation Test for n=5 (2^5 = 32 permutations)
        perms = []
        for sign0 in [-1, 1]:
            for sign1 in [-1, 1]:
                for sign2 in [-1, 1]:
                    for sign3 in [-1, 1]:
                        for sign4 in [-1, 1]:
                            perms.append(np.mean([
                                sign0 * abs(model_diffs[0]),
                                sign1 * abs(model_diffs[1]),
                                sign2 * abs(model_diffs[2]),
                                sign3 * abs(model_diffs[3]),
                                sign4 * abs(model_diffs[4])
                            ]))
        perm_p = float(np.mean([abs(p) >= abs(mean_diff) - 1e-9 for p in perms]))
        
        # Wilcoxon (n=5)
        try:
            w_stat, wilc_p = stats.wilcoxon(model_diffs, alternative='two-sided')
            w_stat, wilc_p = float(w_stat), float(wilc_p)
        except Exception as e:
            w_stat, wilc_p = 0.0, 1.0
            
        model_level_tests[str(b)] = {
            'n_models': len(model_diffs),
            'model_diffs_dp_minus_hawq': [round(x, 2) for x in model_diffs],
            'mean_diff': round(mean_diff, 2),
            'median_diff': round(median_diff, 2),
            'std_diff': round(sd_diff, 2),
            'exact_permutation_p': round(perm_p, 4),
            'wilcoxon_w': w_stat,
            'wilcoxon_p': round(wilc_p, 4),
            'statistically_significant_p05': bool(perm_p < 0.05),
            'dp_vs_uniform_w4_mean_diff': round(float(np.mean(np.array(model_dp_means) - np.array(model_w4_means))), 2)
        }
    summary['model_level_tests'] = model_level_tests
    
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_TESTS, "w", encoding="utf-8") as f:
        json.dump(model_level_tests, f, indent=2)
        
    print("\n=== Multi-Seed Honest Statistical Summary (n=5 Models) ===")
    for b in BUDGETS:
        t = model_level_tests[str(b)]
        print(f"Budget {int(b*100)}%: DP - HAWQ mean={t['mean_diff']:+.2f}%, median={t['median_diff']:+.2f}%, exact perm p={t['exact_permutation_p']:.4f}, sig={t['statistically_significant_p05']}")
        
    # -------------------------------------------------------------------------
    # A6: Ablation Studies
    # -------------------------------------------------------------------------
    print("\n--- Executing Item A6 Ablations ---")
    ablations_out = {}
    
    # 1. Validation Sample Count Ablation (Replaces vacuous calibration size)
    val_sample_counts = [50, 100, 500, 2000]
    val_ablation_results = {}
    cifar_loaders = get_cifar10_loaders(batch_size=128, seed=42)
    sep_model = SeparableCNN(num_classes=10).to(device)
    ckpt = torch.load(CHECKPOINTS_DIR / "fp32_cifar10_separable_cnn_seed42.pt", map_location=device)
    sep_model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    
    for n_samples in val_sample_counts:
        # Subsample validation loader to n_samples
        sub_x_list, sub_y_list = [], []
        cnt = 0
        for bx, by in cifar_loaders['val_loader']:
            sub_x_list.append(bx)
            sub_y_list.append(by)
            cnt += bx.size(0)
            if cnt >= n_samples:
                break
        sub_x = torch.cat(sub_x_list)[:n_samples]
        sub_y = torch.cat(sub_y_list)[:n_samples]
        sub_val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(sub_x, sub_y), batch_size=128, shuffle=False
        )
        
        # Profile sensitivity with this sample size
        layers = get_layer_weights(sep_model)
        s_table = []
        fp32_val_acc = evaluate_loader(sep_model, sub_val_loader, device=device)
        with torch.no_grad():
            for name, mod in layers:
                w_orig = mod.weight.data.clone()
                s_dict = {}
                for b in [2, 4, 8]:
                    scale, zp = compute_quant_params(w_orig.min(), w_orig.max(), bits=b, symmetric=True)
                    q = quantize_affine(w_orig, scale, zp, bits=b)
                    deq = dequantize_affine(q, scale, zp)
                    mod.weight.data.copy_(deq)
                    v_acc = evaluate_loader(sep_model, sub_val_loader, device=device)
                    s_dict[b] = max(0.0, fp32_val_acc - v_acc)
                mod.weight.data.copy_(w_orig)
                s_table.append(s_dict)
                
        # Solve DP knapsack at 50% budget
        costs = compute_layer_costs(sep_model, [2, 4, 8])
        alloc, _, _ = solve_dp_knapsack(
            s_table, costs, cost_budget=compute_total_int8_cost(sep_model)*0.5,
            allowed_bits=[2, 4, 8], critical_indices=[0, 23], critical_bits=8
        )
        m_eval = SeparableCNN(num_classes=10).to(device)
        m_eval.load_state_dict(sep_model.state_dict())
        apply_layer_quantization(m_eval, alloc)
        test_acc = evaluate_loader(m_eval, cifar_loaders['test_loader'], device=device)
        val_ablation_results[str(n_samples)] = {'test_acc': round(test_acc, 2), 'alloc': alloc[:5]}
        print(f"Validation Sensitivity Samples N={n_samples:4d}: Test Acc = {test_acc:.2f}%")
    ablations_out['sensitivity_sample_ablation'] = val_ablation_results
    
    # 2. Endpoint Protection Across Low Budgets (25%, 35%, 50%) across 5 seeds
    print("\nEvaluating Endpoint Protection across budgets (25%, 35%, 50%) over 5 seeds...")
    endpoint_ablation = {}
    for b in [0.25, 0.35, 0.50]:
        prot_accs, unprot_accs = [], []
        for seed in SEEDS:
            # Protected
            r_prot = [r['test_accuracy'] for r in ft_runs if r['dataset'] == 'cifar10' and r['model'] == 'separable_cnn' and r['seed'] == seed and r['budget_target_frac'] == b and r['method'] == 'dp_measured']
            if r_prot:
                prot_accs.append(r_prot[0])
                
            # Unprotected (Run DP without critical indices)
            loaders = get_cifar10_loaders(batch_size=128, seed=seed)
            m_orig = SeparableCNN(num_classes=10).to(device)
            ckpt = torch.load(CHECKPOINTS_DIR / f"fp32_cifar10_separable_cnn_seed{seed}.pt", map_location=device)
            m_orig.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
            costs = compute_layer_costs(m_orig, [2, 3, 4, 5, 6, 7, 8])
            alloc_unprot, _, _ = solve_dp_knapsack(
                sens_table, costs, cost_budget=compute_total_int8_cost(m_orig)*b,
                allowed_bits=[2, 3, 4, 5, 6, 7, 8], critical_indices=[]
            )
            m_unprot = SeparableCNN(num_classes=10).to(device)
            m_unprot.load_state_dict(m_orig.state_dict())
            apply_layer_quantization(m_unprot, alloc_unprot)
            finetune_quantized_model(m_unprot, loaders['train_loader'], loaders['val_loader'], device=device, epochs=5, lr=1e-4, is_cifar=True)
            unprot_acc = evaluate_loader(m_unprot, loaders['test_loader'], device=device)
            unprot_accs.append(unprot_acc)
            
        endpoint_ablation[str(b)] = {
            'protected_mean': round(float(np.mean(prot_accs)), 2),
            'protected_std': round(float(np.std(prot_accs, ddof=1)), 2),
            'unprotected_mean': round(float(np.mean(unprot_accs)), 2),
            'unprotected_std': round(float(np.std(unprot_accs, ddof=1)), 2),
            'delta_protected_minus_unprot': round(float(np.mean(prot_accs) - np.mean(unprot_accs)), 2)
        }
        print(f"Budget {int(b*100)}%: Protected = {endpoint_ablation[str(b)]['protected_mean']:.2f}%, Unprotected = {endpoint_ablation[str(b)]['unprotected_mean']:.2f}%, Delta = {endpoint_ablation[str(b)]['delta_protected_minus_unprot']:+.2f}%")
    ablations_out['endpoint_protection_ablation'] = endpoint_ablation
    
    with open(OUT_ABLATIONS, "w", encoding="utf-8") as f:
        json.dump(ablations_out, f, indent=2)
        
    # -------------------------------------------------------------------------
    # A7: Ground-Up Retrain of Full Row (Seed 99)
    # -------------------------------------------------------------------------
    print("\n--- Executing Item A7: From-Scratch Retrain of Full Row (Seed 99) ---")
    retrain_seed = 99
    torch.manual_seed(retrain_seed)
    np.random.seed(retrain_seed)
    
    loaders99 = get_cifar10_loaders(batch_size=128, seed=retrain_seed)
    m99 = SeparableCNN(num_classes=10).to(device)
    
    t0_retrain = time.time()
    res99 = train_fp32_model(m99, loaders99['train_loader'], loaders99['val_loader'], device=device, epochs=25, lr=0.05, is_cifar=True)
    fp32_99_time = time.time() - t0_retrain
    m99.load_state_dict(res99['state_dict'])
    fp32_99_acc = evaluate_loader(m99, loaders99['test_loader'], device=device)
    
    # Sensitivities
    layers99 = get_layer_weights(m99)
    s_table99 = []
    fp32_val99 = evaluate_loader(m99, loaders99['val_loader'], device=device)
    with torch.no_grad():
        for name, mod in layers99:
            w_orig = mod.weight.data.clone()
            s_dict = {}
            for b in [2, 3, 4, 5, 6, 7, 8]:
                scale, zp = compute_quant_params(w_orig.min(), w_orig.max(), bits=b, symmetric=True)
                q = quantize_affine(w_orig, scale, zp, bits=b)
                deq = dequantize_affine(q, scale, zp)
                mod.weight.data.copy_(deq)
                v_acc = evaluate_loader(m99, loaders99['val_loader'], device=device)
                s_dict[b] = max(0.0, fp32_val99 - v_acc)
            mod.weight.data.copy_(w_orig)
            s_table99.append(s_dict)
            
    costs99 = compute_layer_costs(m99, [2, 3, 4, 5, 6, 7, 8])
    total_int8_99 = compute_total_int8_cost(m99)
    
    row_99_results = {'seed': retrain_seed, 'fp32_acc': round(fp32_99_acc, 2), 'fp32_time_s': round(fp32_99_time, 1), 'budgets': {}}
    for b in BUDGETS:
        alloc_dp99, cost_dp99, _ = solve_dp_knapsack(
            s_table99, costs99, cost_budget=total_int8_99 * b,
            allowed_bits=[2, 3, 4, 5, 6, 7, 8], critical_indices=[0, 23], critical_bits=8
        )
        m_eval99 = SeparableCNN(num_classes=10).to(device)
        m_eval99.load_state_dict(m99.state_dict())
        apply_layer_quantization(m_eval99, alloc_dp99)
        ptq_acc99 = evaluate_loader(m_eval99, loaders99['test_loader'], device=device)
        
        finetune_quantized_model(m_eval99, loaders99['train_loader'], loaders99['val_loader'], device=device, epochs=5, lr=1e-4, is_cifar=True)
        ft_acc99 = evaluate_loader(m_eval99, loaders99['test_loader'], device=device)
        
        row_99_results['budgets'][str(b)] = {
            'alloc_first5': alloc_dp99[:5],
            'size_kb': round(cost_dp99 / 1024.0, 1),
            'ptq_acc': round(ptq_acc99, 2),
            'ft_acc': round(ft_acc99, 2)
        }
        print(f"Seed 99 Retrain | Budget {int(b*100)}%: PTQ = {ptq_acc99:.2f}%, FT = {ft_acc99:.2f}%")
        
    with open(OUT_RETRAIN, "w", encoding="utf-8") as f:
        json.dump(row_99_results, f, indent=2)
        
    total_runtime = time.time() - start_total_time
    print(f"\n=== Multi-Seed Suite Completed Successfully in {total_runtime/60.0:.1f} minutes ===")

if __name__ == "__main__":
    run_matrix()
