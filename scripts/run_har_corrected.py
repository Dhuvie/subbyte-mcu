#!/usr/bin/env python3
"""
Corrected UCI HAR Execution Runner (Endpoint Protocol Fix).

Executes the 160-cell experimental matrix for UCI HAR / CNN1D:
- 5 seeds: 42, 43, 44, 45, 46
- 4 methods: dp_measured, hawq, greedy, uniform_w4_endpoints
- 4 budgets: 0.25, 0.35, 0.50, 0.75
- 2 regimes: ptq, finetune
Total = 160 rows.

Enforces:
- Dynamic critical indices: [0, L-1] (first and last quantizable layer at 8 bits)
- Allocator endpoint assertions (b[0] == 8 and b[-1] == 8, len(alloc) == L)
- Resumable JSONL logging into results/ext_har_corrected.jsonl
- Detailed metadata: bit_allocation, payload_bytes, container_bytes, budget fields, test_read_count=1
- Structured logging to logs/ext/har_corrected.log with provenance, UTC timestamps, memory tracking, and END marker.
"""

import sys
import os
import time
import json
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subbyte.datasets.har import get_har_loaders
from subbyte.models.cnn1d import CNN1D
from subbyte.training.train import finetune_quantized_model
from subbyte.training.evaluate import evaluate_loader
from subbyte.allocation.knapsack import solve_dp_knapsack
from subbyte.allocation.sensitivity import compute_hutchinson_hessian_trace
from subbyte.quant.affine import quantize_affine, dequantize_affine, compute_quant_params
from subbyte.quant.qpack_binary import compute_qpack_formula_size
from scripts.run_matrix import (
    get_layer_weights,
    apply_layer_quantization,
    compute_layer_costs,
    compute_total_int8_cost,
    solve_greedy_allocation,
    validate_allocation_endpoints
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = ROOT / "logs" / "ext"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OUT_RUNS = RESULTS_DIR / "ext_har_corrected_v2.jsonl"
OUT_LOG = LOGS_DIR / "har_corrected_v2.log"

BUDGETS = [0.25, 0.35, 0.50, 0.75]
SEEDS = [42, 43, 44, 45, 46]

class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.file = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str):
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

def get_row_key(record: Dict[str, Any]) -> str:
    return f"{record['seed']}_{record['budget_target_frac']}_{record['method']}_{record['regime']}"

def run():
    logger = Logger(OUT_LOG)
    start_time_utc = datetime.now(timezone.utc)
    utc_offset = "+05:30"
    
    logger.log("================================================================================")
    logger.log("START OF RUN: CORRECTED UCI HAR 160-CELL EXECUTION (ENDPOINT PROTOCOL)")
    logger.log(f"Start Time (UTC)       : {start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} (Host local offset: {utc_offset})")
    logger.log(f"Process PID            : {os.getpid()}")
    logger.log(f"Python Version         : {sys.version.split()[0]}")
    logger.log(f"PyTorch Version        : {torch.__version__}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        logger.log(f"Compute Device         : GPU ({gpu_name})")
        logger.log(f"CUDA Driver / Toolkit  : CUDA {cuda_ver}")
    else:
        device = torch.device('cpu')
        logger.log("Compute Device         : CPU")
        
    logger.log(f"Output Target          : {OUT_RUNS.name}")
    logger.log(f"Log Target             : {OUT_LOG.name}")
    logger.log("================================================================================")

    # Load existing runs for resume capability
    existing_records = {}
    if OUT_RUNS.exists():
        with open(OUT_RUNS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_records[get_row_key(rec)] = rec
        logger.log(f"Resuming runner: Found {len(existing_records)} existing rows in {OUT_RUNS.name}")

    # Initialize model metadata
    dummy_model = CNN1D()
    layers = get_layer_weights(dummy_model)
    L = len(layers)
    crit_idx = [0, L - 1]
    layer_shapes = [list(mod.weight.shape) for _, mod in layers]
    N_l = [mod.weight.numel() for _, mod in layers]
    C_l = [mod.weight.shape[0] for _, mod in layers]
    total_weights = sum(N_l)
    int8_baseline_bytes = total_weights
    int8_container_bytes = compute_qpack_formula_size(layer_shapes, [8] * L)
    
    logger.log(f"Model: CNN1D (L={L} layers, Total Weights={total_weights:,} params)")
    logger.log(f"Dynamic Critical Indices: {crit_idx} (Endpoints at 8-bit strictly enforced)")
    logger.log(f"INT8 Payload Baseline   : {int8_baseline_bytes:,} B ({int8_baseline_bytes/1024:.2f} KB)")
    logger.log(f"INT8 Container Baseline : {int8_container_bytes:,} B ({int8_container_bytes/1024:.2f} KB)")
    logger.log("--------------------------------------------------------------------------------")

    bits_list = [2, 3, 4, 5, 6, 7, 8]
    total_rows_written = len(existing_records)
    batch_size = 64
    epochs = 25
    lr = 0.01

    tracemalloc.start()
    peak_ram_mb = 0.0
    peak_vram_mb = 0.0

    for seed in SEEDS:
        logger.log(f"\n>>> Executing Seed {seed} (Seeds 42-46) <<<")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        loaders = get_har_loaders(batch_size=batch_size, seed=seed)
        train_loader = loaders['train_loader']
        val_loader = loaders['val_loader']
        test_loader = loaders['test_loader']

        ckpt_path = CHECKPOINTS_DIR / f"fp32_har_cnn1d_seed{seed}.pt"
        assert ckpt_path.exists(), f"FP32 Checkpoint missing: {ckpt_path}"
        
        model_fp32 = CNN1D().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model_fp32.load_state_dict(sd)

        # Baseline evaluation
        fp32_val_acc = evaluate_loader(model_fp32, val_loader, device=device)
        logger.log(f"Loaded FP32 Checkpoint (seed {seed}) | Val Acc: {fp32_val_acc:.2f}%")

        layers = get_layer_weights(model_fp32)
        L = len(layers)
        crit_idx = [0, L - 1]
        layer_costs = compute_layer_costs(model_fp32, bits_list)

        # 1. Measure empirical layer sensitivity on val split
        logger.log(f"Profiling layer sensitivities on validation split (N_val={len(val_loader.dataset)})...")
        sens_table = []
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

        # 2. Estimate Hessian traces for HAWQ via Hutchinson estimator
        logger.log("Estimating Hessian traces via Hutchinson estimator (num_samples=100)...")
        criterion = nn.CrossEntropyLoss()
        traces_dict = compute_hutchinson_hessian_trace(
            model_fp32, val_loader, criterion, device=device, num_samples=100
        )
        hessian_traces = [traces_dict.get(f"{name}.weight", 1.0) for name, _ in layers]

        # 3. Iterate over budgets and methods
        for b_frac in BUDGETS:
            target_budget_bytes = int8_baseline_bytes * b_frac
            target_budget_container_bytes = int8_container_bytes * b_frac

            # A. DP Knapsack Allocation
            alloc_dp, cost_dp, _ = solve_dp_knapsack(
                sens_table, layer_costs, cost_budget=target_budget_bytes,
                allowed_bits=bits_list, critical_indices=crit_idx, critical_bits=8
            )

            # B. HAWQ Allocation
            log_tr = [np.log(max(1e-12, t)) for t in hessian_traces]
            min_t, max_t = min(log_tr), max(log_tr)
            norm_tr = [(t - min_t) / (max_t - min_t + 1e-12) for t in log_tr]
            alloc_hawq = []
            for i, nt in enumerate(norm_tr):
                if i in crit_idx:
                    alloc_hawq.append(8)
                else:
                    b_val = int(2 + round(nt * (6.0 * b_frac / 0.75)))
                    b_val = max(2, min(8, b_val))
                    alloc_hawq.append(b_val)
            cur_h_cost = sum(layer_costs[i][alloc_hawq[i]] for i in range(len(alloc_hawq)))
            while cur_h_cost > target_budget_bytes:
                reducible = [i for i in range(len(alloc_hawq)) if i not in crit_idx and alloc_hawq[i] > 2]
                if not reducible:
                    break
                best_i = min(reducible, key=lambda idx: norm_tr[idx])
                alloc_hawq[best_i] -= 1
                cur_h_cost = sum(layer_costs[i][alloc_hawq[i]] for i in range(len(alloc_hawq)))

            # C. Sensitivity-Greedy Allocation
            alloc_greedy = solve_greedy_allocation(
                sens_table, layer_costs, target_budget_bytes, critical_indices=crit_idx
            )

            # D. Uniform W4 + Endpoints (INT4 internal, INT8 endpoints)
            alloc_w4 = [8 if i in crit_idx else 4 for i in range(L)]

            methods = [
                ('dp_measured', alloc_dp),
                ('hawq', alloc_hawq),
                ('greedy', alloc_greedy),
                ('uniform_w4_endpoints', alloc_w4)
            ]

            for meth_name, alloc in methods:
                # Enforce endpoint assertion
                alloc = validate_allocation_endpoints(alloc, L, meth_name)

                # Compute exact bytes
                payload_bytes = sum(int((N_l[i] * alloc[i] + 7) // 8) for i in range(L))
                container_bytes = compute_qpack_formula_size(layer_shapes, alloc)
                payload_ratio = payload_bytes / int8_baseline_bytes
                container_ratio = container_bytes / int8_container_bytes

                # Label floor / infeas status
                is_floor = (payload_bytes > target_budget_bytes) and (meth_name != 'uniform_w4_endpoints')
                status = "[floor]" if is_floor else "ok"

                # Task A Invariant:
                # 1. Endpoints must be at 8 bits
                assert alloc[0] == 8 and alloc[-1] == 8, f"Endpoint violation in {meth_name}: {alloc}"
                # 2. For budget-matched methods, check compliance and under-utilization
                if meth_name != 'uniform_w4_endpoints':
                    if is_floor:
                        assert alloc == [8, 2, 2, 2, 8], f"Floor cell unexpected allocation: {alloc}"
                    else:
                        assert payload_ratio <= b_frac + 1e-4, f"Budget exceeded: {payload_ratio:.4f} > {b_frac:.4f} for {meth_name}"
                        # Under-utilization invariant for DP knapsack solver:
                        if meth_name == 'dp_measured':
                            step_tol = 0.08
                            assert payload_ratio >= b_frac - step_tol, f"Under-utilization in DP: target {b_frac:.4f}, achieved {payload_ratio:.4f} (gap {b_frac - payload_ratio:.4f} > {step_tol})"

                # Check if PTQ and FT keys already exist
                k_ptq = f"{seed}_{b_frac}_{meth_name}_ptq"
                k_ft = f"{seed}_{b_frac}_{meth_name}_finetune"

                # PTQ Run
                if k_ptq in existing_records:
                    rec_ptq = existing_records[k_ptq]
                else:
                    model_ptq = CNN1D().to(device)
                    model_ptq.load_state_dict(model_fp32.state_dict())
                    apply_layer_quantization(model_ptq, alloc)
                    # Test loader read strictly once for final PTQ configuration
                    ptq_acc = float(evaluate_loader(model_ptq, test_loader, device=device))

                    rec_ptq = {
                        'dataset': 'har',
                        'model': 'cnn1d',
                        'seed': seed,
                        'budget_target_frac': b_frac,
                        'method': meth_name,
                        'regime': 'ptq',
                        'bit_allocation': alloc,
                        'test_accuracy': round(ptq_acc, 2),
                        'real_size_kb': round(payload_bytes / 1024.0, 2),
                        'int8_baseline_kb': round(int8_baseline_bytes / 1024.0, 2),
                        'payload_bytes': payload_bytes,
                        'int8_baseline_bytes': int8_baseline_bytes,
                        'payload_ratio': round(payload_ratio, 4),
                        'container_bytes': container_bytes,
                        'int8_container_bytes': int8_container_bytes,
                        'container_ratio': round(container_ratio, 4),
                        'budget_target_bytes': target_budget_bytes,
                        'budget_target_container_bytes': target_budget_container_bytes,
                        'test_read_count': 1,
                        'status': status
                    }
                    existing_records[k_ptq] = rec_ptq
                    with open(OUT_RUNS, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec_ptq) + "\n")
                        f.flush()
                    total_rows_written += 1

                # FT Run
                if k_ft in existing_records:
                    rec_ft = existing_records[k_ft]
                else:
                    model_ft = CNN1D().to(device)
                    model_ft.load_state_dict(model_fp32.state_dict())
                    apply_layer_quantization(model_ft, alloc)
                    t_ft_0 = time.time()
                    finetune_quantized_model(
                        model_ft, train_loader, val_loader, device=device,
                        epochs=5, lr=1e-4, is_cifar=False
                    )
                    ft_time = time.time() - t_ft_0
                    # Test loader read strictly once for final FT configuration
                    ft_acc = float(evaluate_loader(model_ft, test_loader, device=device))

                    rec_ft = {
                        'dataset': 'har',
                        'model': 'cnn1d',
                        'seed': seed,
                        'budget_target_frac': b_frac,
                        'method': meth_name,
                        'regime': 'finetune',
                        'bit_allocation': alloc,
                        'test_accuracy': round(ft_acc, 2),
                        'ft_time_s': round(ft_time, 2),
                        'real_size_kb': round(payload_bytes / 1024.0, 2),
                        'int8_baseline_kb': round(int8_baseline_bytes / 1024.0, 2),
                        'payload_bytes': payload_bytes,
                        'int8_baseline_bytes': int8_baseline_bytes,
                        'payload_ratio': round(payload_ratio, 4),
                        'container_bytes': container_bytes,
                        'int8_container_bytes': int8_container_bytes,
                        'container_ratio': round(container_ratio, 4),
                        'budget_target_bytes': target_budget_bytes,
                        'budget_target_container_bytes': target_budget_container_bytes,
                        'test_read_count': 1,
                        'status': status
                    }
                    existing_records[k_ft] = rec_ft
                    with open(OUT_RUNS, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec_ft) + "\n")
                        f.flush()
                    total_rows_written += 1

                # Update memory tracking
                cur_ram, peak_ram = tracemalloc.get_traced_memory()
                ram_mb = peak_ram / (1024 * 1024)
                if ram_mb > peak_ram_mb:
                    peak_ram_mb = ram_mb
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    if vram_mb > peak_vram_mb:
                        peak_vram_mb = vram_mb

                logger.log(
                    f"[{meth_name:20s}] B={int(b_frac*100):2d}% | "
                    f"Alloc={alloc} | "
                    f"Payload={payload_bytes}B ({payload_ratio*100:.1f}%) | "
                    f"Cont={container_bytes}B ({container_ratio*100:.1f}%) | "
                    f"PTQ={rec_ptq['test_accuracy']:5.2f}% | "
                    f"FT={rec_ft['test_accuracy']:5.2f}%"
                )

    end_time_utc = datetime.now(timezone.utc)
    duration_s = (end_time_utc - start_time_utc).total_seconds()

    logger.log("\n================================================================================")
    logger.log("SUMMARY OF HAR RE-RUN (ENDPOINT PROTOCOL)")
    logger.log(f"End Time (UTC)         : {end_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} (Host local offset: {utc_offset})")
    logger.log(f"Elapsed Wall Time      : {duration_s:.1f} seconds ({duration_s/60:.2f} minutes)")
    logger.log(f"Total Rows in File     : {len(existing_records)}")
    logger.log(f"Peak Host RAM Usage    : {peak_ram_mb:.1f} MB")
    if torch.cuda.is_available():
        logger.log(f"Peak GPU VRAM Usage    : {peak_vram_mb:.1f} MB")
    logger.log(f"Exit Status            : SUCCESS (EXIT 0)")
    logger.log("END OF RUN - END MARKER")
    logger.log("================================================================================")
    logger.close()
    return 0

if __name__ == "__main__":
    sys.exit(run())
