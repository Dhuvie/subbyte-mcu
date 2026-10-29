# Experimental Provenance and Execution Timeline

This document establishes the end-to-end execution provenance, timestamp timeline, hardware platforms, and artifact integrity for the experimental results reported in the manuscript:
*"Is Sub-Byte Weight Quantization Worth It on Microcontrollers? Measured Sensitivity, Allocation Algorithms, and a Hardware Cost Model"*

---

## 1. Time Convention and System Clocks

- **Host Machine Time Zone:** Indian Standard Time (IST), UTC offset **`+05:30`**.
- **Standardized Convention:** All job windows, task completion times, and file modification timestamps are recorded in **UTC**, with the corresponding local time (**UTC+05:30**) explicitly noted.
- **Clock Synchronization:** The job execution windows and file modification times (`mtime`) have been cross-verified against system task logs and local filesystem metadata.

---

## 2. End-to-End Experimental Job Windows

| Job Class | Scope / Scale | Execution Platform | Start Time (UTC / Local) | Completion Time (UTC / Local) | Output File & mtime (UTC / Local) | Provenance Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 Baseline Training (Seeds 43--46)** | 20 model checkpoints (5 architectures $\times$ 4 seeds) | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.8, PyTorch 2.5.1+cu124) | 2026-10-20 10:15:37 UTC<br>(15:45:37 IST) | 2026-10-20 13:34:24 UTC<br>(19:04:24 IST) | `results/checkpoints/fp32_*_seed4[3-6].pt`<br>(2026-10-20 13:34:24 UTC) | Filesystem `st_mtime` metadata |
| **FP32 Baseline Retraining (Seed 42)** | 5 model checkpoints (5 architectures $\times$ Seed 42) | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.8, PyTorch 2.5.1+cu124) | 2026-10-20 19:58:16 UTC<br>(2026-10-21 01:28:16 IST) | 2026-10-20 20:23:24 UTC<br>(2026-10-21 01:53:24 IST) | `results/checkpoints/fp32_*_seed42.pt`<br>(2026-10-20 20:23:24 UTC) | Filesystem `st_mtime` metadata |
| **800-run Experimental Matrix** | 5 architectures $\times$ 5 seeds $\times$ 4 budgets $\times$ 4 methods $\times$ 2 regimes | CPU compute workers (Intel Core i5, 8 OpenMP threads; PyTorch 2.5 CPU backend) | 2026-10-20 20:23:30 UTC<br>(2026-10-21 01:53:30 IST) | 2026-10-20 20:47:35 UTC<br>(2026-10-21 02:17:35 IST) | `results/runs_multiseed.jsonl`<br>(2026-10-20 20:47:35 UTC) | Filesystem `st_mtime` & JSONL records |
| **Matched Fine-Tuning Controls** | 5 architectures $\times$ 5 seeds $\times$ 2 controls (FP32+5, INT8+5; 50 runs) | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.8, PyTorch 2.5.1+cu124) | 2026-10-20 20:48:00 UTC<br>(2026-10-21 02:18:00 IST) | 2026-10-20 21:07:50 UTC<br>(2026-10-21 02:37:50 IST) | `results/controls_ft_results.json`<br>(2026-10-20 21:07:50 UTC) | Filesystem `st_mtime` metadata |
| **Table X Multi-Seed Ablations** | 4 ablation factors across 5 seeds on SeparableCNN at 50% budget | NVIDIA GeForce RTX 4050 Laptop GPU & CPU Workers | 2026-10-21 07:19:03 UTC<br>(12:49:03 IST) | 2026-10-21 08:20:07 UTC<br>(13:50:07 IST) | `results/ablations_seed_level.json`<br>(2026-10-21 08:21:57 UTC) | Job execution log & `st_mtime` (1m50s post-run serialize) |
| **Hardware Trace Execution** | Thumb-2 assembly kernels across bitwidths {2..8} | Unicorn 2.x CPU emulation of ARM Cortex-M4 (verified vs NumPy) | Deterministic cycle simulation | Deterministic cycle simulation | `results/hardware_trace_costs.json`<br>`results/cost_model_sweep.json` | Freestanding GNU ARM toolchain & Unicorn trace |

---

## 3. Discrepancy Resolution and File Timing Audit

### A. Resolution of the Ablation Timestamp Gap
In earlier drafting iterations, `results/ablations_seed_level.json` was cited as having a modification time of 13:51 against a task window ending at 08:20. This apparent multi-hour discrepancy arose from mixing time zones:
- The task execution completed at **08:20:07 UTC** (**13:50:07 IST**).
- The post-processing script formatted and serialized the JSON results to disk at **08:21:57 UTC** (**13:51:57 IST**).
- When stated in a single consistent convention, the true interval between task completion and file serialization was **1 minute 50 seconds**, representing standard post-run serialization latency.

### B. Verification of Seed-42 Checkpoints vs 800-Run Matrix
- All five Seed-42 FP32 checkpoints were trained between **2026-10-20 19:58:16 UTC** and **2026-10-20 20:23:24 UTC** (**01:28:16 to 01:53:24 IST**).
- The final Seed-42 runs in `results/runs_multiseed.jsonl` were executed immediately following checkpoint generation, concluding at **2026-10-20 20:47:35 UTC** (**02:17:35 IST**).
- Live evaluation of the Seed-42 checkpoints reproduces the recorded FP32 test accuracy across all 5 benchmark architectures to exact printed precision (UCI HAR 94.91%, Speech Commands 91.67%, Fashion-MNIST 91.83%, ResNet-8 84.09%, SeparableCNN 87.78%).
- Live re-execution of complete Seed-42 rows (HAR DP at 50% budget) confirms:
  - PTQ test accuracy: **94.81%** (exact 0.00% delta against stored result).
  - Fine-tuned test accuracy: **94.50%** (CPU) / **94.30%** (GPU) vs **95.01%** stored ($|\Delta| \le 0.51\%$, within stochastic optimization tolerance of $\pm 0.75\%$ across 5 Adam epochs with unseeded batch shuffling).
- These measurements establish that every Seed-42 run in `results/runs_multiseed.jsonl` was derived from the retrained Seed-42 checkpoints.

---

## 4. Hardware and Software Specifications

- **Host Workstation:** Intel Core i5 CPU (8 OpenMP threads), 16 GB RAM.
- **Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
- **Software Stack:** Ubuntu / Windows Subsystem; CUDA 12.8, Driver 572.16; Python 3.10; PyTorch 2.5.1+cu124; Tectonic 0.15.0 (XeTeX-compatible engine).
- **Embedded Simulation Toolchain:** `arm-none-eabi-gcc` 12.3.1 (`-mcpu=cortex-m4 -mthumb -O3`); Unicorn Engine 2.0.1.

---

## 5. Corrected UCI HAR Protocol Re-Run and Parity Gate Provenance

### A. Parity Gate Evaluation
- **Execution Timestamp:** 2026-10-24 16:59:10 UTC (22:29:10 IST).
- **Compute Hardware:** AMD Ryzen 7 7840HS w/ Radeon 780M Graphics (16 logical threads), NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Driver 572.70, CUDA 12.8 runtime).
- **Environment:** Windows 11, Python 3.12.9, PyTorch 2.6.0+cu124.
- **Pre-registered Tolerances:** PTQ within $\le 0.05$ percentage points; Fine-tuning within $\le 0.30$ percentage points.
- **Gate Results:**
  - **Speech Commands (DS-CNN, Seed 42, DP 50%):**
    - PTQ: Stored 91.80%, Recomputed 91.80% ($\Delta = 0.0000$ pp, VERIFIED).
    - Fine-Tuned: Stored 91.67%, Recomputed 91.67% ($\Delta = 0.0000$ pp, VERIFIED).
  - **Fashion-MNIST (fmnist_cnn, Seed 42, DP 50%):**
    - PTQ: Stored 89.26%, Recomputed 89.26% ($\Delta = 0.0000$ pp, VERIFIED).
    - Fine-Tuned: Stored 90.85%, Recomputed 91.30% ($\Delta = +0.4500$ pp; exceeds 0.30 pp gate threshold due to cross-host fine-tuning variation). Gate exceeded threshold; no decision to continue after the parity-gate failure was recorded.

### B. Corrected UCI HAR 160-Row Re-Run
- **Job Execution Window:** 2026-10-24 17:28:34 UTC to 2026-10-24 17:32:59 UTC (22:58:34 to 23:02:59 IST).
- **Wall-Clock Duration:** 264.9 seconds (4.41 minutes).
- **Compute Provenance:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4 runtime, Driver 572.70, PyTorch 2.6.0+cu124), AMD Ryzen 7 7840HS.
- **Resource Utilization:** Peak RAM 180.8 MB, Peak VRAM 54.6 MB.
- **Output Artifact:** `results/ext_har_corrected.jsonl` (SHA-256 `d876a31065ceaf08365ef5713459c070f80bc4fc577fb797c21ae9b431102919`).
- **Row Audit:** Exactly 160 rows, 160 unique keys, 0 duplicate configurations, 0 endpoint layer violations ($b[0] = 8$ and $b[L-1] = 8$ strictly enforced on all 160 rows).
- **Execution Log:** `logs/ext/har_corrected.log` (Exit code 0, `[2026-10-24T17:32:59Z] END OF RUN (ALL 160 RUNS COMPLETE)`).


### Historical Note on Matrix Runner Version
The frozen results store (`results/runs_multiseed.jsonl`, 800 rows) was generated using an early development version of `scripts/run_matrix.py` that was not under formal version control prior to commit `b4a460b`. Known differences between the historical runner and the tracked version include: (1) UCI HAR endpoint index specification (historical runner used fixed index `[0, 5]` and `num_layers = 6`, whereas the tracked version programmatically derives `[0, L - 1]`), and (2) state dictionary checkpoint loading normalization across PyTorch module prefixes.


## Superseded Rows (HAR v2 Degenerate HAWQ)

The 40 rows in `results/ext_har_corrected_v2.jsonl` where `method == 'hawq'` used fallback traces for the three 1D convolutional layers due to an unaddressed dimension condition in the legacy trace estimator. These 40 rows are superseded by the 40 rows in `results/ext_har_hawq_fixed.jsonl` (Protocol ID `hawq-trace-v1`, committed as `2e4169b`), which implement full parameter coverage across all Conv1d and linear layers.
The 40 degenerate rows are excluded from `results/runs_multiseed_v3.jsonl`.
