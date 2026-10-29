# Sub-Byte Mixed-Precision Quantization for Microcontrollers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Target Architecture](https://img.shields.io/badge/Target-ARM%20Cortex--M4-0091BD.svg)](https://developer.arm.com/Processors/Cortex-M4)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified%20(8%2F8%20Checks)-brightgreen.svg)](scripts/check.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1437810900.svg)](https://doi.org/10.5281/zenodo.23750996)

This repository contains the standalone open-source implementation, instruction-level hardware cost model, and experimental replication suite for the paper:

> **"Is Sub-Byte Weight Quantization Worth It on Microcontrollers? Measured Sensitivity, Allocation Algorithms, and a Hardware Cost Model"**  
> *Dhruv Narayan Bajaj*  
> *Independent Researcher* (thedhruvbajaj@users.noreply.github.com)

---

## Methodological & Hardware Honesty Disclosure

- **Simulation-Only Hardware Cost Model:** All execution latency, instruction traces, and cycle counts in this repository derive from instruction-level simulation (Unicorn CPU emulation executing verified Thumb-2 assembly kernels and an analytical model grounded in the *ARM Cortex-M4 Technical Reference Manual* ARM DDI 0439D Issue D Table 3-1). No physical microcontroller silicon, hardware timers, or oscilloscope measurements were evaluated.
- **Conditional Pareto Dominance:** Conclusions regarding the Pareto dominance of clean power-of-two bitwidths ($\mathcal{B}_{\text{dep}} = \{2, 4, 8\}$) over arbitrary sub-byte configurations follow conditionally from standard 32-bit Cortex-M ALUs lacking native bitfield extraction instructions and the specific software unpacking routines implemented.
- **Headline result (null):** Dynamic programming knapsack allocation over empirical validation sensitivity tables is 0.06 pp below calibrated second-order perturbation allocation (HAWQ-$\Omega$ with 400 probe vectors) after 5 epochs of fine-tuning at 50% budget (mean delta $-0.06$ pp, 95% BCa CI $[-0.16, -0.00]$, exact sign-flip $p = 0.2500$ across $n = 5$ architectures), an empirical null result below per-model detectable noise. In zero-shot post-training quantization (PTQ) at 50% and 75% budgets, second-order HAWQ-$\Omega$ is higher than DP by $0.78$ pp on average (driven by two models; interval $[-2.22, 0.02]$) and $0.39$ pp on average (interval $[-1.09, -0.08]$, excluding zero). Both allocators outperform an uncalibrated single-probe legacy baseline of our own construction (by $+0.84$ points at 50% budget, interval $[0.22, 1.79]$). The primary practical advantage of empirical profiling is completing sensitivity calibration $5\times$ to $24\times$ faster without computing second-order gradients or Hessian-vector products.
- **Statistical Sample Size Bounds:** With $n = 5$ benchmark architectures treated as the primary unit of analysis to prevent pseudo-replication, the minimal achievable two-sided permutation $p$-value is $p = 2/32 = 0.0625$. Subtle cross-method accuracy differences ($\le 0.5$ pp) cannot attain formal statistical significance at $\alpha = 0.05$.

---

## Repository Structure

```
subbyte-mcu/
├── .gitattributes              # Binary and LF line-ending specifications
├── .gitignore                  # Git ignore rules for build and cache files
├── CITATION.cff                # Academic citation metadata (CFF 1.2.0)
├── DATA.md                     # Dataset citations, sources, and licenses
├── LICENSE                     # MIT License
├── Makefile                    # Automation targets (check, reproduce, test, paper)
├── README.md                   # Project documentation and reproduction guide
├── SHA256SUMS                  # Root cryptographic checksum manifest
├── requirements.txt            # Pinned Python package dependencies
├── paper/                      # Publication manuscript source, figures, tables
│   ├── paper.tex               # IEEE Access manuscript LaTeX source
│   ├── numbers.tex             # LaTeX macros for empirical quantities
│   ├── refs.bib                # Verified academic bibliography
│   ├── claims.yaml             # Empirical claim assertions
│   ├── figures/                # Publication vector PDF and PNG figures
│   └── tables/                 # Publication LaTeX tables (9 generated tables)
├── results/                    # Multi-seed experimental stores and checkpoints
│   ├── SHA256SUMS              # Results directory cryptographic manifest
│   ├── runs_multiseed_v3.jsonl # Primary 1,100-run multi-seed benchmark store
│   ├── runs_multiseed.jsonl    # Frozen 800-run baseline store
│   ├── ext_hawq_omega.jsonl    # 300 HAWQ-Omega 400-probe comparator runs
│   ├── checkpoints/            # 25 trained FP32 PyTorch baseline checkpoints
│   ├── complete_matrix.json    # Complete matrix of evaluated configurations
│   ├── cost_model_sweep.json   # 888-point analytical sensitivity parameter sweep
│   ├── hardware_trace_costs.json # Cycle-accurate instruction emulation traces
│   ├── hypotheses_evaluation.json# Statistical test results for hypotheses H1-H7
│   ├── model_level_tests.json  # Architecture-level paired bootstrap tests
│   ├── network_level_costs.json# Per-layer unpacked Flash/RAM memory footprint
│   └── stage4_headline_stats.json# Headline empirical null statistics
├── scripts/                    # Reproduction, analysis, and validation scripts
│   ├── check.py                # Comprehensive 8-gate reproduction verification suite
│   ├── make_all.py             # Master pipeline regenerating tables and figures
│   ├── plot_figures.py         # Vector graphics figure generation
│   ├── generate_tables.py      # LaTeX table and numbers.tex generation
│   └── run_anova_multiseed.py  # Multi-factor ANOVA variance decomposition
└── subbyte/                    # Core library package
    ├── allocation/             # DP, greedy, HAWQ, Fisher knapsack allocators
    ├── cost_model/             # Cortex-M4 cycle model and Thumb-2 C kernels
    ├── datasets/               # Dataset loaders (CIFAR-10, F-MNIST, Speech, HAR)
    ├── models/                 # Model topologies (ResNet-8, SeparableCNN, DSCNN, etc.)
    ├── quant/                  # Fake-quantization, bitpacking, .qpack format
    └── training/               # Fine-tuning, evaluation, seed management
```

---

## Quickstart & Replication

### 1. Environment Setup
Requires Python 3.10+ (tested on Python 3.10.12 and 3.12.3).
```bash
# Clone the repository
git clone https://github.com/Dhuvie/subbyte-mcu.git
cd subbyte-mcu

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Result Reproducibility (Non-GPU Verification)
To verify all results, cryptographic checksums, ANOVA statistics, and claims without requiring a GPU (~15 seconds):
```bash
python scripts/check.py
```
This suite verifies:
1. All 41 result files and model checkpoints against `results/SHA256SUMS`.
2. Agreement of `README.md` printed hashes with `results/SHA256SUMS`.
3. Byte-for-byte regeneration of `paper/numbers.tex` and all 9 LaTeX tables.
4. Completeness and validity of 387 empirical macros.
5. Exact consistency of all numerical claims in `paper/claims.yaml`.
6. Strict absence of unwhitelisted hard-coded numbers in `paper/paper.tex` and generated tables.
7. Independent re-derivation of all multi-factor ANOVA variance decompositions.
8. Bit-packing reference implementation parity against compiled C kernels.

### 3. Regenerate Figures and Tables
To regenerate all LaTeX tables, macros, and publication figures from the raw results store:
```bash
python scripts/make_all.py
```

---

## Building the Paper Manuscript

The manuscript source is in `paper/paper.tex`, with macros in `paper/numbers.tex` and citations in `paper/refs.bib`.

To compile the paper PDF:
1. Download the official **IEEE Access LaTeX Template** from IEEE (https://ieeeaccess.ieee.org/) or the IEEE Template Selector.
2. Place the required template files (`ieeeaccess.cls`, `IEEEtran.cls`, `IEEEtran.bst`, `spotcolor.sty`, font files `t1-*.pfb`, `*.tfm`, `*.map`, `*.fd`, and logo files `logo.png`, `notaglinelogo.png`, `bullet.png`) into the `paper/` directory.
3. Compile using `make paper` or:
   ```bash
   cd paper
   pdflatex -interaction=nonstopmode paper.tex
   bibtex paper
   pdflatex -interaction=nonstopmode paper.tex
   pdflatex -interaction=nonstopmode paper.tex
   ```

*Note on Preprint PDF:* A standalone preprint PDF will be linked here upon public posting.

---

## Hardware Cost Model and Instruction Tracing

The hardware cost model evaluates execution cycles for sub-byte and mixed-precision convolution kernels under the ARMv7-M architecture (ARM Cortex-M4):

### 1. Cross-Compilation of Thumb-2 Reference Kernels
The reference C unpack and MAC kernels (`subbyte/cost_model/kernels.c`) can be cross-compiled to a freestanding Thumb-2 object using `zig` or `arm-none-eabi-gcc`:

```bash
# Using Zig (freestanding Thumb-2 toolchain)
zig cc -target thumb-freestanding-eabihf -mcpu=cortex_m4 -O2 -c subbyte/cost_model/kernels.c -o subbyte/cost_model/kernels.o

# Using Python ziglang package
python -m ziglang cc -target thumb-freestanding-eabihf -mcpu=cortex_m4 -O2 -c subbyte/cost_model/kernels.c -o subbyte/cost_model/kernels.o

# Or using GNU Arm Embedded Toolchain
arm-none-eabi-gcc -mcpu=cortex_m4 -mthumb -O2 -c subbyte/cost_model/kernels.c -o subbyte/cost_model/kernels.o
```

### 2. Unicorn Emulation Instruction Tracing
Instruction traces and cycle counts are evaluated by running the compiled kernels or Python reference wrappers inside Unicorn CPU emulation:
```bash
python subbyte/cost_model/trace_cost_model.py
```

### 3. Instruction Cycle Summary (Measured Traces & Analytical Model)

Instruction-level execution costs grounded in measured Unicorn CPU emulator traces (`results/hardware_trace_costs.json`) and analytical cycle counts from ARM Cortex-M4 TRM DDI 0439D Table 3-1:

| Kernel Variant | Path | Trace Inst / 128 MAC | Trace Cyc / MAC | Model Cyc / MAC (ARM TRM Table 3-1) | Penalty vs INT8 (Like-for-Like) | Flash Savings |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **INT8 Baseline (CMSIS-NN Style)** | SIMD | 381 | **3.38** | 1.50-2.00 | 1.00x (SIMD Baseline) | 0.0% |
| **INT8 Scalar Baseline** | Scalar | 428 | **3.88** | 2.00 | 1.00x (Scalar Baseline) | 0.0% |
| **INT4 Optimised SIMD (SXTB16)** | SIMD | 608 | **5.28** | 1.75 | 1.56x (vs INT8 SIMD) | 50.0% |
| **INT4 Naive Scalar (SBFX)** | Scalar | 527 | **4.52** | 2.88 | 1.16x (vs INT8 Scalar) | 50.0% |
| **INT2 Optimised SIMD (SXTB16)** | SIMD | 565 | **4.51** | 1.56 | 1.33x (vs INT8 SIMD) | 75.0% |
| **INT2 Naive Scalar (SBFX)** | Scalar | 469 | **3.82** | 2.69 | 0.98x (vs INT8 Scalar) | 75.0% |
| **INT3 Aligned Unpack + SMLAD** | SIMD | 817 | **7.16** | 3.00 | 2.12x (vs INT8 SIMD) | 62.5% |
| **INT3 Scalar Bitfield Extract** | Scalar | 1430 | **12.23** | 4.00 | 3.15x (vs INT8 Scalar) | 62.5% |
| **INT5 Sub-Byte Unaligned Scalar** | Scalar | 1526 | **13.23** | 4.25 | 3.41x (vs INT8 Scalar) | 37.5% |
| **INT6 Semi-Aligned Scalar** | Scalar | 1440 | **12.56** | 4.25 | 3.24x (vs INT8 Scalar) | 25.0% |
| **INT7 Sub-Byte Unaligned Scalar** | Scalar | 1527 | **13.24** | 4.50 | 3.41x (vs INT8 Scalar) | 12.5% |

---

## Benchmark Datasets & Acquisition

All evaluations operate across five benchmark topologies spanning vision, speech, and sensor modalities:

1. **CIFAR-10** (Vision, 10 classes):
   - Topology: Depthwise Separable CNN (497.9 KiB INT8) and ResNet-8 (84.8 KiB INT8).
   - Source: Alex Krizhevsky, 2009.
   - Acquisition: Automatically downloaded via `torchvision.datasets.CIFAR10(root="data", download=True)`.
   - License: MIT / Open Access.

2. **Fashion-MNIST** (Vision, 10 classes):
   - Topology: 4-layer 2D-CNN (43.8 KiB INT8).
   - Source: Han Xiao, Kashif Rasul, Roland Vollgraf, 2017.
   - Acquisition: Automatically downloaded via `torchvision.datasets.FashionMNIST(root="data", download=True)`.
   - License: MIT License (Zalando SE).

3. **Google Speech Commands v2** (Audio, 12 classes):
   - Topology: DS-CNN (Hello Edge keyword spotting, 24.6 KiB INT8).
   - Source: Pete Warden, 2018.
   - Acquisition: Downloadable via `torchaudio.datasets.SPEECHCOMMANDS(root="data", download=True)` or Google Research Speech Commands repository.
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0).

4. **UCI Human Activity Recognition** (Sensors, 6 classes):
   - Topology: 1D-CNN (47.9 KiB INT8).
   - Source: Davide Anguita et al., ESANN 2013.
   - Acquisition: UCI Machine Learning Repository (Smartphone dataset ID 240).
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0).

### Test-Set Quarantine Protocol
All datasets are partitioned into 80% training, 10% validation (for sensitivity profiling and knapsack optimization), and 10% held-out test splits. The test split is quarantined and evaluated strictly once on the final packed `.qpack` container artifact.

---

## Results Directory Layout & Verification

All results are cryptographically tracked in `results/SHA256SUMS`:

- `results/runs_multiseed_v3.jsonl` (SHA-256: `23955704dc1cb8fe7818b99d25d62fad12f5b85f1b2ab3e26f85a59b73870be2`): Complete 1,100-run multi-seed benchmark store spanning 5 models, 4 budgets, 6 allocation methods (`dp_measured`, `fisher_omega_dp`, `greedy`, `hawq`, `hawq_omega_dp`, `uniform_w4_endpoints`), and 5 seeds.
- `results/runs_multiseed.jsonl` (SHA-256: `744accf9f4b0bcfae39746373989f024d69f70ea45e042c7497c099d35ee728d`): Frozen historical baseline 800-run store.
- `results/ext_hawq_omega.jsonl` (SHA-256: `15abc9ac00736801dbb4555afe369027e0c04a446ec809e12b909d90751a087b`): 300 HAWQ-$\Omega$ 400-probe comparator runs across seeds.
- `results/checkpoints/`: 25 trained baseline FP32 model checkpoints (`.pt` files, 5 models $\times$ 5 seeds, total size 15.77 MB).

---

## Citation

If you use this software, hardware cost model, or empirical benchmark data, please cite this repository:

```bibtex
@misc{bajaj2026subbyte,
  title={Is Sub-Byte Weight Quantization Worth It on Microcontrollers? Measured Sensitivity, Allocation Algorithms, and a Hardware Cost Model},
  author={Bajaj, Dhruv Narayan},
  year={2026},
  howpublished={GitHub repository},
  note={Version 1.0}
}
```

Complete citation metadata is also provided in [CITATION.cff](CITATION.cff).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
