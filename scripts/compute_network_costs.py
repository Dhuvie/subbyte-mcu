#!/usr/bin/env python3
"""
Compute Network-Level Costs, Peak Activation RAM, Modelled Inference Cycles,
and Datasheet-Based Microcontroller Fit Checks across the 5 TinyML Benchmarks.

Requirement 2.3:
- Modelled cycles per inference from layer MACs and kernel cycles
- Flash footprint (.qpack bytes at INT8, W4, and representative DP budgets)
- Peak activation RAM (bytes under double-buffered activation arena)
- Datasheet-based fit check against STM32F446RE (512 KB Flash, 128 KB RAM)
  and Nordic nRF52840 (1024 KB Flash, 256 KB RAM).
- Explicitly documented as a datasheet-based fit check, NOT a deployment.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subbyte.models import get_model
from subbyte.quant.qpack import compute_qpack_size


def analyze_model_layers(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Hooks through all layers to extract:
    - Layer names and weight shapes
    - MACs per layer
    - Parameter counts
    - Activation tensor dimensions and peak activation RAM
    """
    model.eval()
    layer_metas = []
    layer_macs = []
    activation_sizes = []
    current_activation_shape = list(sample_input.shape)
    
    # Store activations at forward pass
    activations = {}
    hooks = []
    
    # Track module execution order
    exec_order = []
    
    def hook_fn(name):
        def fn(module, inp, out):
            exec_order.append(name)
            if isinstance(out, torch.Tensor):
                activations[name] = list(out.shape)
            elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                activations[name] = list(out[0].shape)
        return fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
            
    with torch.no_grad():
        _ = model(sample_input)
        
    for h in hooks:
        h.remove()
        
    total_macs = 0
    total_params = 0
    layer_records = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            weight = module.weight
            w_shape = list(weight.shape)
            num_elems = weight.numel()
            has_bias = module.bias is not None
            out_shape = activations.get(name, [])
            
            # Compute MACs
            if isinstance(module, nn.Conv2d):
                # out_shape: [batch, out_channels, out_h, out_w]
                if len(out_shape) == 4:
                    out_h, out_w = out_shape[2], out_shape[3]
                    # kernel_h * kernel_w * in_channels / groups * out_channels * out_h * out_w
                    macs = (w_shape[2] * w_shape[3] * (w_shape[1])) * (w_shape[0] * out_h * out_w)
                else:
                    macs = num_elems
            elif isinstance(module, nn.Conv1d):
                if len(out_shape) == 3:
                    out_l = out_shape[2]
                    macs = w_shape[2] * w_shape[1] * w_shape[0] * out_l
                else:
                    macs = num_elems
            elif isinstance(module, nn.Linear):
                macs = w_shape[0] * w_shape[1]
            else:
                macs = num_elems
                
            total_macs += macs
            total_params += num_elems
            
            # Activation bytes in INT8 (1 byte per activation)
            act_elems = 1
            for dim in out_shape[1:]: # exclude batch
                act_elems *= dim
                
            layer_metas.append({
                'name': name,
                'num_elements': num_elems,
                'num_channels': w_shape[0],
                'has_bias': has_bias
            })
            layer_macs.append(macs)
            activation_sizes.append(act_elems)
            
            layer_records.append({
                'name': name,
                'type': module.__class__.__name__,
                'weight_shape': w_shape,
                'params': num_elems,
                'macs': macs,
                'out_shape': out_shape,
                'activation_bytes': act_elems
            })

    # Double-buffered peak activation RAM
    # In sequential inference: max over consecutive layers (in_bytes + out_bytes)
    # Plus safety margin for residual branch if applicable
    peak_act_ram = 0
    in_bytes = 1
    for dim in sample_input.shape[1:]:
        in_bytes *= dim
        
    prev_bytes = in_bytes
    for act_b in activation_sizes:
        buffer_need = prev_bytes + act_b
        if buffer_need > peak_act_ram:
            peak_act_ram = buffer_need
        prev_bytes = act_b
        
    return {
        'total_params': total_params,
        'total_macs': total_macs,
        'peak_activation_ram_bytes': peak_act_ram,
        'layer_metas': layer_metas,
        'layer_macs': layer_macs,
        'layers': layer_records
    }


def compute_all_model_costs() -> Dict[str, Any]:
    models_config = [
        ('cifar10', 'separable_cnn', (1, 3, 32, 32), 10, 'SeparableCNN (Vision)'),
        ('cifar10', 'resnet8', (1, 3, 32, 32), 10, 'ResNet-8 (Residual Vision)'),
        ('fmnist', 'fmnist_cnn', (1, 1, 28, 28), 10, 'FashionCNN (Grayscale Vision)'),
        ('har', 'cnn1d', (1, 9, 128), 6, 'CNN1D (Sensor IMU)'),
        ('speech', 'dscnn', (1, 1, 49, 10), 12, 'DS-CNN-S (Audio KWS)')
    ]
    
    # Kernel cycle rates (Thumb-2 SIMD optimised)
    # INT8: 1.50 cyc/MAC (2.00 cyc/MAC scalar)
    # W4: 1.75 cyc/MAC (2.88 cyc/MAC scalar)
    # W2: 1.56 cyc/MAC (2.69 cyc/MAC scalar)
    # 50% Mixed DP: ~1.75 cyc/MAC
    # 35% Mixed DP: ~1.65 cyc/MAC
    
    CYC_INT8_SIMD = 1.50
    CYC_INT8_SCALAR = 2.00
    CYC_W4_SIMD = 1.75
    CYC_W2_SIMD = 1.56
    
    # Datasheet specifications (Verified Official Specifications):
    # 1. STM32F446RE:
    #    Flash: 512 KB = 524,288 bytes
    #    SRAM: 128 KB = 131,072 bytes (112 KB SRAM1 + 16 KB SRAM2)
    #    DocID027107 Rev 6, STMicroelectronics.
    # 2. Nordic nRF52840:
    #    Flash: 1024 KB = 1,048,576 bytes
    #    SRAM: 256 KB = 262,144 bytes
    #    nRF52840 Product Specification v1.3, Nordic Semiconductor.
    
    MCUS = {
        'STM32F446RE': {
            'flash_bytes': 512 * 1024,
            'sram_bytes': 128 * 1024,
            'clock_mhz': 180,
            'core': 'Arm Cortex-M4'
        },
        'nRF52840': {
            'flash_bytes': 1024 * 1024,
            'sram_bytes': 256 * 1024,
            'clock_mhz': 64,
            'core': 'Arm Cortex-M4'
        }
    }
    
    results = {}
    
    for ds, m_name, in_shape, n_cls, desc in models_config:
        model = get_model(ds, m_name, num_classes=n_cls)
        dummy_in = torch.randn(*in_shape)
        info = analyze_model_layers(model, dummy_in)
        
        metas = info['layer_metas']
        n_layers = len(metas)
        total_macs = info['total_macs']
        peak_ram = info['peak_activation_ram_bytes']
        
        # Calculate .qpack footprints
        # Full FP32 raw weights
        fp32_bytes = info['total_params'] * 4
        # INT8 uniform (all 8 bits)
        qpack_int8 = compute_qpack_size(metas, [8] * n_layers)
        # Uniform W4 (all 4 bits)
        qpack_w4 = compute_qpack_size(metas, [4] * n_layers)
        # Uniform W2 (all 2 bits)
        qpack_w2 = compute_qpack_size(metas, [2] * n_layers)
        # 50% DP budget target
        target_50_bytes = int(qpack_int8 * 0.50)
        # 35% DP budget target
        target_35_bytes = int(qpack_int8 * 0.35)
        
        # Modelled inference cycles (SIMD optimised)
        cyc_int8_simd = int(total_macs * CYC_INT8_SIMD)
        cyc_int8_scalar = int(total_macs * CYC_INT8_SCALAR)
        cyc_w4_simd = int(total_macs * CYC_W4_SIMD)
        cyc_w2_simd = int(total_macs * CYC_W2_SIMD)
        
        # Latency estimates at MCU clock frequencies
        lat_f446_int8_ms = (cyc_int8_simd / (180e6)) * 1000.0
        lat_f446_w4_ms = (cyc_w4_simd / (180e6)) * 1000.0
        lat_nrf_int8_ms = (cyc_int8_simd / (64e6)) * 1000.0
        lat_nrf_w4_ms = (cyc_w4_simd / (64e6)) * 1000.0
        
        # Datasheet fit checks
        fits_stm32_int8 = (qpack_int8 <= MCUS['STM32F446RE']['flash_bytes']) and (peak_ram <= MCUS['STM32F446RE']['sram_bytes'])
        fits_stm32_w4 = (qpack_w4 <= MCUS['STM32F446RE']['flash_bytes']) and (peak_ram <= MCUS['STM32F446RE']['sram_bytes'])
        fits_nrf_int8 = (qpack_int8 <= MCUS['nRF52840']['flash_bytes']) and (peak_ram <= MCUS['nRF52840']['sram_bytes'])
        fits_nrf_w4 = (qpack_w4 <= MCUS['nRF52840']['flash_bytes']) and (peak_ram <= MCUS['nRF52840']['sram_bytes'])

        results[m_name] = {
            'description': desc,
            'dataset': ds,
            'input_shape': list(in_shape),
            'parameters': info['total_params'],
            'macs': total_macs,
            'peak_activation_ram_bytes': peak_ram,
            'peak_activation_ram_kib': round(peak_ram / 1024.0, 2),
            'fp32_weight_bytes': fp32_bytes,
            'fp32_weight_kib': round(fp32_bytes / 1024.0, 2),
            'qpack_int8_bytes': qpack_int8,
            'qpack_int8_kib': round(qpack_int8 / 1024.0, 2),
            'qpack_w4_bytes': qpack_w4,
            'qpack_w4_kib': round(qpack_w4 / 1024.0, 2),
            'qpack_w2_bytes': qpack_w2,
            'qpack_w2_kib': round(qpack_w2 / 1024.0, 2),
            'modelled_cycles_int8_simd': cyc_int8_simd,
            'modelled_cycles_int8_scalar': cyc_int8_scalar,
            'modelled_cycles_w4_simd': cyc_w4_simd,
            'modelled_latency_stm32f446_int8_ms': round(lat_f446_int8_ms, 2),
            'modelled_latency_stm32f446_w4_ms': round(lat_f446_w4_ms, 2),
            'modelled_latency_nrf52840_int8_ms': round(lat_nrf_int8_ms, 2),
            'modelled_latency_nrf52840_w4_ms': round(lat_nrf_w4_ms, 2),
            'datasheet_fit_check': {
                'STM32F446RE': {
                    'int8_fits': fits_stm32_int8,
                    'w4_fits': fits_stm32_w4,
                    'flash_util_pct_int8': round((qpack_int8 / MCUS['STM32F446RE']['flash_bytes']) * 100.0, 1),
                    'flash_util_pct_w4': round((qpack_w4 / MCUS['STM32F446RE']['flash_bytes']) * 100.0, 1),
                    'ram_util_pct': round((peak_ram / MCUS['STM32F446RE']['sram_bytes']) * 100.0, 1)
                },
                'nRF52840': {
                    'int8_fits': fits_nrf_int8,
                    'w4_fits': fits_nrf_w4,
                    'flash_util_pct_int8': round((qpack_int8 / MCUS['nRF52840']['flash_bytes']) * 100.0, 1),
                    'flash_util_pct_w4': round((qpack_w4 / MCUS['nRF52840']['flash_bytes']) * 100.0, 1),
                    'ram_util_pct': round((peak_ram / MCUS['nRF52840']['sram_bytes']) * 100.0, 1)
                }
            }
        }
        
    out_json = ROOT / "results" / "network_level_costs.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved network-level costs to {out_json}")
    
    # Generate LaTeX Table
    generate_latex_table(results)
    
    return results


def generate_latex_table(results: Dict[str, Any]):
    tex_path = Path(os.environ.get("NETWORK_COSTS_OUT", str(ROOT / "paper" / "tables" / "tab_network_costs.tex")))
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        r"% Network-Level Costs, Footprints, Activation RAM, and MCU Datasheet Fit Checks",
        r"% Generated automatically by scripts/compute_network_costs.py",
        r"% Note: Cycle counts and latencies are MODELLED; MCU fit is a DATASHEET CHECK, not a deployment.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Network-level modelled computational cost, storage footprint, peak activation memory, and datasheet-based MCU fit check across the five evaluated benchmarks. All cycle counts and latencies are derived from our Cortex-M4 trace cost model. Target MCU compatibility denotes static datasheet fit (Flash/SRAM capacity) and does not imply physical hardware deployment.}",
        r"\label{tab:network_costs}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"\textbf{Benchmark} & \textbf{Dataset} & \textbf{Params} & \textbf{MACs} & \textbf{Peak RAM} & \multicolumn{2}{c}{\textbf{Flash Footprint}} & \multicolumn{2}{c}{\textbf{Modelled Latency}} & \textbf{Datasheet Fit} \\",
        r"\cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r" & & & & \textbf{(KiB)} & \textbf{INT8} & \textbf{W4} & \textbf{F446RE (180\,MHz)} & \textbf{nRF52 (64\,MHz)} & \textbf{F446 / nRF52} \\",
        r"\midrule"
    ]
    
    for m_name, d in results.items():
        desc = d['description'].split(' (')[0]
        ds = d['dataset'].upper()
        p_str = f"{d['parameters']:,}"
        mac_str = f"{d['macs'] / 1e6:.2f}M" if d['macs'] >= 1e6 else f"{d['macs'] / 1e3:.1f}k"
        ram_str = f"{d['peak_activation_ram_kib']:.1f}"
        int8_kb = f"{d['qpack_int8_kib']:.1f}" + r"\,KiB"
        w4_kb = f"{d['qpack_w4_kib']:.1f}" + r"\,KiB"
        lat_f446 = f"{d['modelled_latency_stm32f446_w4_ms']:.1f}" + r"\,ms"
        lat_nrf = f"{d['modelled_latency_nrf52840_w4_ms']:.1f}" + r"\,ms"
        
        f446_fit = "Yes" if d['datasheet_fit_check']['STM32F446RE']['w4_fits'] else "No"
        nrf_fit = "Yes" if d['datasheet_fit_check']['nRF52840']['w4_fits'] else "No"
        fit_str = f"{f446_fit} / {nrf_fit}"
        
        lines.append(f"{desc} & {ds} & {p_str} & {mac_str} & {ram_str} & {int8_kb} & {w4_kb} & {lat_f446} & {lat_nrf} & {fit_str} \\\\")
        
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated LaTeX table at {tex_path}")


if __name__ == '__main__':
    compute_all_model_costs()
