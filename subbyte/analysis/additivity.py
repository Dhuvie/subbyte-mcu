"""
Additivity Analysis Suite.

Tests whether joint multi-layer accuracy drops can be predicted by the sum
of isolated single-layer accuracy drops:
    Predicted Drop = sum_{l} DeltaAcc_l(b_l)

Evaluates 200 random allocations on validation split, computes Spearman rho,
MAE (both raw and clipped to [0, FP32 - chance]), and verifies against the
specified hypothesis criterion: rho >= 0.80 AND MAE <= 1.0% at budget >= 35%.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def analyze_additivity_data(
    df: pd.DataFrame,
    chance_acc: float = 10.0,
    fp32_val_acc: float = 85.68,
    min_avg_bits: float = 3.9
) -> Dict[str, Any]:
    """Analyzes additivity dataset with both raw and clipped accuracy drops."""
    # Filter to budget >= 35% (avg_bits >= min_avg_bits)
    filtered_df = df[df["avg_bits"] >= min_avg_bits].copy()
    
    pred_raw = filtered_df["predicted_drop"].values
    act_raw = filtered_df["actual_drop"].values
    
    # Raw correlation and MAE
    rho_raw, p_raw = spearmanr(pred_raw, act_raw)
    mae_raw = float(np.mean(np.abs(pred_raw - act_raw)))
    
    # Max feasible drop from FP32 down to random chance
    max_drop = max(0.0, fp32_val_acc - chance_acc)
    pred_clipped = np.clip(pred_raw, 0.0, max_drop)
    act_clipped = np.clip(act_raw, 0.0, max_drop)
    
    rho_clipped, p_clipped = spearmanr(pred_clipped, act_clipped)
    mae_clipped = float(np.mean(np.abs(pred_clipped - act_clipped)))
    
    # Exact hypothesis evaluation: rho >= 0.80 AND MAE <= 1.0%
    passed = bool(rho_clipped >= 0.80 and mae_clipped <= 1.0)
    support = "full" if passed else ("partial" if rho_clipped >= 0.80 else "none")
    
    return {
        "dataset": "cifar10",
        "model": "resnet8",
        "num_allocations": len(filtered_df),
        "min_avg_bits": float(min_avg_bits),
        "max_avg_bits": float(filtered_df["avg_bits"].max()),
        "spearman_rho_raw": float(rho_raw),
        "spearman_p_raw": float(p_raw),
        "mae_drop_raw": float(mae_raw),
        "spearman_rho_clipped": float(rho_clipped),
        "spearman_p_clipped": float(p_clipped),
        "mae_drop_clipped": float(mae_clipped),
        "spearman_rho": round(float(rho_clipped), 3),
        "spearman_p": float(f"{p_clipped:.2e}"),
        "mae_drop": round(float(mae_clipped), 2),
        "passed_h7": passed,
        "support": support
    }
