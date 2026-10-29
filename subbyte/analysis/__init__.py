"""
Analysis and statistical modeling package.
"""

from subbyte.analysis.stats import (
    bootstrap_ci,
    compute_cohens_d,
    paired_comparison,
    apply_holm_bonferroni
)
from subbyte.analysis.additivity import analyze_additivity_data
from subbyte.analysis.anova import get_anova_decomposition, format_latex_table
from subbyte.analysis.break_even import evaluate_break_even

__all__ = [
    "bootstrap_ci",
    "compute_cohens_d",
    "paired_comparison",
    "apply_holm_bonferroni",
    "analyze_additivity_data",
    "get_anova_decomposition",
    "format_latex_table",
    "evaluate_break_even"
]
