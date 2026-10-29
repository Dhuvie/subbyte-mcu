#!/usr/bin/env python3
"""
Generate Formal Hypotheses Table (H1-H7) for the Sub-Byte MCU Manuscript.
Requirement 2.6:
- Statements, tests, thresholds, observed values, and verdicts (Supported / Partially Supported / Not Supported).
- Plain language; negative and partial outcomes reported plainly.
- Does NOT use the banned word 'pre-registered'.
- Generates paper/tables/tab_hypotheses.tex and results/hypotheses_evaluation.json.
"""

import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = Path(os.environ.get("TABLES_OUT_DIR", str(ROOT / "paper" / "tables")))

HYPOTHESES = [
    {
        "id": "H1",
        "name": "Formula Score vs Measured Sensitivity",
        "statement": "Analytic second-order formula scores correlate strongly with empirical layer sensitivity drops measured on validation data.",
        "test": "Spearman rank correlation $\\rho$ between formula scores and empirical sensitivity drops.",
        "threshold": "$\\rho \\ge 0.70$, $p < 0.05$",
        "observed": r"$\rho = \MacroPartialRho$ (ResNet-8: $\MacroResNetRho$, Speech DS-CNN: $\MacroSpeechRho$, SeparableCNN: $\MacroSeparableCNNRho$)",
        "verdict": "Not Supported",
        "discussion": "Heuristic formula scores exhibit near-zero or negative correlation with true layer degradation on compact micro-architectures."
    },
    {
        "id": "H2",
        "name": "Error Additivity",
        "statement": "Joint multi-layer quantization degradation is accurately predicted by linear sum of per-layer sensitivity drops.",
        "test": "Spearman $\\rho$ and MAE between predicted and actual joint accuracy drops ($B \\ge 35\\%$).",
        "threshold": "$\\rho \\ge 0.80$ AND $\\text{MAE} \\le 1.0\\%$",
        "observed": r"$\rho = \MacroAdditivityRho$ ($p < 0.001$), but $\text{MAE} = \MacroAdditivityMAE\%$ (clipped to valid drop range)",
        "verdict": "Partially Supported",
        "discussion": "Relative ranking order is preserved, but absolute drop magnitude overestimates due to compounding inter-layer error interactions."
    },
    {
        "id": "H3",
        "name": "Measured-Sensitivity DP vs Allocation Baselines",
        "statement": "Measured-sensitivity dynamic programming knapsack allocation achieves higher top-1 accuracy than heuristic baselines under identical bit budgets.",
        "test": "Exact permutation test across model-level paired differences ($n=5$ models, $2^5=32$ signs) and paired seed-level sign test ($N=25$).",
        "threshold": "$p < 0.05$ across 4 budgets (Holm-Bonferroni corrected)",
        "observed": r"Against HAWQ-$\Omega$ (400 probe vectors) in PTQ at 50\% budget, HAWQ-$\Omega$ is $+\MacroPTQDPOmegaAbsMeanFifty$ pp higher than DP on average (CI $[\MacroPTQDPOmegaCILowFifty, +\MacroPTQDPOmegaCIHighFifty]$, driven by Speech DS-CNN and SeparableCNN; DP is equal or higher on the other three); at 75\% budget, HAWQ-$\Omega$ is $+\MacroPTQDPOmegaAbsMeanSeventyFive$ pp higher (CI $[\MacroPTQDPOmegaCILowSeventyFive, \MacroPTQDPOmegaCIHighSeventyFive]$, excluding zero). After fine-tuning at 50\% budget, mean delta is $\MacroOmegaMeanDelta$ pp (BCa 95\% CI $[\MacroOmegaCILow, \MacroOmegaCIHigh]$, exact $p = \MacroOmegaPermP$, \MacroOmegaWins/5 models positive), exhibiting a difference below the per-model detectable effect when solved with the same knapsack optimizer. Against legacy trace-rank heuristic, fine-tuned DP is $+\MacroLegacyMeanDelta$ pp higher (BCa 95\% CI $[+\MacroLegacyCILow, +\MacroLegacyCIHigh]$, \MacroLegacyWins\ of 5 models positive, exact $p = \MacroLegacyPermP$). At 35\% PTQ, exact knapsack methods prevent zero-shot collapse.",
        "verdict": "Supported against legacy heuristic on effect size; Not Supported against HAWQ-$\\Omega$",
        "discussion": r"In zero-shot PTQ at 50% budget, HAWQ-$\Omega$ exceeds DP by \MacroPTQDPOmegaAbsMeanFifty\ pp on average (driven by two models; at 75% budget by \MacroPTQDPOmegaAbsMeanSeventyFive\ pp, interval [\MacroPTQDPOmegaCILowSeventyFive, \MacroPTQDPOmegaCIHighSeventyFive], excluding zero), whereas after fine-tuning, DP and HAWQ-$\Omega$ achieve a difference below the per-model detectable effect (\MacroOmegaMeanDelta\ pp difference at 50% budget, exact p = \MacroOmegaPermP), while DP exceeds the legacy trace-rank heuristic (+\MacroLegacyMeanDelta\ pp at 50% budget, exact p = \MacroLegacyPermP)."
    },
    {
        "id": "H4",
        "name": "Regime Crossover (Collapse vs Plateau)",
        "statement": r"Accuracy retention exhibits a sharp collapse boundary as budget drops below $\sim$40\% (average bitwidth $\le 3.2$ bits).",
        "test": r"Piecewise comparison of PTQ drop between $\le 35\%$ budgets and $\ge 50\%$ budgets.",
        "threshold": r"Median PTQ accuracy drop from FP32 $> 20.0$ points at $\le 35\%$ vs $< 2.0$ points at $\ge 50\%$",
        "observed": r"Median PTQ drop: $\MacroPTQMedianDeltaThirtyFive$ points at 35\% budget (mean $\MacroPTQMeanDeltaThirtyFive$) vs $\MacroPTQMedianDeltaFifty$ points at 50\% (mean $\MacroPTQMeanDeltaFifty$) and $\MacroPTQMedianDeltaSeventyFive$ points at 75\% (mean $\MacroPTQMeanDeltaSeventyFive$). Both pre-fixed thresholds ($>20.0$ and $<2.0$) were narrowly missed.",
        "verdict": "Partially Supported",
        "discussion": r"Empirical evidence reveals a steep transition between zero-shot PTQ degradation at $\le 35\%$ budget and an intact plateau at $\ge 50\%$ budget, but the pre-fixed quantitative thresholds were narrowly missed (\MacroPTQMedianDeltaThirtyFive\ points vs 20.0 at 35%, \MacroPTQMedianDeltaFifty\ points vs 2.0 at 50%). Brief 5-epoch fine-tuning recovers 15--20 accuracy points."
    },
    {
        "id": "H5",
        "name": "Variance Decomposition (ANOVA with Interactions)",
        "statement": "Target bit budget accounts for the dominant share of accuracy variance, while allocation algorithm explains a minor fraction.",
        "test": r"Factorial ANOVA with interaction terms (Budget, Method, Method $\times$ Budget) across $N=60$ runs on CIFAR-10 / SeparableCNN only.",
        "threshold": r"Budget variance $> 70.0\%$; Method main effect $< 5.0\%$ in deployable plateau regime ($\ge 50\%$)",
        "observed": r"All budgets ($N=60$, on CIFAR-10 / SeparableCNN only): Budget \MacroVarBitsPct\% ($F=\MacroANOVAFStatBudget, p<0.001$), Method \MacroVarMethodPct\% ($F=\MacroANOVAFStatMethod, p<0.001$), Interaction \MacroVarMethodBudgetInteractPct\% ($F=\MacroANOVAFStatInteract, p<0.001$), Residual \MacroVarResidualPct\%. Plateau ($N=30$, on CIFAR-10 / SeparableCNN only): Method explains \MacroVarPlateauMethodPct\% ($F=\MacroANOVAPlateauFStatMethod, p<0.001$, with per-model mean differences between DP and HAWQ \MacroDPHAWQFiftyRange\ across architectures (\MacroMethodPlateauSepEffect\ points on SeparableCNN pooled across the plateau, versus \MacroSepCNNGainFifty\ points at 50\% budget)), while seed variation accounts for \MacroVarPlateauResidualPct\%.",
        "verdict": "Partially Supported",
        "discussion": r"The pre-fixed threshold for budget dominance across all budgets is satisfied (\MacroVarBitsPct\% > 70.0\%), but in the deployable plateau regime on CIFAR-10 / SeparableCNN only, allocation method accounts for \MacroVarPlateauMethodPct\% of variance ($F=\MacroANOVAPlateauFStatMethod, p<0.001$), failing the <5.0\% threshold. The practical difference between DP and HAWQ across models ranges \MacroDPHAWQFiftyRange\ across architectures (\MacroMethodPlateauSepEffect\ points on SeparableCNN pooled across the plateau, versus \MacroSepCNNGainFifty\ points at 50\% budget)."
    },
    {
        "id": "H6",
        "name": "Power-of-Two vs Arbitrary Bitwidths under Cost Model",
        "statement": r"Clean power-of-two bitwidths $\{2, 4, 8\}$ Pareto-dominate arbitrary sub-byte bitwidths $\{2..8\}$ in latency and throughput.",
        "test": "Instruction-level trace cost model on Cortex-M4 Thumb-2 kernels across wait states, load latency, and unpack penalties $p$.",
        "threshold": r"Break-even penalty $p^* \le 1.05\times$ (SIMD), and Pareto dominance in $\ge 95\%$ of swept configurations",
        "observed": r"Under the parametric model across all 888 swept configurations, break-even penalties are $p^* \le \MacroBreakEvenPenaltySIMD\times$ for CMSIS-NN SIMD and $p^* = \MacroBreakEvenPenalty\times$ for scalar execution. Measured instruction traces are: INT8 SIMD \MacroCyclesINTEightSIMDExact, INT4 SIMD \MacroCyclesFourBitSIMDExact, INT2 SIMD \MacroCyclesTwoBitSIMD, INT3 aligned SIMD \MacroCyclesINTThreeSIMD, and scalar odd widths \MacroCyclesScalarUnpackLow--\MacroCyclesScalarUnpackHigh\ cycles/MAC. Power-of-two formats reduce Flash footprint by 50\% (INT4) and 75\% (INT2) and avoid odd-width extraction penalties, but INT4 and INT2 SIMD kernels need \MacroCyclesRatioRangeSIMDLow--\MacroCyclesRatioRangeSIMDHigh\ times more cycles per MAC than INT8 SIMD.",
        "verdict": "Partially Supported",
        "discussion": r"Under the parametric model, clean power-of-two bitwidths {2, 4, 8} Pareto-dominate arbitrary sub-byte configurations whenever odd formats incur bitfield extraction penalties versus native execution. Power-of-two formats achieve 50% (INT4) and 75% (INT2) flash savings without odd-width extraction penalties, but measured traces show INT4 SIMD (\MacroCyclesFourBitSIMDExact\ cyc/MAC) and INT2 SIMD (\MacroCyclesTwoBitSIMD\ cyc/MAC) need \MacroCyclesRatioRangeSIMDLow--\MacroCyclesRatioRangeSIMDHigh\ times more cycles per MAC than INT8 SIMD (\MacroCyclesINTEightSIMDExact\ cyc/MAC)."
    },
    {
        "id": "H7",
        "name": "Generalisation Across Datasets",
        "statement": r"Two-regime behavior and measured allocation differences (DP minus greedy model means within $\pm \MacroDPGreedyPlateauMaxAbsDelta$ points, DP minus HAWQ $+\MacroStoryDPHAWQThirtyFive$ / $+\MacroStoryDPHAWQFifty$ / $+\MacroStoryDPHAWQSeventyFive$ points at $35$ / $50$ / $75\%$, minimum exact $p = 0.0625$ with $n = 5$) hold across distinct TinyML modalities.",
        "test": "Cross-dataset evaluation across 5 models spanning 4 datasets and 3 modalities across 5 seeds under matched protocols.",
        "threshold": r"Plateau retention ($<1.5\%$ drop from INT8 at 50\% budget) and paired cross-model difference bounds ($p > 0.05$ under permutation test)",
        "observed": r"Plateau retention within $\MacroPlateauRetentionDropMax\%$ of INT8 across all tasks (maximum drop over models). At 50\% budget, DP outperforms HAWQ on 4/5 models (mean delta $+\MacroStoryDPHAWQFifty\%$, Speech $+\MacroSpeechGainFifty\%$), with HAR delta $\MacroHARDeltasFifty\%$ at 50\% (overall HAR tier mean $+\MacroHARDeltasMean\%$). Uniform W4+Endpoints is not budget-matched at $\le 35\%$ (consumes $\MacroUniformWFourFootprintPct\%$ bytes); at 50\%, DP matches or exceeds W4 on 4/5 models (FMNIST DP \MacroFMNISTDPAccFifty\% vs W4 \MacroFMNISTWFourAccFifty\%).",
        "verdict": "Partially Supported",
        "discussion": r"The multi-domain evaluation demonstrates strong plateau retention across all five models, with domain-specific characteristics: DP outperforms HAWQ on Speech Commands (+\MacroSpeechGainFifty\%) and SeparableCNN (+\MacroSepCNNGainFifty\%), while across all tiers UCI HAR exhibits an average delta of +\MacroHARDeltasMean\%."
    }
]


def generate_hypotheses_artifacts():
    # 1. Save JSON
    out_json = RESULTS_DIR / "hypotheses_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(HYPOTHESES, f, indent=2)
    print(f"Saved hypotheses evaluation data to {out_json}")

    # 2. Save LaTeX Table
    out_tex = TABLES_DIR / "tab_hypotheses.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"% Formal Empirical Hypotheses Table (H1-H7)",
        r"% Generated automatically by scripts/generate_hypotheses_table.py",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Formal empirical hypotheses evaluation across experimental benchmarks and hardware cost models. Each hypothesis specifies its test, fixed evaluation threshold, observed outcome, and resulting empirical verdict (Supported / Partially Supported / Not Supported).}",
        r"\label{tab:hypotheses}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{l p{3.6cm} p{3.2cm} X p{2.4cm}}",
        r"\toprule",
        r"\textbf{ID} & \textbf{Statement} & \textbf{Evaluation Test \& Threshold} & \textbf{Observed Outcome} & \textbf{Verdict} \\",
        r"\midrule"
    ]

    for h in HYPOTHESES:
        h_id = h['id']
        name = h['name']
        stmt = h['statement']
        test_thresh = f"{h['test']} Threshold: {h['threshold']}"
        obs = h['observed']
        verd = h['verdict']

        verdict_str = f"\\textbf{{{verd}}}" if verd == "Supported" else (f"\\textit{{{verd}}}" if "Partially" in verd else f"\\textbf{{{verd}}}")

        lines.append(f"\\textbf{{{h_id}}} & \\textbf{{{name}}}: {stmt} & {test_thresh} & {obs} & {verdict_str} \\\\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabularx}",
        r"\end{table*}"
    ])

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated LaTeX hypotheses table at {out_tex}")


if __name__ == "__main__":
    generate_hypotheses_artifacts()
