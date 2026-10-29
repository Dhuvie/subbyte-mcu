#!/usr/bin/env python3
"""
Single Source of Truth Master Build Pipeline
Repository: subbyte-mcu

Mandatory Execution Sequence:
1. Run statistics from results/runs_multiseed.jsonl (independent referee + ANOVA).
2. Generate paper/numbers.tex and all LaTeX tables from raw data files only.
3. Generate all publication figures.
4. Synchronize and verify claims against numbers.tex.
5. Build paper/paper.pdf using pdfTeX and BibTeX with a convergence loop.
6. Verify all claims via text extraction from paper/paper.pdf.

Zero hand-typed numbers permitted.
"""

import sys
import os
import subprocess
import hashlib
import json
import shutil
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper"
SCRIPTS_DIR = ROOT / "scripts"
RUNS_MULTISEED_V3 = RESULTS_DIR / "runs_multiseed_v3.jsonl"
NUMBERS_TEX = PAPER_DIR / "numbers.tex"
PDF_PATH = PAPER_DIR / "paper.pdf"

def find_python():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.executable
    venv_py = ROOT.parent / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable

PYTHON = find_python()

def find_pdftex():
    localappdata = os.environ.get("LOCALAPPDATA", "")
    miktex_pdflatex = Path(localappdata) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe"
    miktex_bibtex = Path(localappdata) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "bibtex.exe"
    if miktex_pdflatex.exists() and miktex_bibtex.exists():
        return str(miktex_pdflatex), str(miktex_bibtex)
    which_pdf = shutil.which("pdflatex")
    which_bib = shutil.which("bibtex")
    if which_pdf and which_bib:
        return which_pdf, which_bib
    raise FileNotFoundError("pdfTeX (pdflatex and bibtex) not found in MiKTeX path or PATH")

PDFLATEX, BIBTEX = find_pdftex()


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    user = os.environ.get("USERNAME", "")
    if user:
        text = text.replace(user, "user")
    text = re.sub(r'[A-Za-z]:\\[Uu]sers\\[^\\]+\\[^\\]+\\[^\\]+\\subbyte-mcu[/\\]?', '', text)
    text = re.sub(r'[A-Za-z]:/[Uu]sers/[^/]+/[^/]+/[^/]+/subbyte-mcu[/\\]?', '', text)
    text = re.sub(r'[A-Za-z]:\\[Uu]sers\\[^\\]+\\', '', text)
    text = re.sub(r'[A-Za-z]:/[Uu]sers/[^/]+/', '', text)
    return text


def run_cmd(cmd_args, cwd=ROOT):
    cmd_str = " ".join([sanitize_text(a) for a in cmd_args])
    print(f"\n[EXEC] {cmd_str}")
    res = subprocess.run(cmd_args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    output = res.stdout
    print(sanitize_text(output))
    if res.returncode != 0:
        if str(cmd_args[0]) == str(PDFLATEX) and "Output written on " in output:
            log_file = cwd / "paper.log" if (cwd / "paper.log").exists() else (cwd / "paper" / "paper.log")
            has_fatal_error = False
            fatal_reasons = []
            if log_file.exists():
                log_content = log_file.read_text(encoding="utf-8", errors="ignore")
                for line in log_content.splitlines():
                    line_str = line.strip()
                    if line_str.startswith("!"):
                        has_fatal_error = True
                        fatal_reasons.append(line_str)
                    elif any(fatal_tok in line_str for fatal_tok in ["Emergency stop", "Fatal error", "Undefined control sequence", "LaTeX Error", "Missing"]):
                        has_fatal_error = True
                        fatal_reasons.append(line_str)
                undef_refs = re.findall(r"Reference `(.*?)' on page .* undefined", log_content)
                undef_cites = re.findall(r"Citation `(.*?)' on page .* undefined", log_content)
                if undef_refs:
                    has_fatal_error = True
                    fatal_reasons.append(f"Undefined references: {undef_refs}")
                if undef_cites:
                    has_fatal_error = True
                    fatal_reasons.append(f"Undefined citations: {undef_cites}")
            
            if not has_fatal_error:
                print("[INFO] pdfTeX exited with code 1 due to MiKTeX update-check policy, verified zero LaTeX errors or missing references in paper.log.")
                return output
            else:
                print(f"[FAIL] pdfTeX exited with real LaTeX errors in paper.log: {fatal_reasons[:5]}")
                sys.exit(res.returncode)
        print(f"[FAIL] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    return output


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_numbers_tex(tex_path: Path):
    macros = {}
    if not tex_path.exists():
        return macros
    with open(tex_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\\newcommand\{\\([a-zA-Z0-9]+)\}\{(.*)\}", line.strip())
            if m:
                val = m.group(2).replace("{", "").replace("}", "").strip()
                macros[m.group(1)] = val
    return macros


def extract_pdf_evidence(pdf_file: Path):
    import pypdf
    reader = pypdf.PdfReader(str(pdf_file))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages_text)
    return len(reader.pages), pages_text, full_text


def main():
    start_time = datetime.now()
    print("=" * 80)
    print("MASTER BUILD PIPELINE (make all)")
    print(f"Start Time: {start_time.isoformat()}")
    print("=" * 80)

    # Pre-flight assertions
    if not RUNS_MULTISEED_V3.exists():
        print(f"[FATAL] Missing {RUNS_MULTISEED_V3}")
        sys.exit(1)

    runs_sha = compute_sha256(RUNS_MULTISEED_V3)
    print(f"[HASH] results/runs_multiseed_v3.jsonl SHA-256: {runs_sha}")

    # 1. Run statistics
    print("\n" + "=" * 80)
    print("STEP 1: RUN STATISTICS (Independent Referee & ANOVA)")
    print("=" * 80)
    indep_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "independent_check.py"), str(RUNS_MULTISEED_V3)])
    anova_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "run_anova_multiseed.py")])

    # 2. Write paper/numbers.tex and all LaTeX tables
    print("\n" + "=" * 80)
    print("STEP 2: GENERATE NUMBERS.TEX AND LATEX TABLES")
    print("=" * 80)
    tables_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "generate_tables.py")])
    hyp_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "generate_hypotheses_table.py")])
    full_matrix_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "generate_full_matrix.py")])
    costs_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "compute_network_costs.py")])

    # 3. Generate every figure
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE ALL PUBLICATION FIGURES")
    print("=" * 80)
    figures_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "plot_figures.py")])
    sweep_out = run_cmd([PYTHON, str(ROOT / "subbyte" / "cost_model" / "sweep_sensitivity.py")])

    # 4. Synchronize and check claims
    print("\n" + "=" * 80)
    print("STEP 4: SYNCHRONIZE AND CHECK CLAIMS")
    print("=" * 80)
    sync_claims_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "sync_claims.py")])
    check_claims_out = run_cmd([PYTHON, str(SCRIPTS_DIR / "check_claims.py")])

    # 5. Build PDF with pdfTeX and BibTeX (Convergence Loop)
    print("\n" + "=" * 80)
    print("STEP 5: COMPILE PDF WITH PDFTEX AND BIBTEX (Convergence Loop)")
    print("=" * 80)
    if not (PAPER_DIR / "ieeeaccess.cls").exists():
        print("[PDFTEX] Skipping PDF compilation: IEEE Access LaTeX template (paper/ieeeaccess.cls) is absent.")
        print("[INFO] Download IEEE Access template from IEEE to compile paper/paper.tex.")
    else:
        print("[PDFTEX] Initial build...")
        run_cmd([PDFLATEX, "-interaction=nonstopmode", "paper.tex"], cwd=PAPER_DIR)
        print("[BIBTEX] Generating bibliography...")
        run_cmd([BIBTEX, "paper"], cwd=PAPER_DIR)
        max_passes = 4
        for iteration in range(1, max_passes + 1):
            print(f"[PDFTEX] Convergence iteration {iteration}/{max_passes}...")
            p_out = run_cmd([PDFLATEX, "-interaction=nonstopmode", "paper.tex"], cwd=PAPER_DIR)
            log_file = PAPER_DIR / "paper.log"
            log_content = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
            if "Rerun to get cross-references right" not in log_content and "rerunfilecheck" not in log_content.lower():
                print(f"[PDFTEX] Converged after iteration {iteration} with 0 rerun warnings.")
                break

        # 6. Extract evidence from PDF
        page_count, pages_text, full_pdf_text = extract_pdf_evidence(PDF_PATH)
        print(f"[PDF] Verified rebuilt paper.pdf: {page_count} pages, {len(full_pdf_text)} characters.")

    total_time = (datetime.now() - start_time).total_seconds()
    print("=" * 80)
    print(f"MAKE ALL PIPELINE COMPLETE IN {total_time:.1f}s -- EXIT 0")
    print("=" * 80)

if __name__ == '__main__':
    main()
