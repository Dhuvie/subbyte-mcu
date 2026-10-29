# Makefile for Sub-Byte MCU Research Repository

PYTHON ?= python
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex

.PHONY: all reproduce reproduce-full test lint claims paper clean

all:
	$(PYTHON) scripts/make_all.py

reproduce:
	$(PYTHON) scripts/reproduce.py

reproduce-full:
	$(PYTHON) scripts/reproduce_full.py


test:
	$(PYTHON) subbyte/cost_model/test_kernels.py

lint:
	$(PYTHON) scripts/lint_paper.py

claims:
	$(PYTHON) scripts/check_claims.py

paper:
	cd paper && $(PDFLATEX) -interaction=nonstopmode paper.tex && $(BIBTEX) paper && $(PDFLATEX) -interaction=nonstopmode paper.tex && $(PDFLATEX) -interaction=nonstopmode paper.tex

clean:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out
