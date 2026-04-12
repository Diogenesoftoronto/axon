# Axon Paper Draft

This directory contains a peer review style draft describing Axon and summarizing current benchmark results.

## Files

- `paper/main.tex`: manuscript (all tables and figures are inline using pgfplots/tikz)
- `paper/refs.bib`: bibliography

## Build

Requires a LaTeX distribution with `pgfplots`, `tikz`, `booktabs`, `hyperref`, and `listings` packages (e.g., TeX Live with `texlive-pgfplots`). Example:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternatively, `latexmk -pdf main.tex` or `tectonic main.tex` also work.

## Reproducing Results

The manuscript references benchmark artifacts from the following directories:

- `benchmarks/overall-20260226-refresh/` — books and ledger depth-profile results (Tables 5–6)
- `benchmarks/overall-20260228-165727/` — failure taxonomy aggregate slice
- `benchmarks/overall-20260228-204653/` — latest full benchmark run (multi-model, HF packs, Codeforces)

Multi-model Pareto results (Tables 7–13, Figures 5–11) were collected across seven models on the Synthetic inference platform; six models pre-FINAL-fix, Kimi-K2.5 post-fix. See `benchmarks/overall-20260228-204653/overall-summary.md` for run details.

## Benchmark Plots and Notebook Analysis

To generate updated figures from benchmark outputs and HF dataset packs:

```bash
.venv/bin/python scripts/plot_benchmark_findings.py --out-dir benchmarks/analysis
```

Artifacts are written under `benchmarks/analysis/`.

For interactive analysis and custom slices, open the marimo notebook:

```bash
.venv/bin/marimo edit notebooks/benchmark_results_analysis.py
```

Benchmark execution and analysis workflows are documented in `docs/benchmarking.md`.
