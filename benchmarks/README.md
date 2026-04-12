# Benchmarks Directory Layout

- `*.json` in this directory: benchmark task datasets and static model lists.
- `results/`: single-run result artifacts (`results-*.json`, `summary-*.md`).
- `overall-*` and `hf-runs-*`: grouped multi-suite run outputs.
- `analysis/`: derived CSV/plot artifacts.

## Conventions

- Prefer writing ad-hoc run outputs to `benchmarks/results/`.
- Keep task datasets at benchmark root so existing scripts work without extra flags.
- Use dated run folders (`overall-...`, `hf-runs-...`) for multi-suite experiments.
