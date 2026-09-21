# aayush_honours_thesis_B
# Thesis B: ANN-MOGA Composite Sandwich Optimisation

Replication of Sahib, M.M. and Kovács, G. (2024). Multi-objective
optimization of composite sandwich structures using Artificial Neural
Networks and Genetic Algorithm. *Results in Engineering*, 21, 101937.

UNSW Thesis B. Supervisor: Dr Shiva Abdoli.

## Status

**Week 1** — repository scaffold and environment complete. Config
transcription and gap register in progress.

## Setup

```bash
git clone <repo> && cd <repo>
conda env create -f environment.yml -n thesisb
conda activate thesisb
pip install -r requirements-pip.txt
pip install -e .
pytest -m "not slow"
```

## Progress

| Week | Deliverable | Status |
|---|---|---|
| 1 | Config (Tables 1–4), gap register v1, FEM decision | in progress |
| 2 | CLT and Tsai-Wu modules, hand calcs | |
| 3 | Analytical solver, Table 5 reproduced | |
| 4 | 9,000-point dataset, normalisation pipeline | |
| 5 | Baseline ANN, FEM runs | |
| 6 | Tuned ANN, parity plots, frozen baseline | |
| 7 | ANN-MOGA, first Pareto front | |
| 8 | Sweep, knee point, Table 5 benchmark | |
| 9 | FEM comparison, c sensitivity, gap register closed | |
| 10 | Refactor, Thesis C interface spec | |

## Layout

| Path | Contents |
|---|---|
| `config/` | Published parameters |
| `src/sandwich/` | Solver, surrogate, optimiser |
| `scripts/` | Numbered pipeline stages |
| `docs/` | Gap register, decisions, hand calcs |
| `results/tables/` | Reproduced tables (committed) |