
import json
from pathlib import Path

import matplotlib.pyplot as plt

from sandwich.config import load_problem
from sandwich.preprocess import prepare
from sandwich.sampling import (OUTPUT_COLS, coverage_report, evaluate_designs,
                               feasibility_flags, sample_designs)

COST_NORMALISATION = 1.0

FIT_ON = "train"

def main():
    # seed and sample count from problem.yaml (42, 9000)
    run = load_problem()["run"]
    
    seed, n = run["seed"], run["n_samples"]
    for d in ("data/raw", "data/processed", "results/tables", "results/figures"):
        Path(d).mkdir(parents=True, exist_ok=True)
        
    df = evaluate_designs(sample_designs(n, seed), cost_normalisation=COST_NORMALISATION)
    flags = feasibility_flags(df)
    df.join(flags).to_csv("data/raw/dataset_9000.csv", index=False)
    
    summary = {
        "seed": seed, "n": n, "cost_normalisation": COST_NORMALISATION,
        "feasible_fraction": {c: float(flags[c].mean()) for c in flags},
        "coverage": coverage_report(df),
    }
    with open("results/tables/week4_coverage_feasibility.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    df[OUTPUT_COLS].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T \
        .to_csv("results/tables/week4_output_stats.csv")
        
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, OUTPUT_COLS):
        ax.hist(df.loc[flags.feasible, col], bins=50, alpha=0.6, label="feasible")
        ax.hist(df.loc[~flags.feasible, col], bins=50, alpha=0.6, label="infeasible")
        ax.set_title(col)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig("results/figures/week4_output_distributions.png", dpi=150)
    
    prepare(df, seed + 1, "data/processed/prepared.npz",
            "data/processed.scalers.json", fit_on=FIT_ON)
    
    print(f"{n} designs, {flags.feasible.mean():.1%} feasible")
    for c in flags.columns[:-1]:
        print(f" {c:<12} pass {flags[c].mean():.1%}")
        
        
if __name__ == "__main__":
    main()
    
    