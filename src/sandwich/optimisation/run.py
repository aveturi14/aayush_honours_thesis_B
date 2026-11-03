# Runs the ANN-MOGA: reads moga.yaml, builds NSGA-II, runs it, measures and saves the front.
#
# The surrogate is loaded once and can be passed into run_moga() again and again,
# so a Week 8 sweep of many runs never reloads it.

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

from ..ann.surrogate import ROOT, Surrogate
from ..config import CONFIG_DIR
from ..sampling import LAYER_COLS, OUTPUT_COLS
from . import metrics
from .problem import OUT, RoundDiscrete, SandwichProblem

MOGA_CONFIG = CONFIG_DIR / "moga.yaml"


# merges overrides b into a copy of a; nested blocks (e.g. algorithm.crossover) are merged key by key
def merge_config(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = merge_config(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# reads moga.yaml; overrides change single settings without editing the file
# e.g. load_moga_config(overrides={"algorithm": {"pop_size": 50}})
def load_moga_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(path or MOGA_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return merge_config(cfg, overrides)


# NSGA-II from the algorithm block of moga.yaml
def build_algorithm(alg: dict) -> NSGA2:
    # parent selection: binary tournament on rank and crowding, or plain random
    selection = {
        "binary_tournament": TournamentSelection(func_comp=binary_tournament),
        "random": RandomSelection(),
    }[alg.get("selection", "binary_tournament")]

    # two designs count as the same if their genes are this close (index genes differ by
    # at least 1, so in practice this is a t_c tolerance in mm); pymoo's own default of
    # 1e-16 lets copies that differ only by rounding noise fill up the front
    dupes = DefaultDuplicateElimination(epsilon=alg.get("duplicate_tolerance", 1e-6)) \
        if alg.get("eliminate_duplicates", True) else False

    cx, mu = alg["crossover"], alg["mutation"]
    return NSGA2(
        pop_size=alg["pop_size"],
        sampling=FloatRandomSampling(),                       # random start inside the bounds
        selection=selection,
        crossover=SBX(prob=cx["prob"], eta=cx["eta"]),
        mutation=PM(prob=mu.get("prob", 0.9), prob_var=mu.get("prob_var"), eta=mu["eta"]),
        repair=RoundDiscrete(),                               # snap index genes after every change
        eliminate_duplicates=dupes,
    )


# the trained ANN named in the surrogate block of moga.yaml
def load_surrogate(cfg: dict) -> Surrogate:
    return Surrogate(ROOT / cfg["surrogate"]["model_dir"], ROOT / cfg["surrogate"]["scalers"])


# everything one run produces
@dataclass
class MogaResult:
    pareto: pd.DataFrame            # one row per front point: design, ANN and analytical outputs
    metrics: dict                   # cardinality, spread, violation rate, hypervolume, knee
    config: dict                    # the exact settings used
    archive: pd.DataFrame = field(repr=False)   # every evaluated design (W_t, C_t, feasible)


# runs the MOGA once and measures the front
def run_moga(cfg: dict | None = None, surrogate: Surrogate | None = None,
             seed: int | None = None, verbose: bool = False) -> MogaResult:
    cfg = cfg or load_moga_config()
    alg = cfg["algorithm"]
    seed = alg.get("seed", 1) if seed is None else seed

    # surrogate from the paths in moga.yaml, unless one is passed in (sweeps pass one in)
    surrogate = surrogate or load_surrogate(cfg)

    # where feasibility is judged: "ann" (paper) or "exact" (closed-form solver)
    feas = cfg.get("feasibility", {})
    problem = SandwichProblem(surrogate, constraints_from=feas.get("constraints_from", "ann"))

    # run until the evaluation budget is spent
    t0 = time.time()
    res = minimize(problem, build_algorithm(alg), ("n_eval", alg["n_evaluations"]),
                   seed=seed, verbose=verbose)
    seconds = time.time() - t0

    # final front: pymoo returns the feasible non-dominated designs
    if res.X is None or not np.all(res.G <= 0):
        raise RuntimeError("no feasible design found; check the constraints and the budget")
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    order = np.argsort(F[:, 0])                 # lightest first, like Table 5
    X, F = X[order], F[order]

    # final reporting rule: only designs that pass the exact solver may be reported
    # (always true in "exact" mode; in "ann" mode, failing designs are removed and the
    # front is re-sorted among the designs that are left)
    n_front_before_filter = len(X)
    if feas.get("report_only_feasible", False):
        ok = (problem.violations(problem.analytical(X).to_numpy()) <= 0).all(axis=1)
        if not ok.any():
            raise RuntimeError("no design on the front passes the exact solver")
        X, F = X[ok], F[ok]
        keep = np.sort(metrics.non_dominated(F))        # still sorted lightest first
        X, F = X[keep], F[keep]
        
        
    # design table with the ANN objectives and the exact solver's re-check
    pareto = problem.designs_frame(X)
    pareto["W_t_ann"], pareto["C_t_ann"] = F[:, 0], F[:, 1]
    exact = problem.analytical(X)
    pareto["W_t_exact"], pareto["C_t_exact"] = exact["W_t"].to_numpy(), exact["C_t"].to_numpy()
    pareto["feasible_exact"] = (problem.violations(exact[OUTPUT_COLS].to_numpy()) <= 0).all(axis=1)
    # deflection ratio both ways, since C_delta is the constraint that binds on the front
    pareto["C_delta_ann"] = surrogate.predict(problem.decode(X))[:, OUT["C_delta"]]
    pareto["C_delta_exact"] = exact["C_delta"].to_numpy()
    knee, D = metrics.imdsm_knee(F)
    pareto["D_imdsm"] = D
    pareto["knee"] = np.arange(len(F)) == knee

    m = {
        "seed": int(seed),
        "constraints_from": problem.constraints_from,
        "front_before_feasibility_filter": int(n_front_before_filter),
        "n_evaluations": int(problem.n_evaluated),
        "n_generations": int(res.algorithm.n_gen),
        "run_seconds": round(seconds, 2),
        "cardinality": metrics.cardinality(F),
        "distinct_designs": metrics.distinct_designs(pareto.rho_c, pareto[LAYER_COLS].to_numpy()),
        "spread_delta": metrics.spread_delta(F),
        "violation_rate": metrics.violation_rate(problem.n_evaluated, problem.n_feasible),
        "hypervolume": metrics.hypervolume(F, cfg["metrics"]["hv_reference"]),
        "W_t_range": [float(F[:, 0].min()), float(F[:, 0].max())],
        "C_t_range": [float(F[:, 1].min()), float(F[:, 1].max())],
        "knee_W_t": float(F[knee, 0]),
        "knee_C_t": float(F[knee, 1]),
        "knee_design": pareto.loc[knee, ["rho_c", "t_c", "layup"]].to_dict(),
        "front_feasible_exact": int(pareto["feasible_exact"].sum()),
        "max_rel_err_W_t": float(np.max(np.abs(pareto.W_t_ann / pareto.W_t_exact - 1))),
        "max_rel_err_C_t": float(np.max(np.abs(pareto.C_t_ann / pareto.C_t_exact - 1))),
    }
    return MogaResult(pareto=pareto, metrics=m, config=cfg, archive=problem.archive())


# writes the front, the metrics and the settings used into one folder
def save_result(result: MogaResult, out_dir: str | Path | None = None) -> Path:
    out = ROOT / (out_dir or result.config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    result.pareto.to_csv(out / "pareto.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump(result.metrics, f, indent=2, default=str)
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump(result.config, f, sort_keys=False)
    result.archive.to_csv(out / "all_evaluated.csv", index=False)
    return out


# one run of a sweep, returned as a metrics row (top-level so parallel workers can call it)
def _sweep_one(label: str, cfg: dict, surrogate: Surrogate, seed: int) -> dict:
    r = run_moga(cfg, surrogate=surrogate, seed=seed)
    return {"variant": label, **r.metrics}


# runs several settings x several seeds, reusing one surrogate
# variants: {"label": overrides dict}; returns one metrics row per run
# n_jobs > 1 spreads the runs over that many CPU cores (-1 = all cores)
def run_sweep(variants: dict[str, dict], seeds: list[int],
              base_cfg: dict | None = None, surrogate: Surrogate | None = None,
              verbose: bool = True, n_jobs: int = 1) -> pd.DataFrame:
    base_cfg = base_cfg or load_moga_config()
    surrogate = surrogate or load_surrogate(base_cfg)        # loaded once for every run
    tasks = [(label, merge_config(base_cfg, ov), seed)
             for label, ov in variants.items() for seed in seeds]

    if n_jobs == 1:
        rows = []
        for label, cfg, seed in tasks:
            rows.append(_sweep_one(label, cfg, surrogate, seed))
            if verbose:
                m = rows[-1]
                print(f"{label:<24} seed {seed}: {m['cardinality']:>3} points, "
                      f"knee W {m['knee_W_t']:.2f} C {m['knee_C_t']:.2f}")
    else:
        from joblib import Parallel, delayed                 # ships with Anaconda
        rows = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_sweep_one)(label, cfg, surrogate, seed) for label, cfg, seed in tasks)
    return pd.DataFrame(rows)