
# Surrogate wrapper, MOGA problem definition, front metrics and a short MOGA run.

import numpy as np
import pandas as pd
import pytest

from sandwich.ann.surrogate import Surrogate
from sandwich.config import load_problem
from sandwich.optimisation import metrics
from sandwich.optimisation.problem import RoundDiscrete, SandwichProblem
from sandwich.optimisation.run import load_moga_config, run_moga
from sandwich.sampling import INPUT_COLS, OUTPUT_COLS, evaluate_designs, feasibility_flags, sample_designs


@pytest.fixture(scope="module")
def surrogate():
    return Surrogate()


@pytest.fixture(scope="module")
def problem(surrogate):
    return SandwichProblem(surrogate)


# surrogate

def test_surrogate_close_to_solver(surrogate):
    # trained ANN within 1% of the exact solver for weight and cost on fresh designs
    Z = sample_designs(200, 11)
    ann = surrogate.predict_frame(Z)
    exact = evaluate_designs(Z)[OUTPUT_COLS]
    for col in ("W_t", "C_t"):
        assert np.max(np.abs(ann[col] / exact[col] - 1)) < 0.01, col


def test_surrogate_flags_out_of_range_inputs(surrogate):
    Z = sample_designs(3, 0).to_numpy(float)
    Z[1, 1] = 80.0                                   # t_c far beyond the 50 mm training max
    assert surrogate.outside_training_range(Z).tolist() == [False, True, False]


# problem definition

def test_decode_knee_design(problem):
    # genes for Table 5 row 6: t_c 17.93, density 29 (index 0), layup 3,1,1,2,1,2
    x = np.array([[17.93, 0, 2, 0, 0, 1, 0, 1]], float)
    row = dict(zip(INPUT_COLS, problem.decode(x)[0]))
    assert row["rho_c"] == 29 and row["t_c"] == 17.93
    assert [row[f"L{k}"] for k in range(1, 7)] == [3, 1, 1, 2, 1, 2]
    assert row["p"] == 0.004142


def test_bounds_follow_table4(problem):
    assert problem.xl[0] == 5 and problem.xu[0] == 20


def test_rounding_gives_every_option_an_equal_share(problem):
    # uniform genes across the bounds should land on each density about equally often
    rng = np.random.default_rng(0)
    X = rng.uniform(problem.xl, problem.xu, size=(60_000, problem.n_var))
    X = RoundDiscrete()._do(problem, X)
    counts = np.bincount(X[:, 1].astype(int), minlength=6)
    assert counts.min() / counts.max() > 0.95
    assert set(np.unique(X[:, 2:])) == {0, 1, 2}


def test_violations_agree_with_feasibility_flags(problem):
    # the MOGA constraints and the Week 4 feasibility flags are the same rule
    df = evaluate_designs(sample_designs(300, 5))
    viol_ok = (problem.violations(df[OUTPUT_COLS].to_numpy()) <= 0).all(axis=1)
    assert np.array_equal(viol_ok, feasibility_flags(df).feasible.to_numpy())


# metrics

def test_imdsm_picks_published_knee():
    # Eq. 19 on the published Table 5 points selects row 6 (Wt 2.39, Ct 2.1), as in Sec. 5.2
    rows = load_problem()["pareto_designs"]["rows"]
    F = np.array([[r["Wt"], r["Ct"]] for r in rows])
    knee, _ = metrics.imdsm_knee(F)
    assert rows[knee]["n"] == 6


def test_table5_has_16_distinct_designs():
    # rows 7/8 and 9/10 of Table 5 are the same laminates in a different ply order
    rows = load_problem()["pareto_designs"]["rows"]
    assert metrics.distinct_designs([r["rho_c"] for r in rows], [r["layup"] for r in rows]) == 16


def test_spread_is_zero_for_even_front():
    F = np.column_stack([np.linspace(0, 1, 10), np.linspace(1, 0, 10)])
    assert metrics.spread_delta(F) == pytest.approx(0.0, abs=1e-12)


def test_cardinality_ignores_exact_repeats():
    F = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])
    assert metrics.cardinality(F) == 2


# short end-to-end run

def test_short_moga_run(surrogate):
    cfg = load_moga_config(overrides={"algorithm": {"n_evaluations": 2000, "pop_size": 50}})
    r = run_moga(cfg, surrogate=surrogate)
    assert r.metrics["n_evaluations"] >= 2000
    assert r.metrics["cardinality"] >= 3
    assert r.pareto["knee"].sum() == 1
    # every front point is feasible according to the ANN
    assert (r.pareto.t_c.between(5, 20)).all()
    # front is sorted lightest first and is non-dominated
    F = r.pareto[["W_t_ann", "C_t_ann"]].to_numpy()
    assert np.all(np.diff(F[:, 0]) >= 0) and np.all(np.diff(F[:, 1]) <= 0)
    
    
def test_exact_feasibility_front_passes_exact_solver(surrogate):
    # with constraints from the exact solver, every reported design satisfies Eqs. 22-27 exactly
    cfg = load_moga_config(overrides={"algorithm": {"n_evaluations": 2000, "pop_size": 50},
                                      "feasibility": {"constraints_from": "exact",
                                                      "report_only_feasible": True}})
    r = run_moga(cfg, surrogate=surrogate)
    assert r.pareto["feasible_exact"].all()
    assert (r.pareto["C_delta_exact"] <= 1.0).all()


def test_report_filter_removes_exact_failures(surrogate):
    # in the paper's "ann" mode, the reporting filter still keeps only exactly feasible designs
    cfg = load_moga_config(overrides={"algorithm": {"n_evaluations": 2000, "pop_size": 50},
                                      "feasibility": {"constraints_from": "ann",
                                                      "report_only_feasible": True}})
    r = run_moga(cfg, surrogate=surrogate)
    assert r.pareto["feasible_exact"].all()


def test_unknown_feasibility_mode_rejected(surrogate):
    with pytest.raises(ValueError):
        SandwichProblem(surrogate, constraints_from="fem")