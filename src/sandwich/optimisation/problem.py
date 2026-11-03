
# The optimisation problem the MOGA solves (Sec. 4.1-4.3):
#   minimise W_t and C_t (Eqs. 20-21), subject to Eqs. 22-27,
#   over core density, core thickness and the six layer materials (Table 4).
#
# Gene layout (8 genes):
#   gene 0      core thickness t_c, continuous, 5-20 mm (Table 4)
#   gene 1      index into the six core densities (0-5)
#   genes 2-7   index into the three layer materials, one per layer (0-2)
#
# Ranges, codes, load and limits are read from design_space.yaml once, when the
#
# Feasibility source (constraints_from):
#   "ann"   - constraints come from the ANN's predictions, as in the paper. A design the
#             ANN wrongly calls feasible can be kept, bred from and reported.
#   "exact" - constraints come from the closed-form solver (Eqs. 1-12, 22-27). Only truly
#             feasible designs can rank as feasible, so they are the ones kept, bred from
#             and reported. Objectives still come from the ANN in both modes.
# problem is created, and reused for every one of the 23,000 evaluations.

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair

from ..config import load_design_space
from ..sampling import INPUT_COLS, LAYER_COLS, OUTPUT_COLS, evaluate_designs
from ..ann.surrogate import Surrogate

# position of each output in OUTPUT_COLS, so the code reads OUT["W_t"] instead of 7
OUT = {name: k for k, name in enumerate(OUTPUT_COLS)}

# the six constraints, in the order they appear in the G matrix (Eqs. 22-27)
CONSTRAINT_NAMES = ["SF_c", "SF_f", "SF_int", "SF_wr", "C_delta", "t_sw"]
C_DELTA_MAX = 1.0       # Eq. 26: C_delta = 400 * delta / l <= 1 (same limit as feasibility_flags)


class SandwichProblem(Problem):
    """Weight-cost problem with the trained ANN as the fitness function."""

    # reads the design space once and sets the gene bounds
    def __init__(self, surrogate: Surrogate, constraints_from: str = "ann",
                 keep_archive: bool = True):
        if constraints_from not in ("ann", "exact"):
            raise ValueError(f"constraints_from must be 'ann' or 'exact', not {constraints_from!r}")
        self.constraints_from = constraints_from
        ds = load_design_space()
        cont, disc, lim = ds["continuous"], ds["discrete"], ds["constraints"]

        # lookup tables that turn integer genes into physical values
        self.densities = np.asarray(disc["core_density"]["values"], float)       # Table 2
        self.codes = np.asarray(disc["layer_material"]["values"], int)           # Table 4
        self.n_layers = disc["layer_material"]["n_layers"]
        self.p = cont["applied_load"]["optimisation_value"]["value"]             # load held fixed
        tc = cont["core_thickness"]["optimisation"]                              # Table 4, 5-20 mm

        # constraint limits (Eqs. 22-27)
        self.sf_min = lim["min_safety_factor"]["value"]
        self.t_max = lim["max_total_thickness"]["value"]
        self.surrogate = surrogate

        # index genes run from -0.5 to (n - 0.5), so after rounding every option
        # gets an equal-width slice of the range (bounds of 0 to n-1 would give the
        # first and last option only half a slice each)
        n_rho, n_mat = len(self.densities), len(self.codes)
        xl = np.array([tc["min"]["value"], -0.5] + [-0.5] * self.n_layers, float)
        xu = np.array([tc["max"]["value"], n_rho - 0.5] + [n_mat - 0.5] * self.n_layers, float)
        super().__init__(n_var=2 + self.n_layers, n_obj=2,
                         n_ieq_constr=len(CONSTRAINT_NAMES), xl=xl, xu=xu)

        # running record of every design the GA asks about
        self.keep_archive = keep_archive
        self.n_evaluated, self.n_feasible, self._archive = 0, 0, []

    # integer genes -> ANN input rows in INPUT_COLS order (rho_c, t_c, L1..L6, p)
    def decode(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        cols = {
            "rho_c": self.densities[X[:, 1].astype(int)],
            "t_c": X[:, 0],
            **{c: self.codes[X[:, 2 + k].astype(int)] for k, c in enumerate(LAYER_COLS)},
            "p": np.full(len(X), self.p),
        }
        return np.column_stack([cols[c] for c in INPUT_COLS]).astype(float)

    # the six constraints in "<= 0 is satisfied" form, for outputs Y (n x 8)
    def violations(self, Y: np.ndarray) -> np.ndarray:
        Y = np.atleast_2d(Y)
        return np.column_stack([
            self.sf_min - Y[:, OUT["SF_c"]],            # Eq. 22, SF_c >= 1
            self.sf_min - Y[:, OUT["SF_f"]],            # Eq. 23, SF_f >= 1
            self.sf_min - Y[:, OUT["SF_int"]],          # Eq. 24, SF_int >= 1
            self.sf_min - Y[:, OUT["SF_wr"]],           # Eq. 25, SF_wr >= 1
            Y[:, OUT["C_delta"]] - C_DELTA_MAX,         # Eq. 26, C_delta <= 1
            Y[:, OUT["t_sw"]] - self.t_max,             # Eq. 27, t_sw <= 20 mm
        ])

    # pymoo calls this with the whole population: one ANN call scores everyone
    def _evaluate(self, X, out, *args, **kwargs):
        Y = self.surrogate.predict(self.decode(X))
        out["F"] = Y[:, [OUT["W_t"], OUT["C_t"]]]       # Eqs. 20, 21, both minimised, from the ANN

        # constraints (Eqs. 22-27): from the ANN (paper) or from the exact solver
        if self.constraints_from == "exact":
            Y_con = self.analytical(X).to_numpy()       # exact SF_c ... t_sw for every design
        else:
            Y_con = Y
        out["G"] = self.violations(Y_con)

        # bookkeeping for the violation rate and the all-designs scatter plot
        feasible = (out["G"] <= 0).all(axis=1)
        self.n_evaluated += len(X)
        self.n_feasible += int(feasible.sum())
        if self.keep_archive:
            self._archive.append(np.column_stack([out["F"], feasible]))

    # every evaluated design as a table of W_t, C_t and feasibility
    def archive(self) -> pd.DataFrame:
        if not self._archive:
            return pd.DataFrame(columns=["W_t", "C_t", "feasible"])
        A = np.vstack(self._archive)
        return pd.DataFrame({"W_t": A[:, 0], "C_t": A[:, 1], "feasible": A[:, 2].astype(bool)})

    # genes -> readable design table (density, thickness, layers, load, layup string)
    def designs_frame(self, X: np.ndarray) -> pd.DataFrame:
        Z = pd.DataFrame(self.decode(X), columns=INPUT_COLS)
        Z[LAYER_COLS + ["rho_c"]] = Z[LAYER_COLS + ["rho_c"]].astype(int)
        Z["layup"] = Z[LAYER_COLS].astype(str).agg(",".join, axis=1)    # Table 5 style, e.g. 3,1,1,2,1,2
        return Z

    # re-checks designs with your existing closed-form solver (no ANN error)
    def analytical(self, X: np.ndarray) -> pd.DataFrame:
        Z = self.designs_frame(X)[INPUT_COLS]
        return evaluate_designs(Z)[OUTPUT_COLS]


class RoundDiscrete(Repair):
    """Keeps t_c continuous and snaps the density and layer genes to whole-number indices."""

    # rounds genes 1-7 and keeps them inside 0..n-1
    def _do(self, problem, X, **kwargs):
        X = X.copy()
        X[:, 1:] = np.round(X[:, 1:])
        X[:, 1] = np.clip(X[:, 1], 0, len(problem.densities) - 1)
        X[:, 2:] = np.clip(X[:, 2:], 0, len(problem.codes) - 1)
        return X