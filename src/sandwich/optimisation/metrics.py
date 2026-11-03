
# Measures of a Pareto front, for the Week 7 first run and the Week 8 sweep.
# F is always an (n x 2) array with columns [W_t, C_t], both minimised.

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# indices of the points no other point beats on both objectives
def non_dominated(F: np.ndarray) -> np.ndarray:
    return NonDominatedSorting().do(np.asarray(F, float), only_non_dominated_front=True)


# number of distinct points on the front (the paper reports 18, Sec. 5.2)
def cardinality(F: np.ndarray, decimals: int = 9) -> int:
    return int(len(np.unique(np.round(np.asarray(F, float), decimals), axis=0)))


# number of physically different designs on the front
# all plies are at 0 deg, so the same materials in a different order give the same
# laminate (CLT is order-blind); designs are compared by density + sorted layup
def distinct_designs(rho_c, layups) -> int:
    keys = {(int(r), tuple(sorted(int(c) for c in l))) for r, l in zip(rho_c, layups)}
    return len(keys)


# Deb's spread measure (Deb et al. 2002), using the front's own end points
# 0 = perfectly even gaps between neighbours, larger = more uneven
# objectives are scaled to [0, 1] first so kg and unit price count equally
def spread_delta(F: np.ndarray) -> float:
    F = np.unique(np.asarray(F, float), axis=0)
    if len(F) < 3:
        return float("nan")
    span = np.ptp(F, axis=0)
    Fn = (F - F.min(axis=0)) / np.where(span == 0, 1.0, span)
    Fn = Fn[np.argsort(Fn[:, 0])]                       # walk along the front by weight
    d = np.linalg.norm(np.diff(Fn, axis=0), axis=1)     # gap to the next point
    return float(np.abs(d - d.mean()).sum() / (len(d) * d.mean()))


# share of all evaluated designs that broke at least one constraint
def violation_rate(n_evaluated: int, n_feasible: int) -> float:
    return float(1.0 - n_feasible / n_evaluated) if n_evaluated else float("nan")


# area of objective space dominated by the front, up to a fixed reference point
# (the same reference must be used for every run that is compared)
def hypervolume(F: np.ndarray, ref_point: tuple[float, float]) -> float:
    return float(HV(ref_point=np.asarray(ref_point, float))(np.asarray(F, float)))


# IMDSM knee point, Eq. 19: the front point closest to the ideal (min W_t, min C_t)
# after dividing each objective by its own minimum
def imdsm_knee(F: np.ndarray) -> tuple[int, np.ndarray]:
    F = np.asarray(F, float)
    W, C = F[:, 0], F[:, 1]
    D = np.sqrt((W / W.min() - 1.0) ** 2 + (C / C.min() - 1.0) ** 2)
    return int(np.argmin(D)), D


# for each published Table 5 point, the closest reproduced point (Week 8 comparison)
# distance is measured after scaling both objectives by the published ranges
def match_to_reference(F: np.ndarray, F_ref: np.ndarray) -> pd.DataFrame:
    F, F_ref = np.asarray(F, float), np.asarray(F_ref, float)
    scale = np.ptp(F_ref, axis=0)
    dist = np.linalg.norm((F_ref[:, None, :] - F[None, :, :]) / scale, axis=2)
    nearest = dist.argmin(axis=1)
    return pd.DataFrame({
        "W_t_paper": F_ref[:, 0], "C_t_paper": F_ref[:, 1],
        "W_t_ours": F[nearest, 0], "C_t_ours": F[nearest, 1],
        "scaled_distance": dist[np.arange(len(F_ref)), nearest],
        "our_index": nearest,
    })