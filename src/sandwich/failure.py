
from dataclasses import dataclass
import math

import numpy as np

from .clt import (Material, extensional_stiffness, reduced_stiffness,
                  transverse_strain_ratio, effective_modulus)

# define class for Tsai-Wu coefficients
@dataclass(frozen=True)
class TsaiWu:
    H1: float
    H2: float
    H11: float
    H22: float
    H66: float
    H12: float

    # checks / ensures tsai-wu cofficient produces a closed surface, not something with infinite stress state
    def closed_envelope(self) -> bool:
        return self.H12 ** 2 < self.H11 * self.H22

# function that takes material and computes the corresponding tsai-wu coefficients
def coefficients(m: Material, h12_model: str = "mises_hencky") -> TsaiWu:
    H1 = 1.0 / m.Xt - 1.0 / m.Xc
    H11 = 1.0 / (m.Xt * m.Xc)
    H2 = 1.0 / m.Yt - 1.0 / m.Yc
    H22 = 1.0 / (m.Yt * m.Yc)
    H66 = 1.0 / m.S ** 2
    # H6 = 0 by Kaw Eq. 2.163.
    if h12_model == "mises_hencky":            # Kaw Eq. 2.171c
        H12 = -0.5 * math.sqrt(H11 * H22)
    elif h12_model == "tsai_hill":             # Kaw Eq. 2.171a
        H12 = -0.5 / m.Xt ** 2
    elif h12_model == "hoffman":               # Kaw Eq. 2.171b
        H12 = -0.5 / (m.Xt * m.Xc)
    elif h12_model == "zero":                  # sensitivity case
        H12 = 0.0
    else:
        raise ValueError(f"unknown H12 model: {h12_model!r}")
    return TsaiWu(H1, H2, H11, H22, H66, H12)

# calculates how much stress state can be scaled before reaching tsai-wu and ply failure
def strength_ratio(s1: float, s2: float, t12: float, c: TsaiWu) -> float:
    
    # terms that scale with R-squared
    a = (c.H11 * s1 ** 2 + c.H22 * s2 ** 2 + c.H66 * t12 ** 2
         + 2.0 * c.H12 * s1 * s2)
    
    # terms that scale with R, zero for isotropic materials
    b = c.H1 * s1 + c.H2 * s2
    if a <= 0.0:
        raise ValueError("non-positive quadratic term; check H coefficients")
    return (-b + math.sqrt(b * b + 4.0 * a)) / (2.0 * a)


# laminate strength finds first ply failure strenght of face sheet  
def laminate_strength(layup: list[Material], h12_model: str = "mises_hencky"):
    A = extensional_stiffness(layup)
    eps_x = 1.0                       # reference strain; R rescales it away
    eps_y = transverse_strain_ratio(A) * eps_x

    ratios = []
    for m in layup:
        Q = reduced_stiffness(m)
        s1 = Q[0, 0] * eps_x + Q[0, 1] * eps_y
        s2 = Q[0, 1] * eps_x + Q[1, 1] * eps_y
        t12 = 0.0                     # no shear: all plies at 0 deg, N_xy = 0
        ratios.append(strength_ratio(s1, s2, t12, coefficients(m, h12_model)))

    k = int(np.argmin(ratios))
    sigma_fx = ratios[k] * effective_modulus(layup)
    return sigma_fx, k, ratios