"""Objective functions: Eqs. 13-15, 20-21.

Eq. 13:  W_t = (rho_c*t_c + 2*(rho_f1*N_l1*t_l1 + rho_f2*N_l2*t_l2
                              + rho_f3*N_l3*t_l3)) * L * B
Eq. 14:  t_f = N_l1*t_l1 + N_l2*t_l2 + N_l3*t_l3
Eq. 15:  C_t = C_f + C_c
Eq. 20:  minimise W_t
Eq. 21:  minimise C_t

L and B are never defined in the paper.  Fig. 3 gives overall panel
dimensions 1120 x 582 mm and effective span 800 x 390 mm.  Back-solving
Eq. 13 from the published weights of the two homogeneous Table 6 designs
gives implied areas of 634,544 and 647,780 mm^2 against 1120 x 582 =
651,840, so L*B is the OVERALL panel.  Adopted as the default.

The FML row of Table 6 implies 562,700 mm^2, 14% low.  Its face sheets
alone give 2.094 kg against a published W_t of 2.1 kg, so that row omits
the core weight.  Treated as a published error; see gap register.

Eq. 15 gives no expression.  Prices are stated relative to GFRP (CFRP
1.5, core 0.5, Al 0.31) with no stated basis.  Neither a volume basis nor
a mass basis reproduces the Table 6 costs under a single normalisation
(volume: 4.09 / 6.19 / 4.48; mass: 1611 / 1517 / 1383, where a constant
would indicate the correct basis).  Mass is closer and is implemented;
resolution deferred to the Week 3 Table 5 reproduction.
"""

from .clt import Material
from .solver import Core

PANEL_L_MM = 1120.0          # Fig. 3, overall length
PANEL_B_MM = 582.0           # Fig. 3, overall width
PANEL_AREA_MM2 = PANEL_L_MM * PANEL_B_MM


# --- weight, Eq. 13 --------------------------------------------------------

def face_areal_density(layup: list[Material]) -> float:
    """sum(rho_i * N_li * t_li) for ONE face sheet.  kg/m^3 * mm."""
    return sum(m.rho * m.t_ply for m in layup)


def total_weight(layup: list[Material], core: Core, t_c: float,
                 area_mm2: float = PANEL_AREA_MM2) -> float:
    """W_t  (Eq. 13).  kg.

    1e-9 converts kg/m^3 * mm * mm^2 to kg.
    """
    areal = core.rho * t_c + 2.0 * face_areal_density(layup)
    return areal * area_mm2 * 1e-9


# --- cost, Eq. 15 ----------------------------------------------------------

def face_areal_cost(layup: list[Material]) -> float:
    """sum(cost_ratio_i * rho_i * N_li * t_li) for ONE face sheet.

    Mass basis; see module docstring and gap register.
    """
    return sum(m.cost_ratio * m.rho * m.t_ply for m in layup)


def total_cost(layup: list[Material], core: Core, t_c: float,
               area_mm2: float = PANEL_AREA_MM2,
               normalisation: float = 1.0) -> float:
    """C_t = C_f + C_c  (Eq. 15).  Unit price.

    `normalisation` is an open constant: the paper reports C_t in "unit
    price" with no stated scale.  Fit it in Week 3.
    """
    areal = core.cost_ratio * core.rho * t_c + 2.0 * face_areal_cost(layup)
    return areal * area_mm2 * normalisation * 1e-9