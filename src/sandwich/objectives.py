
from .clt import Material
from .solver import Core

# overall geometry of panel being investigated, Figure 3
PANEL_L_MM = 1120.0          # Fig. 3, overall length
PANEL_B_MM = 582.0           # Fig. 3, overall width
PANEL_AREA_MM2 = PANEL_L_MM * PANEL_B_MM

# weight, (13) from base paper
def face_areal_density(layup: list[Material]) -> float:
    return sum(m.rho * m.t_ply for m in layup)

def total_weight(layup: list[Material], core: Core, t_c: float,
                 area_mm2: float = PANEL_AREA_MM2) -> float:
    areal = core.rho * t_c + 2.0 * face_areal_density(layup)
    return areal * area_mm2 * 1e-9

# cost, (15) from base paper
def face_areal_cost(layup: list[Material]) -> float:
    return sum(m.cost_ratio * m.rho * m.t_ply for m in layup)

def total_cost(layup: list[Material], core: Core, t_c: float,
               area_mm2: float = PANEL_AREA_MM2,
               normalisation: float = 1.0) -> float:
    areal = core.cost_ratio * core.rho * t_c + 2.0 * face_areal_cost(layup)
    return areal * area_mm2 * normalisation * 1e-9