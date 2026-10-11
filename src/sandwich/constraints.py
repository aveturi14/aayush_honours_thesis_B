
from dataclasses import dataclass

from .solver import Core, Response

# overall thickness limit for sandwich panel, (27) from base paper
MAX_THICKNESS_MM = 20.0       

# max deflection constraint, (26) from base paper
DEFLECTION_DIVISOR = 400.0   

# class defined to store all relevant safety factors tracked in base paper
@dataclass(frozen=True)
class SafetyFactors:
    SF_c: float         # honeycomb core shear
    SF_f: float         # face sheet strength
    SF_int: float       # intra-cell buckling
    SF_wr: float        # face sheet wrinkling

# computes and stores all safety factors from a given response and core, adjusted based on isolated issues with formulae
def safety_factors(r: Response, core: Core) -> SafetyFactors:
    return SafetyFactors(
        SF_c=core.tau_x / r.tau_cs,
        SF_f=r.sigma_fx / r.sigma_f,
        SF_int=r.sigma_in_cr / r.sigma_f,
        SF_wr=r.sigma_wr / r.sigma_f,
    )

# computes and stores all safety factors from a given response and core, uses (22), (23), (24), (25) in base paper
def safety_factors_as_printed(r: Response, core: Core) -> SafetyFactors:
    return SafetyFactors(
        SF_c=r.tau_cs / core.tau_x,
        SF_f=r.sigma_f / r.sigma_fx,
        SF_int=r.sigma_in_cr / r.sigma_fx,
        SF_wr=r.sigma_wr / r.sigma_fx,
    )

# calculates deflection limit, (26) from base paper
def deflection_limit(r: Response, span_l: float = 800.0) -> float:
    return DEFLECTION_DIVISOR * r.delta / span_l


# (27) from base paper
def total_thickness(t_f: float, t_c: float) -> float:
    return t_c + 2.0 * t_f

# returns boolean based on requirements of all safety factors, thickness limit for panel
def is_feasible(sf: SafetyFactors, C_delta: float, t_sw: float) -> bool:
    return (sf.SF_c >= 1.0 and sf.SF_f >= 1.0
            and sf.SF_int >= 1.0 and sf.SF_wr >= 1.0
            and C_delta <= 1.0 and t_sw <= MAX_THICKNESS_MM)