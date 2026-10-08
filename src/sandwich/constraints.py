"""Safety factors and design constraints: Eqs. 9-12, 22-27.

SIGN RESOLUTION - see docs/safety_factor_resolution.md.

Eqs. 9-12 as printed are not safety factors:

  Eq. 9   SF_c   = tau_cs / sigma_xz    applied / allowable
  Eq. 10  SF_f   = sigma_f / sigma_fx   applied / allowable
  Eq. 11  SF_int = sigma_in,cr / sigma_fx   allowable / allowable
  Eq. 12  SF_wr  = sigma_wr / sigma_fx       allowable / allowable

Eqs. 22-25 constrain all four to be >= 1.0.  Under the printed forms of
Eqs. 9 and 10 that would require the applied stress to exceed the
strength, i.e. only failed designs would be feasible.  They are inverted
here.

Eqs. 11 and 12 are a separate error: neither contains an applied stress,
so they measure which failure mode occurs first rather than margin
against load.  No inversion fixes that.  The applied face sheet stress
sigma_f is adopted as the denominator.

Both the adopted and the printed forms are implemented, so Week 9 can
report what each produces.
"""

from dataclasses import dataclass

from .solver import Core, Response

MAX_THICKNESS_MM = 20.0        # Eq. 27
DEFLECTION_DIVISOR = 400.0     # Eq. 26, delta <= l / 400


@dataclass(frozen=True)
class SafetyFactors:
    SF_c: float         # honeycomb core shear
    SF_f: float         # face sheet strength
    SF_int: float       # intra-cell buckling
    SF_wr: float        # face sheet wrinkling


def safety_factors(r: Response, core: Core) -> SafetyFactors:
    """Adopted forms: allowable / applied, so SF >= 1 means safe.

    Eqs. 9 and 10 inverted; Eqs. 11 and 12 re-based on sigma_f.
    """
    return SafetyFactors(
        SF_c=core.tau_x / r.tau_cs,
        SF_f=r.sigma_fx / r.sigma_f,
        SF_int=r.sigma_in_cr / r.sigma_f,
        SF_wr=r.sigma_wr / r.sigma_f,
    )


def safety_factors_as_printed(r: Response, core: Core) -> SafetyFactors:
    """Eqs. 9-12 exactly as published.  Retained for the Week 9 verdict."""
    return SafetyFactors(
        SF_c=r.tau_cs / core.tau_x,
        SF_f=r.sigma_f / r.sigma_fx,
        SF_int=r.sigma_in_cr / r.sigma_fx,
        SF_wr=r.sigma_wr / r.sigma_fx,
    )


def deflection_limit(r: Response, span_l: float = 800.0) -> float:
    """C_delta = 400 * delta / l   (Eq. 26).  Constrained <= 1.0.

    Equivalent to delta <= l / 400 = 2.0 mm for l = 800 mm.
    """
    return DEFLECTION_DIVISOR * r.delta / span_l


def total_thickness(t_f: float, t_c: float) -> float:
    """t_sw = t_c + 2*t_f   (Eq. 27).  Constrained <= 20.0 mm."""
    return t_c + 2.0 * t_f


def is_feasible(sf: SafetyFactors, C_delta: float, t_sw: float) -> bool:
    """Eqs. 22-27 combined."""
    return (sf.SF_c >= 1.0 and sf.SF_f >= 1.0
            and sf.SF_int >= 1.0 and sf.SF_wr >= 1.0
            and C_delta <= 1.0 and t_sw <= MAX_THICKNESS_MM)