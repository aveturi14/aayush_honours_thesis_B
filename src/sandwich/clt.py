"""Classical lamination theory.

Q matrices:   Kaw Eq. 2.10 / Nettles RP-1351 Eq. 10
A matrix:     Nettles RP-1351 Sec. IV
E_f:          Kaw Eq. 4.34-4.35, reduced for A16 = A26 = 0

All plies are at 0 deg in this study, so the transformed Q-bar equals Q
and A16 = A26 = 0. See gap register for the membrane-vs-flexural choice.
Units: MPa and mm throughout. A is in N/mm.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    E1: float        # MPa
    E2: float        # MPa
    nu12: float
    G12: float       # MPa
    t_ply: float     # mm
    rho: float       # kg/m^3
    Xt: float        # MPa, longitudinal tensile strength
    Xc: float        # MPa, longitudinal compressive strength
    Yt: float        # MPa, transverse tensile strength
    Yc: float        # MPa, transverse compressive strength
    S: float         # MPa, in-plane shear strength
    cost_ratio: float


def reduced_stiffness(m: Material) -> np.ndarray:
    """Q for one ply in its material axes. Returns 3x3 in MPa."""
    nu21 = m.nu12 * m.E2 / m.E1
    denom = 1.0 - m.nu12 * nu21
    Q11 = m.E1 / denom
    Q22 = m.E2 / denom
    Q12 = m.nu12 * m.E2 / denom
    return np.array([[Q11, Q12, 0.0],
                     [Q12, Q22, 0.0],
                     [0.0, 0.0, m.G12]])


def ply_boundaries(layup: list[Material]) -> np.ndarray:
    """z coordinates of ply boundaries, measured from the midplane (mm).

    Not needed for A (the differences are just ply thicknesses), but
    required for B and D, and z[-1] == +h/2 is a free correctness check.
    """
    t = np.array([m.t_ply for m in layup])
    h = t.sum()
    return np.concatenate([[-h / 2.0], -h / 2.0 + np.cumsum(t)])


def extensional_stiffness(layup: list[Material]) -> np.ndarray:
    """A = sum over plies of Q * (z_k - z_{k-1}).  3x3 in N/mm."""
    z = ply_boundaries(layup)
    A = np.zeros((3, 3))
    for k, m in enumerate(layup):
        A += reduced_stiffness(m) * (z[k + 1] - z[k])
    return A


def effective_modulus(layup: list[Material]) -> float:
    """E_fx, the laminate's in-plane longitudinal modulus (MPa).

    Kaw Eq. 4.35 is E_x = 1 / (h * A*_11).  With A16 = A26 = 0 the
    inversion has the closed form below.
    """
    A = extensional_stiffness(layup)
    h = sum(m.t_ply for m in layup)
    return (A[0, 0] - A[0, 1] ** 2 / A[1, 1]) / h


def transverse_strain_ratio(A: np.ndarray) -> float:
    """eps_y / eps_x under N_y = N_xy = 0.  Kaw Eq. 4.34."""
    return -A[0, 1] / A[1, 1]


def build_layup(codes: list[int], materials: dict[int, Material]) -> list[Material]:
    """Turn a design-vector layup such as [3,1,1,2,1,2] into Material objects.

    Codes are defined in config/problem.yaml: 1 = CFRP, 2 = GFRP, 3 = Al.
    """
    return [materials[c] for c in codes]