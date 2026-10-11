
# classical lamination theory formulas turned into functions acting on the data
from dataclasses import dataclass
import numpy as np

# class definitions for a material, and its corresponding properties
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

# computes Q stiffness matrix for a single ply in x and y axes
def reduced_stiffness(m: Material) -> np.ndarray:
    nu21 = m.nu12 * m.E2 / m.E1
    denom = 1.0 - m.nu12 * nu21
    Q11 = m.E1 / denom
    Q22 = m.E2 / denom
    Q12 = m.nu12 * m.E2 / denom
    return np.array([[Q11, Q12, 0.0],
                     [Q12, Q22, 0.0],
                     [0.0, 0.0, m.G12]])


# returns z coordinates of ply boundaries, measured from the midplane (mm) (essentially just ply thicknesses)
def ply_boundaries(layup: list[Material]) -> np.ndarray:
    t = np.array([m.t_ply for m in layup])
    h = t.sum()
    return np.concatenate([[-h / 2.0], -h / 2.0 + np.cumsum(t)])


# computes A matrix for a laminate 
def extensional_stiffness(layup: list[Material]) -> np.ndarray:
    z = ply_boundaries(layup)
    A = np.zeros((3, 3))
    for k, m in enumerate(layup):
        A += reduced_stiffness(m) * (z[k + 1] - z[k])
    return A

# solves for E_fx, laminates effective modulus  
def effective_modulus(layup: list[Material]) -> float:
    A = extensional_stiffness(layup)
    h = sum(m.t_ply for m in layup)
    return (A[0, 0] - A[0, 1] ** 2 / A[1, 1]) / h

# solves for strain ration in two planes, as a function of A matrix components 
def transverse_strain_ratio(A: np.ndarray) -> float:
    return -A[0, 1] / A[1, 1]

# Turn a design-vector layup such as [3,1,1,2,1,2] into a material object layup
def build_layup(codes: list[int], materials: dict[int, Material]) -> list[Material]:
    return [materials[c] for c in codes]