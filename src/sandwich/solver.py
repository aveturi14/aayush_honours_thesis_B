
from dataclasses import dataclass

from .clt import Material, effective_modulus, transverse_strain_ratio, \
    extensional_stiffness
from .failure import laminate_strength


# class defined for material properties of core
@dataclass(frozen=True)
class Core:
    rho: float          # kg/m^3
    tau_x: float        # MPa, shear strength, x
    G_x: float          # MPa, shear modulus, x
    tau_y: float        # MPa, shear strength, y
    G_y: float          # MPa, shear modulus, y
    sigma_z: float      # MPa, compressive strength, z
    E_z: float          # MPa, compressive modulus, z
    cost_ratio: float = 0.5


# class defined for bending, shear deflection and panel geometry
@dataclass(frozen=True)
class Panel:
    l: float = 800.0        # mm, effective length
    b: float = 390.0        # mm, effective width
    k_b: float = 5.0 / 384.0
    k_s: float = 1.0 / 8.0


# loading 
# equivalent load = p * l * b
def equivalent_load(p: float, panel: Panel) -> float:
    return p * panel.l * panel.b

# max bending moment = P * l / 8
def max_bending_moment(P: float, panel: Panel) -> float:
    return P * panel.l / 8.0

# max shear force = P / 2
def max_shear_force(P: float) -> float:
    return P / 2.0


# geometry 
# total laminate thickness based on a layup
def face_thickness(layup: list[Material]) -> float:
    return sum(m.t_ply for m in layup)

# distance from neutral axis to face sheet
def centroid_distance(t_f: float, t_c: float) -> float:
    return t_f + t_c


# stiffness 
# bending stiffness, (4) from base paper
def bending_stiffness(E_f: float, t_f: float, d: float, panel: Panel) -> float:
    return E_f * t_f * d ** 2 * panel.b / 2.0

# shear stiffness, (5) from base paper
def shear_stiffness(d: float, G_xz: float, panel: Panel) -> float:
    return panel.b * d * G_xz


# responses
# core shear stress, (1) from base paper
def core_shear_stress(F: float, d: float, panel: Panel) -> float:
    return F / (d * panel.b)

# deflection, (3) from base paper
def deflection(P: float, D: float, S: float, panel: Panel) -> float:
    return panel.k_b * P * panel.l ** 3 / D + panel.k_s * P * panel.l / S


# face sheet stress, (6) from base paper
def face_sheet_stress(M_max: float, d: float, t_f: float) -> float:
    return M_max / (d * t_f)

# intracell buckling stress, (7) from base paper
def intracell_buckling_stress(E_fx: float, nu_xy: float,
                              t_f: float, c: float) -> float:
    return 2.0 * E_fx / (1.0 - nu_xy ** 2) * (2.0 * t_f / c) ** 2

# wrinkling stress, (8) from base paper
def wrinkling_stress(E_fx: float, E_zz: float, G_xz: float) -> float:
    return 0.5 * (E_fx * E_zz * G_xz) ** (1.0 / 3.0)

# class defined to store all parameters for a given layup, core, core thickness, applied pressure and cell size
@dataclass(frozen=True)
class Response:
    t_f: float
    d: float
    E_f: float
    sigma_fx: float
    governing_ply: int
    P: float
    M_max: float
    F: float
    D: float
    S: float
    tau_cs: float
    delta: float
    sigma_f: float
    sigma_in_cr: float
    sigma_wr: float


# computes all responses for a given layup, core, core thickness, applied pressure and cell size
def evaluate(layup: list[Material], core: Core, t_c: float, p: float,
             cell_size: float, panel: Panel = Panel()) -> Response:
    t_f = face_thickness(layup)
    d = centroid_distance(t_f, t_c)

    E_f = effective_modulus(layup)
    sigma_fx, k, _ = laminate_strength(layup)

    # Laminate Poisson ratio for Eq. 7.  nu_xy = -eps_y/eps_x under N_y = 0.
    nu_xy = -transverse_strain_ratio(extensional_stiffness(layup))

    P = equivalent_load(p, panel)
    M_max = max_bending_moment(P, panel)
    F = max_shear_force(P)

    D = bending_stiffness(E_f, t_f, d, panel)
    S = shear_stiffness(d, core.G_x, panel)

    return Response(
        t_f=t_f, d=d, E_f=E_f, sigma_fx=sigma_fx, governing_ply=k,
        P=P, M_max=M_max, F=F, D=D, S=S,
        tau_cs=core_shear_stress(F, d, panel),
        delta=deflection(P, D, S, panel),
        sigma_f=face_sheet_stress(M_max, d, t_f),
        sigma_in_cr=intracell_buckling_stress(E_f, nu_xy, t_f, cell_size),
        sigma_wr=wrinkling_stress(E_f, core.E_z, core.G_x),
    )