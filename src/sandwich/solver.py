"""Analytical forward solver: Eqs. 1-8 of Sahib & Kovacs (2024).

Design vector in, structural responses out.  Pure functions, no globals,
no config reads.  Units are MPa, mm, N throughout.
"""

from dataclasses import dataclass

from .clt import Material, effective_modulus, transverse_strain_ratio, \
    extensional_stiffness
from .failure import laminate_strength


@dataclass(frozen=True)
class Core:
    """One row of Table 2.  Axis assignment taken from the column headers;
    the printed subscripts are identical across all three columns (GR-010).
    """
    rho: float          # kg/m^3
    tau_x: float        # MPa, shear strength, x
    G_x: float          # MPa, shear modulus, x
    tau_y: float        # MPa, shear strength, y
    G_y: float          # MPa, shear modulus, y
    sigma_z: float      # MPa, compressive strength, z
    E_z: float          # MPa, compressive modulus, z
    cost_ratio: float = 0.5


@dataclass(frozen=True)
class Panel:
    """Table 3 geometry and the beam coefficients."""
    l: float = 800.0        # mm, effective length
    b: float = 390.0        # mm, effective width
    k_b: float = 5.0 / 384.0
    k_s: float = 1.0 / 8.0


# --- loading ---------------------------------------------------------------

def equivalent_load(p: float, panel: Panel) -> float:
    """P = p * l * b   (Table 3).  N."""
    return p * panel.l * panel.b


def max_bending_moment(P: float, panel: Panel) -> float:
    """M_max = P * l / 8   (Table 3).  N.mm."""
    return P * panel.l / 8.0


def max_shear_force(P: float) -> float:
    """F = P / 2   (Table 3).  N."""
    return P / 2.0


# --- geometry --------------------------------------------------------------

def face_thickness(layup: list[Material]) -> float:
    """t_f = sum N_li * t_li   (Eq. 14).  mm."""
    return sum(m.t_ply for m in layup)


def centroid_distance(t_f: float, t_c: float) -> float:
    """d = t_f + t_c   (Eq. 2).  mm.

    Distance between the centres of the upper and lower face sheets.
    """
    return t_f + t_c


# --- stiffness -------------------------------------------------------------

def bending_stiffness(E_f: float, t_f: float, d: float, panel: Panel) -> float:
    """D = E_f * t_f * d^2 * b / 2   (Eq. 4).  N.mm^2."""
    return E_f * t_f * d ** 2 * panel.b / 2.0


def shear_stiffness(d: float, G_xz: float, panel: Panel) -> float:
    """S = b * d * G_xz   (Eq. 5).  N."""
    return panel.b * d * G_xz


# --- responses -------------------------------------------------------------

def core_shear_stress(F: float, d: float, panel: Panel) -> float:
    """tau_cs = F / (d * b)   (Eq. 1).  MPa."""
    return F / (d * panel.b)


def deflection(P: float, D: float, S: float, panel: Panel) -> float:
    """delta = k_b*P*l^3/D + k_s*P*l/S   (Eq. 3).  mm.

    Bending term plus shear term.
    """
    return panel.k_b * P * panel.l ** 3 / D + panel.k_s * P * panel.l / S


def face_sheet_stress(M_max: float, d: float, t_f: float) -> float:
    """sigma_f = M_max / (d * t_f)   (Eq. 6).  MPa."""
    return M_max / (d * t_f)


def intracell_buckling_stress(E_fx: float, nu_xy: float,
                              t_f: float, c: float) -> float:
    """sigma_in,cr = 2*E_fx/(1 - nu_xy^2) * (2*t_f/c)^2   (Eq. 7).  MPa.

    c is the honeycomb cell size, which the paper never tabulates.
    See gap register GR-001.
    """
    return 2.0 * E_fx / (1.0 - nu_xy ** 2) * (2.0 * t_f / c) ** 2


def wrinkling_stress(E_fx: float, E_zz: float, G_xz: float) -> float:
    """sigma_wr,x = 0.5 * (E_fx * E_zz * G_xz)^(1/3)   (Eq. 8).  MPa."""
    return 0.5 * (E_fx * E_zz * G_xz) ** (1.0 / 3.0)


# --- driver ----------------------------------------------------------------

@dataclass(frozen=True)
class Response:
    """Everything Eqs. 1-8 produce for one design."""
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


def evaluate(layup: list[Material], core: Core, t_c: float, p: float,
             cell_size: float, panel: Panel = Panel()) -> Response:
    """Run Eqs. 1-8 for one design point."""
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