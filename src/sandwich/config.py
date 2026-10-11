
# Load config YAML into material and core objects.
from pathlib import Path
import yaml
from .clt import Material
from .solver import Core

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _v(d: dict) -> float:
    val = d["value"]
    if d.get("unit") == "GPa":
        return val * 1000.0
    return val

# read materials.yaml, and configure material / core properties to relevant variables that can be used in solver
def load_materials(path: Path | None = None) -> tuple[dict[int, Material], dict[int, Core]]:
    path = path or CONFIG_DIR / "materials.yaml"
    cfg = yaml.safe_load(open(path))

    mats = {}
    for code, name in [(1, "CFRP"), (2, "GFRP"), (3, "Al")]:
        m = cfg["face_sheets"][name]
        s = m["strength"]
        mats[code] = Material(
            name=name,
            E1=_v(m["E1"]), E2=_v(m["E2"]), nu12=_v(m["nu12"]), G12=_v(m["G12"]),
            t_ply=_v(m["t_ply"]), rho=_v(m["rho"]),
            Xt=_v(s["Xt"]), Xc=_v(s["Xc"]), Yt=_v(s["Yt"]), Yc=_v(s["Yc"]),
            S=_v(s["S"]), cost_ratio=_v(m["cost_ratio"]),
        )

    cores = {}
    for c in cfg["cores"].values():
        rho = _v(c["density"])
        cores[rho] = Core(
            rho=rho,
            tau_x=_v(c["x"]["strength"]), G_x=_v(c["x"]["modulus"]),
            tau_y=_v(c["y"]["strength"]), G_y=_v(c["y"]["modulus"]),
            sigma_z=_v(c["z"]["strength"]), E_z=_v(c["z"]["modulus"]),
            cost_ratio=_v(c["cost_ratio"]),
        )
    return mats, cores

# read problem.yaml and return readable object for the problem
def load_problem(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "problem.yaml"
    return yaml.safe_load(open(path))

# read design_space.yaml and return readable object for design space
def load_design_space(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "design_space.yaml"
    return yaml.safe_load(open(path))