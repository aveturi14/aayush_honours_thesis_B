
import numpy as np
import pandas as pd
from scipy.stats import chisquare, qmc

from .config import load_design_space, load_materials
from .constraints import deflection_limit, safety_factors, total_thickness
from .objectives import total_cost, total_weight
from .solver import Panel, evaluate

LAYER_COLS = [f"L{k}" for k in range(1,7)]
INPUT_COLS = ["rho_c", "t_c", *LAYER_COLS, "p"]

OUTPUT_COLS = ["SF_c", "SF_f", "SF_int", "SF_wr", "C_delta", "t_sw", "C_t", "W_t"]

def sampling_ranges(ds: dict | None = None) -> dict:
    
    # load the design space from yaml material files
    ds = ds or load_design_space()
    cont, disc = ds["continuous"], ds["discrete"]
    tc = cont["core_thickness"]["sampling"]
    p = cont["applied_load"]
    
    return {
        "t_c": (tc["min"]["value"], tc["max"]["value"]),
        "p": (p["min"]["value"], p["max"]["value"]),
        "rho_c": (disc["core_density"]["values"]),
        "materials": disc["layer_material"]["values"],
        "n_layers": disc["layer_material"]["n_layers"],
    }
    
def sample_designs(n: int, seed: int, ds: dict | None = None) -> pd.DataFrame:
    r = sampling_ranges(ds)
    
    if r["n_layers"] != len(LAYER_COLS):
        raise ValueError(f"n_layers is {r['n_layers']}, ANN expects {len(LAYER_COLS)}")
        
    rng = np.random.default_rng(seed)
    
    df = pd.DataFrame({"rho_c": rng.choice(r["rho_c"], size=n)})
    
    df["t_c"] = rng.uniform(*r["t_c"], size=n)
    
    layers = rng.choice(r["materials"], size=(n, r["n_layers"]))
    for k, col in enumerate(LAYER_COLS):
        df[col] = layers[:, k]
        
    df["p"] = rng.uniform(*r["p"], size=n)
    return df[INPUT_COLS]

def evaluate_designs(df: pd.DataFrame, cell_size: float | None = None,
                     cost_normalisation: float = 1.0) -> pd.DataFrame:
    MATERIALS, CORES = load_materials()
    panel = Panel()
    
    rows = []
    for rec in df.itertuples(index=False):
        layup = [MATERIALS[int(getattr(rec, col))] for col in LAYER_COLS]
        core = CORES[int(rec.rho_c)]
        
        resp = evaluate(layup, core, rec.t_c, rec.p, cell_size, panel)
        
        sf = safety_factors(resp, core)
        
        rows.append((
            sf.SF_c, sf.SF_f, sf.SF_int, sf.SF_wr,
            deflection_limit(resp, panel.l),    # Eq. 26
            total_thickness(resp.t_f, rec.t_c),     # Eq. 27
            total_cost(layup, core, rec.t_c, normalisation=cost_normalisation),     # Eq. 15
            total_weight(layup, core, rec.t_c)      # Eq. 13
        ))
        
    out = pd.DataFrame(rows, columns=OUTPUT_COLS, index=df.index)
    
    # stop if any part of space breaks the closed-form model
    bad = ~np.isfinite(out.to_numpy()).all(axis=1)
    if bad.any():
        raise ValueError(f"{bad.sum()} non-finite designs, first rwos: {np.where(bad)[0][:5]}")
    return pd.concat([df, out], axis=1)

def feasibility_flags(df: pd.DataFrame, ds: dict | None = None) -> pd.DataFrame:
    lim = (ds or load_design_space())["constraints"]
    sf_min = lim["min_safety_factor"]["value"]
    
    flags = pd.DataFrame({
        "ok_SF_c":  df.SF_c >= sf_min,  # Eq. 22
        "ok_SF_f":  df.SF_f >= sf_min,  # Eq. 23
        "ok_SF_int":  df.SF_int >= sf_min,  # Eq. 24
        "ok_SF_wr":  df.SF_wr >= sf_min,    # Eq. 25
        "ok_C_delta":  df.C_delta <= 1.0,   # Eq. 26
        "ok_t_sw":  df.t_sw <= lim["max_total_thickness"]["value"],     # Eq. 27
    })
    
    # feasible only if all six pass at once
    flags["feasible"] = flags.all(axis=1)
    return flags

def coverage_report(df: pd.DataFrame, ds: dict | None = None) -> dict:
    r = sampling_ranges(ds)
    rep, unit = {}, []
    
    for col in ("t_c", "p"):
        lo, hi = r[col]
        counts, _ = np.histogram(df[col], bins=10, range=(lo, hi))
        rep[col] = {
            "bounds": [lo, hi],
            "observed": [float(df[col].min()), float(df[col].max())],
            "bin_counts": counts.tolist(),
            "chi2_p": float(chisquare(counts).pvalue),
        }
        unit.append((df[col] - lo) / (hi - lo))
        
    rep["discrepancy_tc_p"] = float(qmc.discrepancy(np.column_stack(unit)))
    
    levels = {"rho_c": r["rho_c"], **{col: r["materials"] for col in LAYER_COLS}}
    for col, vals in levels.items():
        counts = df[col].value_counts().reindex(vals, fill_value=0)
        rep[col] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "chi2_p": float(chisquare(counts.to_numpy()).pvalue)
        }
    return rep