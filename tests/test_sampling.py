
from sandwich.config import load_design_space, load_materials
from sandwich.constraints import (DEFLECTION_DIVISOR, MAX_THICKNESS_MM, SafetyFactors, is_feasible)
from sandwich.sampling import (LAYER_COLS, evaluate_designs, feasibility_flags, sample_designs, sampling_ranges)

def test_sampler_reproducible():
    assert sample_designs(300, 1).equals(sample_designs(300, 1))
    assert not sample_designs(300, 1).equals(sample_designs(300, 2))
        

def test_every_core_has_cell_size():
    r = sampling_ranges()
    df = sample_designs(9000, 42)
    assert df.t_c.between(*r["t_c"]).all()
    assert df.p.between(*r["p"]).all()
    assert set(df.rho_c) == set(r["rho_c"])
    for col in LAYER_COLS:
        assert set(df[col]) == set(r["materials"])
        

def test_every_core_has_cell_size():
    _, cores = load_materials()
    for rho, core in cores.items():
        assert core.cell_size is not None and core.cell_size > 0, f"core {rho} has no cell size"
        
def test_yaml_limits_match_constraints_module():
    lim = load_design_space()["constraints"]
    assert lim["max_total_thickness"]["value"] == MAX_THICKNESS_MM
    assert lim["deflection_divisor"]["value"] == DEFLECTION_DIVISOR
    
def test_flags_agree_with_is_feasible():
    df = evaluate_designs(sample_designs(200, 0))
    flags = feasibility_flags(df)
    for rec, ok in zip(df.itertuples(), flags.feasible):
        sf = SafetyFactors(rec.SF_c, rec.SF_f, rec.SF_int, rec.SF_wr)
        assert is_feasible(sf, rec.C_delta, rec.t_sw) == ok
    
def test_cell_size_override_only_moves_sf_int():
    designs = sample_designs(50, 3)
    a = evaluate_designs(designs)
    b = evaluate_designs(designs, cell_size=3.0)
    others = ["SF_c", "SF_f", "SF_wr", "C_delta", "t_sw", "C_t", "W_t"]
    assert a[others].equals(b[others])
    assert not a["SF_int"].equals(b["SF_int"])
    
