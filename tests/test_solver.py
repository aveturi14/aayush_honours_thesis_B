
from sandwich.config import load_materials
from sandwich.solver import Panel, evaluate

# Eq. 6 with b reproduces the three Table 7 face sheet stresses (corrects incorrect safety factor feasibility percentages from initial dataset generation)
def test_face_stress_matches_table7():
    MATERIALS, CORES = load_materials()
    cases = [([1]*6, 37, 15.21, 27.53), ([3]*6, 29, 17.46, 14.71),
             ([3, 1, 1, 2, 1, 2], 29, 17.93, 21.30)]
    for codes, rho, tc, published in cases:
        resp = evaluate([MATERIALS[c] for c in codes], CORES[rho], tc, 0.004142, panel=Panel())
        assert abs(resp.sigma_f - published) / published < 0.01