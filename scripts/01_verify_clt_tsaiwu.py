
# imports functions from clt.py, config.py and failure.py  
from sandwich.clt import (extensional_stiffness, effective_modulus,
                          reduced_stiffness, transverse_strain_ratio)
from sandwich.config import load_materials, load_problem
from sandwich.failure import coefficients, laminate_strength


def main():
    # take material properties from materials.yaml
    MATERIALS, _ = load_materials()
    
    # take problem configuration from problem.yaml (table 5 and 6)
    prob = load_problem()

    # Compute Q matrix using classical laminate theory functions
    print("REDUCED STIFFNESS Q  [MPa]")
    print(f"{'':6} {'Q11':>12} {'Q22':>12} {'Q12':>12} {'Q66':>12}")
    for code in (1, 2, 3):
        m = MATERIALS[code]
        Q = reduced_stiffness(m)
        print(f"{m.name:<6} {Q[0,0]:>12.1f} {Q[1,1]:>12.1f} "
              f"{Q[0,1]:>12.1f} {Q[2,2]:>12.1f}")

    # Compute Tsai-Wu coefficients
    print("\nTSAI-WU COEFFICIENTS  (Mises-Hencky H12)")
    print(f"{'':6} {'H1':>12} {'H11':>12} {'H2':>12} {'H22':>12} {'H12':>12}")
    for code in (1, 2, 3):
        m = MATERIALS[code]
        c = coefficients(m)
        print(f"{m.name:<6} {c.H1:>12.4e} {c.H11:>12.4e} "
              f"{c.H2:>12.4e} {c.H22:>12.4e} {c.H12:>12.4e}")

    # For each reference design, compute laminate properties and compare to published values
    print("\nTABLE 6 LAYUPS")
    for name, d in prob["reference_designs"].items():
        layup = [MATERIALS[c] for c in d["layup"]]
        A = extensional_stiffness(layup)
        E_f = effective_modulus(layup)
        sigma_fx, k, ratios = laminate_strength(layup)
        t_f = sum(m.t_ply for m in layup)

        print(f"\n  {name}  {d['layup']}   ({d.get('note','')})")
        print(f"    t_f        {t_f:.3f} mm   published {d['tf']['value']:.3f}  "
              f"{'OK' if abs(t_f - d['tf']['value']) < 1e-9 else 'MISMATCH'}")
        print(f"    A11 {A[0,0]:>10.1f}   A12 {A[0,1]:>9.1f}   A22 {A[1,1]:>9.1f}  N/mm")
        print(f"    A16 {A[0,2]:>10.1f}   A26 {A[1,2]:>9.1f}   (must be zero)")
        print(f"    eps_y/eps_x  {transverse_strain_ratio(A):.6f}")
        print(f"    E_f        {E_f:>10.1f} MPa")
        print(f"    R per ply  {[f'{r:.6f}' for r in ratios]}")
        print(f"    sigma_fx   {sigma_fx:>10.2f} MPa   governed by {layup[k].name}")

    # Again eye checks for theorems, specifically single material laminates (CFRP and Al) to ensure that 
    # effective modulus and laminate strength values are consistent. 
    print("\nSELF-CHECKS  (theorems, not recorded values)")
    for code, label in [(1, "CFRP"), (3, "Al")]:
        m = MATERIALS[code]
        E = effective_modulus([m] * 6)
        s, _, _ = laminate_strength([m] * 6)
        print(f"  all-{label:<5} E_f {E:>9.1f} vs E1 {m.E1:>9.1f}   "
              f"sigma_fx {s:>8.2f} vs Xt {m.Xt:>8.2f}   "
              f"{'OK' if abs(E - m.E1) < 1 and abs(s - m.Xt) < 0.1 else 'FAIL'}")


if __name__ == "__main__":
    main()