
# imports functions from clt.py, config.py and failure.py  
from sandwich.config import load_materials, load_problem
from sandwich.objectives import total_cost, total_weight


def main():
    # take material properties from materials.yaml
    MATERIALS, CORES = load_materials()
    
    # take problem configuration from problem.yaml (table 5 and 6)
    prob = load_problem()

    # Compute weight and cost for each Pareto design and compare to published values
    # Reproduce Table 5 from the thesis, with weight error and cost ratio columns.
    print("\nTABLE 5 PARETO DESIGNS")
    print(f"{'#':>3} {'Wt pub':>7} {'Wt calc':>8} {'err %':>7}   "
          f"{'Ct pub':>7} {'Ct calc':>9} {'ratio':>7}  layup")
    print("-" * 78)

    w_errs, c_ratios = [], []
    
    # for each pareto design, compute weight and cost, compare to values in table 5, and compute error and ratio accordingly storing in
    # array w_errs and c_ratios  
    for r in prob["pareto_designs"]["rows"]:
        layup = [MATERIALS[c] for c in r["layup"]]
        core = CORES[r["rho_c"]]

        W = total_weight(layup, core, r["tc"])
        C = total_cost(layup, core, r["tc"])
        w_err = 100 * (W - r["Wt"]) / r["Wt"]
        c_ratio = C / r["Ct"]
        w_errs.append(w_err)
        c_ratios.append(c_ratio)

        print(f"{r['n']:3d} {r['Wt']:7.2f} {W:8.3f} {w_err:+7.1f}   "
              f"{r['Ct']:7.2f} {C:9.3f} {c_ratio:7.2f}  "
              f"{''.join(str(c) for c in r['layup'])}")

    print("-" * 78)
    n_al = [sum(1 for c in r["layup"] if c == 3)
            for r in prob["pareto_designs"]["rows"]]
    print(f"Weight: mean err {sum(w_errs)/len(w_errs):+.2f}%, "
          f"max |err| {max(abs(e) for e in w_errs):.2f}%")
    print(f"Cost:   ratio {min(c_ratios):.2f} to {max(c_ratios):.2f}, "
          f"mean {sum(c_ratios)/len(c_ratios):.3f}")

if __name__ == "__main__":
    main()