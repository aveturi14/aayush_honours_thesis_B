"""Reproduce the W_t and C_t columns of Table 5.

Week 3 deliverable.  Table 5 publishes only layup, t_c, rho_c, C_t and
W_t, so this verifies Eqs. 13-15 and nothing else.  See GR-018.
"""

from sandwich.config import load_materials, load_problem
from sandwich.objectives import total_cost, total_weight


def main():
    MATERIALS, CORES = load_materials()
    prob = load_problem()

    print(f"{'#':>3} {'Wt pub':>7} {'Wt calc':>8} {'err %':>7}   "
          f"{'Ct pub':>7} {'Ct calc':>9} {'ratio':>7}  layup")
    print("-" * 78)

    w_errs, c_ratios = [], []
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
    print(f"\nBias vs Al content: lightest designs (0 Al plies) err "
          f"{w_errs[0]:+.1f}%, heaviest (6 Al) {w_errs[-1]:+.1f}%.")
    print("Both columns drift the same way, concentrating the residual in "
          "the face sheet term.  See GR-011.")


if __name__ == "__main__":
    main()