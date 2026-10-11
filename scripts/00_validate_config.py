
# lets python read yaml files
import yaml

# imports functions from config.py to load yaml files and get configuration directory
from sandwich.config import CONFIG_DIR, load_materials

# banner function to print a banner with a given string
def banner(s):
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")

# main function to validate the configuration files
def main():
      
    raw = yaml.safe_load(open(CONFIG_DIR / "materials.yaml"))

    # print the face sheet material properties (matching format to base paper)
    banner("TABLE 1 - face sheet material properties")
    keys = ["E1", "E2", "nu12", "G12", "rho", "t_ply"]
    sk = ["Xt", "Xc", "Yt", "Yc", "S"]
    print(f"{'property':<10} {'CFRP':>12} {'GFRP':>12} {'Al':>12}  unit")
    for k in keys:
        row = [raw["face_sheets"][n][k] for n in ("CFRP", "GFRP", "Al")]
        print(f"{k:<10} " + " ".join(f"{r['value']:>12g}" for r in row)
              + f"  {row[0].get('unit', '')}")
    for k in sk:
        row = [raw["face_sheets"][n]["strength"][k] for n in ("CFRP", "GFRP", "Al")]
        print(f"{k:<10} " + " ".join(f"{r['value']:>12g}" for r in row)
              + f"  {row[0].get('unit', '')}")
    row = [raw["face_sheets"][n]["cost_ratio"] for n in ("CFRP", "GFRP", "Al")]
    print(f"{'cost_ratio':<10} " + " ".join(f"{r['value']:>12g}" for r in row))
    
    # print the core material properties (matching format to base paper)
    banner("TABLE 2 - core properties")
    print(f"{'rho_c':>7} {'tau_x':>8} {'G_x':>8} {'tau_y':>8} {'G_y':>8} "
          f"{'sig_z':>8} {'E_z':>8} {'cost':>6}")
    for c in raw["cores"].values():
        print(f"{c['density']['value']:>7g} "
              f"{c['x']['strength']['value']:>8g} {c['x']['modulus']['value']:>8g} "
              f"{c['y']['strength']['value']:>8g} {c['y']['modulus']['value']:>8g} "
              f"{c['z']['strength']['value']:>8g} {c['z']['modulus']['value']:>8g} "
              f"{c['cost_ratio']['value']:>6g}")

    # print the problem configuration 
    banner("TABLE 3 - geometry and loading")
    prob = yaml.safe_load(open(CONFIG_DIR / "problem.yaml"))
    for sect in ("geometry", "beam_coefficients", "nominal_case"):
        for k, d in prob[sect].items():
            if isinstance(d, dict) and "value" in d:
                print(f"  {k:<8} {d['value']:>14g}  {d.get('unit',''):<8} "
                      f"{d.get('source','')}")
    print(f"  {'BC':<8} {prob['boundary_conditions']}")

    # print the design variable bounds
    banner("TABLE 4 - design variable bounds")
    ds = yaml.safe_load(open(CONFIG_DIR / "design_space.yaml"))
    print(yaml.dump(ds, sort_keys=False, default_flow_style=False))

    # validates / verifies material properties inputs to see if any errors in material / yaml file definitions
    banner("CROSS-CHECKS")
    mats, cores = load_materials()
    ds_rho = ds["discrete"]["core_density"]["values"]
    missing = [r for r in ds_rho if r not in cores]
    print(f"  core densities in design_space present in materials: "
          f"{'OK' if not missing else f'MISSING {missing}'}")
    print(f"  material codes: " +
          ", ".join(f"{k}={v.name}" for k, v in sorted(mats.items())))

    # Al isotropy
    al = mats[3]
    
    # calculating shear modulus for isotropic material using the formula G = E / (2 * (1 + nu))
    # verifying rough agreement to assigned material property G12 for aluminum
    g_iso = al.E1 / (2 * (1 + al.nu12))
    print(f"  Al G12: published {al.G12:.0f}, isotropic {g_iso:.0f}, "
          f"diff {100*(al.G12-g_iso)/g_iso:+.1f}%  (GR-006)")

    # checks material hierarchy for eye test
    print(f"  E1 hierarchy CFRP > Al > GFRP: "
          f"{mats[1].E1:.0f} > {mats[3].E1:.0f} > {mats[2].E1:.0f}  "
          f"{'OK' if mats[1].E1 > mats[3].E1 > mats[2].E1 else 'FAIL'}  (GR-005)")


if __name__ == "__main__":
    main()