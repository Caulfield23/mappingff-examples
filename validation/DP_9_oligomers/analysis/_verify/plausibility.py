"""Physical plausibility table for the simulated oligomer melt densities."""
import numpy as np
import pandas as pd
from pathlib import Path

ANA = Path(r"C:\schoolwork\Projects\mappingff\validation\longer\analysis")
df = pd.read_csv(ANA / "density_rg_per_run.csv")

AMU_A3 = 1.66053906660          # 1 amu/A^3 -> g/cm^3

# Reference data (literature / measured)
REF = {
    # system: (bulk melt rho, lo, hi, oligomer-analogue rho, analogue name)
    "9_peo":  (1.225, 1.20, 1.25, 1.127, "PEG-400 (C18H38O10) liquid, 20-25 C"),
    "9_pmma": (1.190, 1.18, 1.20, 1.156, "PMMA DP=9 via Flory 1/M extrap. from MMA (0.940)"),
    "9_ps":   (1.075, 1.05, 1.10, 1.032, "PS DP=9 via Flory 1/M extrap. from styrene (0.909)"),
    "9_pva":  (1.250, 1.20, 1.30, 1.127, "PEG-like polyol analogue (PEG-400 liq 1.127)"),
}
NATOMS = {"9_peo": 6600, "9_pmma": 13700, "9_ps": 14600, "9_pva": 6500}
MASS = {"9_peo": 41449.2, "9_pmma": 90306.9, "9_ps": 93938.4, "9_pva": 39849.3}
# measured monomer / smallest-oligomer liquid densities
MONO = {"9_peo": 1.113, "9_pmma": 0.940, "9_ps": 0.909, "9_pva": 1.024}
MONO_NAME = {"9_peo": "ethylene glycol", "9_pmma": "MMA monomer",
             "9_ps": "styrene monomer", "9_pva": "1,6-hexanediol"}

rows = []
for _, r in df.iterrows():
    s = r.system
    bulk, lo, hi, ana, ana_name = REF[s]
    nat = NATOMS[s]
    M = MASS[s]
    rho_exact = r.rho_nvt_exact_mean          # clean, frozen-box NVT density
    rho_nvt = r.rho_nvt_mean                  # nominal window (1 NPT row included)
    rho_npt = r.rho_npt_mean
    for label, rho in (("rho_nvt_mean (nominal window)", rho_nvt),
                       ("rho_nvt_exact (clean NVT)", rho_exact),
                       ("rho_npt_mean", rho_npt)):
        V = M * AMU_A3 / rho                  # A^3  (rho[g/cm3] = M/V * 1.66054)
        rows.append(dict(
            system=s, source=r.source, quantity=label, rho=rho,
            vs_bulk_pct=(rho - bulk) / bulk * 100,
            vs_bulk_range_lo_pct=(rho - lo) / lo * 100,
            vs_bulk_range_hi_pct=(rho - hi) / hi * 100,
            vs_oligomer_analogue_pct=(rho - ana) / ana * 100,
            vs_monomer_pct=(rho - MONO[s]) / MONO[s] * 100,
            A3_per_atom=V / nat,
            bulk_A3_per_atom=(M * AMU_A3 / bulk) / nat,
        ))

out = pd.DataFrame(rows)
out.to_csv(ANA / "_verify" / "plausibility.csv", index=False)

pd.set_option("display.width", 250)
for s in ("9_peo", "9_pmma", "9_ps", "9_pva"):
    sub = out[out.system == s]
    print("=" * 130)
    bulk, lo, hi, ana, ana_name = REF[s]
    print(f"{s}: bulk melt rho={bulk} (range {lo}-{hi});  oligomer analogue = {ana:.3f} [{ana_name}]; "
          f"monomer/parent liquid = {MONO[s]:.3f} [{MONO_NAME[s]}]")
    print(f"  {'quantity':30s} {'source':10s} {'rho':>9s} {'vs bulk %':>10s} "
          f"{'vs olg %':>10s} {'vs mono %':>10s} {'A3/atom':>8s} {'bulk A3/atom':>13s} {'excess %':>9s}")
    for _, r in sub.iterrows():
        exc = (r.A3_per_atom / r.bulk_A3_per_atom - 1) * 100
        print(f"  {r.quantity:30s} {r.source:10s} {r.rho:9.5f} {r.vs_bulk_pct:10.2f} "
              f"{r.vs_oligomer_analogue_pct:10.2f} {r.vs_monomer_pct:10.2f} "
              f"{r.A3_per_atom:8.2f} {r.bulk_A3_per_atom:13.2f} {exc:9.1f}")
print("\nwrote", ANA / "_verify" / "plausibility.csv")
