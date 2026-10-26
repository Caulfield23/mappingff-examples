"""Cross-check: import analyze_density_rg (read-only) and compare its
parse_thermo()/density_stats() output against my independent derivation."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
ANA = BASE / "analysis"
sys.path.insert(0, str(ANA))
spec = importlib.util.spec_from_file_location("adg", ANA / "analyze_density_rg.py")
adg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adg)          # __name__ != "__main__" -> main() not run

print("analyze_density_rg.py md5 =",
      hashlib.md5((ANA / "analyze_density_rg.py").read_bytes()).hexdigest())

mine = pd.read_csv(ANA / "_verify" / "audit_density.csv")
print(f"\n{'run':22s} {'rows(anag)':>10s} {'rows(mine)':>10s} {'nA anag':>9s} {'nA mine':>9s} "
      f"{'rhoA anag':>12s} {'rhoA mine':>12s} {'|d|':>9s}  {'nB anag':>8s} {'nB mine':>8s} "
      f"{'rhoB anag':>12s} {'rhoB mine':>12s} {'|d|':>9s}")
print("-" * 160)
ok = True
for _, r in mine.iterrows():
    p = BASE / r.source / r.system / "log.lammps"
    rows, block_first = adg.parse_thermo(p)
    dA = adg.density_stats(rows, adg.STEP_LO, adg.STEP_HI, include_hi=True)
    dB = adg.density_stats(rows, adg.NPT_LO, adg.NPT_HI, include_hi=False)
    dE = adg.density_stats(rows, r.nvt_start, adg.STEP_HI, include_hi=True)
    dA_mine = abs(dA["mean"] - r.my_rhoA)
    dB_mine = abs(dB["mean"] - r.my_rhoB)
    ok &= (len(block_first) == r.n_blocks and dA["n"] == r.my_nA and dB["n"] == r.my_nB
           and dA_mine < 1e-12 and dB_mine < 1e-12
           and abs(dE["mean"] - r.my_rhoE) < 1e-12
           and block_first[-1] == r.nvt_start
           and adg.parse_min_steps(p) == r.n_min_log)
    print(f"{r.source + '/' + r.system:22s} {len(rows):10d} {r.n_blocks:10d} "
          f"{dA['n']:9d} {r.my_nA:9d} {dA['mean']:12.7f} {r.my_rhoA:12.7f} {dA_mine:9.1e}  "
          f"{dB['n']:8d} {r.my_nB:8d} {dB['mean']:12.7f} {r.my_rhoB:12.7f} {dB_mine:9.1e}")
print("\nALL MATCH:", ok)
