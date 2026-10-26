"""
Estimate the model's own isothermal compressibility from the eq_lo NPT stage
using the equilibrium fluctuation relation

    dV/dP |_eq  =  cov(V, P) / var(P)        (and  kappa_T = -1/V * dV/dP)

so that the systematic error in the *frozen-box* density can be quantified
without assuming an external compressibility.

The frozen NVT box sits at a volume whose equilibrium pressure is the observed
NVT mean pressure P_nvt, not the 1 atm NPT target.  The density error is then

    d(rho)/rho  ~  kappa_T * (1 atm - P_nvt).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
RUNS = [("9_peo", "ligpargen"), ("9_peo", "mappingff"),
        ("9_pmma", "ligpargen"), ("9_pmma", "mappingff"),
        ("9_ps", "ligpargen"), ("9_ps", "mappingff"),
        ("9_pva", "ligpargen"), ("9_pva", "mappingff")]


def rows_of(log: Path):
    lines = log.read_text(errors="replace").splitlines()
    out = []
    i, n = 0, len(lines)
    while i < n:
        if re.match(r"^\s*Step\b", lines[i]) and "Density" in lines[i]:
            cols = lines[i].split()[1:]
            idx = {c: k for k, c in enumerate(cols)}
            j = i + 1
            while j < n:
                p = lines[j].split()
                if not p:
                    j += 1
                    continue
                try:
                    step = int(p[0])
                except ValueError:
                    break
                if len(p) != len(cols) + 1:
                    break
                try:
                    vals = [float(c) for c in p[1:]]
                except ValueError:
                    break
                out.append((step, vals[idx["Density"]], vals[idx["Lx"]], vals[idx["Press"]]))
                j += 1
            i = j
            continue
        i += 1
    return out


print(f"{'run':22s} {'kappa_T(1/atm)':>16s} {'dV/dP':>10s} {'sigma_V/V%':>11s} "
      f"{'P_nvt(atm)':>10s} {'drho/rho %':>11s}")
print("-" * 95)
for sysname, src in RUNS:
    log = BASE / src / sysname / "log.lammps"
    rows = rows_of(log)
    txt = log.read_text(errors="replace")
    nmin = int(re.search(r"Loop time of [\d.eE+-]+ on \d+ procs for (\d+) steps", txt).group(1))
    eqlo = [r for r in rows if nmin + 1_000_000 <= r[0] <= nmin + 1_500_000]
    nvt = [r for r in rows if nmin + 1_500_000 <= r[0] <= 2_000_000]
    L = np.array([r[2] for r in eqlo])
    V = L ** 3
    P = np.array([r[3] for r in eqlo])
    dVdP = float(np.cov(V, P, ddof=1)[0, 1] / np.var(P, ddof=1))
    kappa = -dVdP / V.mean()
    sv = float(V.std(ddof=1) / V.mean() * 100)
    p_nvt = float(np.mean([r[3] for r in nvt]))
    drho = kappa * (1.0 - p_nvt) * 100      # % : rho_eq(1 atm) - rho_frozen
    print(f"{src+'/'+sysname:22s} {kappa:16.3e} {dVdP:10.4f} {sv:11.4f} "
          f"{p_nvt:10.1f} {drho:11.3f}")
