"""Final consolidated numbers for the report."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
RUNS = [("9_peo", "ligpargen"), ("9_peo", "mappingff"),
        ("9_pmma", "ligpargen"), ("9_pmma", "mappingff"),
        ("9_ps", "ligpargen"), ("9_ps", "mappingff"),
        ("9_pva", "ligpargen"), ("9_pva", "mappingff")]
KAPPA = 5e-5      # 1/atm, typical isothermal compressibility of an organic melt


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
                out.append((step, vals[idx["Density"]], vals[idx["Lx"]],
                            vals[idx["Press"]], vals[idx["Temp"]]))
                j += 1
            i = j
            continue
        i += 1
    return out


print(f"{'run':22s} {'n_min':>5s} {'nvt_start':>9s} | {'rhoNVT':>9s} {'std_rep':>9s} "
      f"{'std_true':>9s} {'outlier_rho':>11s} {'bias%':>8s} | {'rhoNPT':>9s} {'std':>8s} "
      f"{'d(1.3->1.5)%':>12s} {'eqlo_chg%':>9s} | {'P_nvt(atm)':>10s} {'drho_eq%':>9s}")
print("-" * 160)
summary = []
for sysname, src in RUNS:
    log = BASE / src / sysname / "log.lammps"
    rows = rows_of(log)
    txt = log.read_text(errors="replace")
    nmin = int(re.search(r"Loop time of [\d.eE+-]+ on \d+ procs for (\d+) steps", txt).group(1))
    nvt_start = nmin + 1_500_000
    win = [(s, d, L, P, T) for s, d, L, P, T in rows if 1_500_000 <= s <= 2_000_000]
    nvt = [(s, d, L, P, T) for s, d, L, P, T in rows if nvt_start <= s <= 2_000_000]
    npt = [(s, d, L, P, T) for s, d, L, P, T in rows if 1_300_000 <= s < 1_500_000]
    eqlo = [(s, d, L, P, T) for s, d, L, P, T in rows if nmin + 1_000_000 <= s <= nmin + 1_500_000]
    rho_win = np.array([r[1] for r in win])
    rho_nvt = np.array([r[1] for r in nvt])
    rho_npt = np.array([r[1] for r in npt])
    rho_eqlo = np.array([r[1] for r in eqlo])
    # the single contaminating row: the only window row with step < nvt_start
    outliers = [r for r in win if r[0] < nvt_start]
    out_rho = outliers[0][1] if outliers else float("nan")
    bias = (rho_win.mean() - rho_nvt.mean()) / rho_nvt.mean() * 100
    # density change across the NPT window and across the whole eq_lo stage
    d13 = [r for r in npt if r[0] >= 1_300_000]
    d_npt_change = (rho_npt[-1] - rho_npt[0]) / rho_npt.mean() * 100
    # eq_lo: compare 100 ps mean at start vs 100 ps mean at end
    a = rho_eqlo[:100].mean()
    b = rho_eqlo[-100:].mean()
    eqlo_chg = (b - a) / a * 100
    p_nvt = np.mean([r[3] for r in nvt])
    drho_eq = p_nvt - 1.0            # atm offset from the 1 atm NPT target
    drho_eq_pct = -drho_eq * KAPPA * 100   # sign: P<1 -> V too big -> rho too low
    print(f"{src+'/'+sysname:22s} {nmin:5d} {nvt_start:9d} | {rho_win.mean():9.6f} "
          f"{rho_win.std(ddof=1):9.2e} {rho_nvt.std(ddof=1):9.2e} {out_rho:11.6f} "
          f"{bias:8.5f} | {rho_npt.mean():9.6f} {rho_npt.std(ddof=1):8.5f} "
          f"{d_npt_change:12.3f} {eqlo_chg:9.3f} | {p_nvt:10.1f} {drho_eq_pct:9.3f}")
    summary.append((sysname, src, nmin, nvt_start, rho_win.mean(), rho_win.std(ddof=1),
                    rho_nvt.std(ddof=1), out_rho, bias, rho_npt.mean(), rho_npt.std(ddof=1),
                    d_npt_change, eqlo_chg, p_nvt, drho_eq_pct,
                    len(win), len(nvt)))

print()
print("=== NPT-window density fluctuation = the honest noise floor for rho ===")
for sysname, src in RUNS:
    log = BASE / src / sysname / "log.lammps"
    rows = rows_of(log)
    npt = np.array([r[1] for r in rows if 1_300_000 <= r[0] < 1_500_000])
    print(f"  {src+'/'+sysname:22s} rho_npt std = {npt.std(ddof=1):.5f} "
          f"({npt.std(ddof=1)/npt.mean()*100:.3f} % of rho)")
