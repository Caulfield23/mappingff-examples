"""Autocorrelation check on the reported rho_npt +- std/sqrt(n) error bar."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
ANA = BASE / "analysis"

HDR = re.compile(
    r"(?m)^\s*Step\s+[^\n]*\bDensity\b[^\n]*$\n(?P<rows>(?:[ \t]*\d+[^\n]*\n)+)")


def density_series(path, lo, hi):
    text = path.read_text(errors="replace")
    out = []
    for m in HDR.finditer(text):
        cols = text[m.start():m.start() + text[m.start():].index("\n") + 1].split()
        di = cols.index("Density")
        for line in m.group("rows").splitlines():
            p = line.split()
            s = int(p[0])
            if lo <= s <= hi:
                out.append(float(p[di]))
    return np.asarray(out)


print(f"{'run':22s} {'n':>4s} {'std':>10s} {'naive SEM':>11s} "
      f"{'lag1 ac':>9s} {'tau_int(ps)':>11s} {'corr SEM':>10s} {'naive/corr':>10s}")
print("-" * 100)
recs = []
for sysname in ("9_peo", "9_pmma", "9_ps", "9_pva"):
    for src in ("ligpargen", "mappingff"):
        d = density_series(BASE / src / sysname / "log.lammps", 1_300_000, 1_499_999)
        n = len(d)
        x = d - d.mean()
        var = (x ** 2).sum() / n
        # autocorrelation time via initial positive sequence
        g = np.empty(n)
        for k in range(n):
            g[k] = (x[:n - k] * x[k:]).sum() / n if k else var
        tau = 0.5
        for k in range(1, n):
            if g[k] <= 0:
                break
            tau += g[k] / var
        lag1 = g[1] / var
        naive = np.sqrt(var / n)
        corr = np.sqrt(tau * var / n)
        print(f"{src + '/' + sysname:22s} {n:4d} {np.sqrt(var):10.6f} {naive:11.6f} "
              f"{lag1:9.3f} {tau:11.2f} {corr:10.6f} {corr / naive:10.2f}")
        recs.append(dict(system=sysname, source=src, n=n, std=float(np.sqrt(var)),
                         naive_sem=float(naive), lag1=float(lag1), tau_ps=float(tau),
                         corr_sem=float(corr)))
pd.DataFrame(recs).to_csv(ANA / "_verify" / "autocorr.csv", index=False)
print("\ntau_int is in units of the 1000-step (1 ps) thermo sampling interval.")
