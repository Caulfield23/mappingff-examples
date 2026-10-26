"""
Block-averaged drift test.

Naive OLS slopes on 500 correlated thermo samples are badly underestimated in
uncertainty.  Here every series is block-averaged (blocks of B rows) and the
slope + its standard error are obtained from the scatter of the block means,
which is the standard autocorrelation-robust estimate.

Reported per run:
  * NVT window [1.5e6, 2.0e6]  : Temp / Press / PotEng / Density drift
  * eq_lo NPT stage              : Density drift (is the box still relaxing?)
  * NPT window [1.3e6, 1.5e6)   : Density drift (is rho_npt a stationary mean?)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
MARKERS = ("Temp", "Press", "PotEng", "Density", "Lx")


def parse(log: Path):
    lines = log.read_text(errors="replace").splitlines()
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        if re.match(r"^\s*Step\b", lines[i]) and "Density" in lines[i]:
            cols = lines[i].split()[1:]
            idx = {c: k for k, c in enumerate(cols)}
            j = i + 1
            rows = []
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
                rows.append((step, vals))
                j += 1
            blocks.append((idx, rows))
            i = j
            continue
        i += 1
    out = []
    for idx, rows in blocks:
        for step, vals in rows:
            out.append((step, {c: vals[idx[c]] for c in MARKERS if c in idx}))
    return out


def block_drift(steps, y, B=20):
    """Block-average then OLS; return slope/ns, its SE, t-stat."""
    steps = np.asarray(steps, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    nb = n // B
    if nb < 3:
        return None
    bm = y[:nb * B].reshape(nb, B).mean(axis=1)
    bt = steps[:nb * B].reshape(nb, B).mean(axis=1) / 1e6      # ns
    A = np.vstack([bt, np.ones(nb)]).T
    coef, *_ = np.linalg.lstsq(A, bm, rcond=None)
    resid = bm - A @ coef
    dof = nb - 2
    s2 = float(resid @ resid) / dof
    var = s2 * np.linalg.inv(A.T @ A)[0, 0]
    se = float(np.sqrt(var))
    slope = float(coef[0])
    return {
        "n_rows": n, "n_blocks": nb, "block_size": B,
        "slope_per_ns": slope, "slope_se": se,
        "t_stat": float(slope / se) if se > 0 else float("nan"),
        "mean": float(y.mean()), "std": float(y.std(ddof=1)),
        "half_delta": float(y[n // 2:].mean() - y[:n // 2].mean()),
    }


def main():
    res = []
    for sysname in SYSTEMS:
        for src in SOURCES:
            run = BASE / src / sysname
            rows = parse(run / "log.lammps")
            txt = (run / "log.lammps").read_text(errors="replace")
            m = re.search(r"Loop time of [\d.eE+-]+ on \d+ procs for (\d+) steps", txt)
            nmin = int(m.group(1))
            rec = {"source": src, "system": sysname, "n_min": nmin}

            def sel(lo, hi):
                return [(s, d) for s, d in rows if lo <= s <= hi]

            nvt_lo = nmin + 1_500_000
            sets = {
                "NVT_window_1.5_2.0M": sel(1_500_000, 2_000_000),
                "NVT_stage_exact": sel(nvt_lo, 2_000_000),
                "eq_lo_NPT_full": sel(nmin + 1_000_000, nmin + 1_500_000),
                "eq_lo_tail_100ps": sel(nmin + 1_400_000, nmin + 1_500_000),
                "NPT_window_1.3_1.5M": sel(1_300_000, 1_499_999),
            }
            rec["drift"] = {}
            for name, s in sets.items():
                if not s:
                    continue
                steps = [x[0] for x in s]
                d = {}
                for k in MARKERS:
                    b = block_drift(steps, [x[1][k] for x in s])
                    if b:
                        d[k] = b
                rec["drift"][name] = d

            # residual density drift at the moment the box was frozen
            s_eqlo = sel(nmin + 1_000_000, nmin + 1_500_000)
            rho_eqlo_end = s_eqlo[-1][1]["Density"]
            rho_nvt = rec["drift"]["NVT_stage_exact"]["Density"]["mean"]
            rec["rho_eqlo_last"] = rho_eqlo_end
            rec["rho_nvt_frozen"] = rho_nvt
            rec["rho_gap_eqlo_end_to_nvt"] = rho_nvt - rho_eqlo_end
            rec["rho_gap_pct"] = (rho_nvt - rho_eqlo_end) / rho_nvt * 100
            # pressure in NVT vs the 1 atm NPT target
            rec["press_nvt_mean_atm"] = rec["drift"]["NVT_stage_exact"]["Press"]["mean"]
            rec["press_nvt_slope_per_ns"] = rec["drift"]["NVT_stage_exact"]["Press"]["slope_per_ns"]
            rec["press_nvt_tstat"] = rec["drift"]["NVT_stage_exact"]["Press"]["t_stat"]
            res.append(rec)
            d = rec["drift"]
            print(f"{src}/{sysname}  nmin={nmin}")
            for name in sets:
                if name in d:
                    dd = d[name]
                    print(f"   {name:24s} n={dd['Density']['n_rows']:4d} "
                          f"| T slope={dd['Temp']['slope_per_ns']:9.3f} K/ns (t={dd['Temp']['t_stat']:6.2f}) "
                          f"| P slope={dd['Press']['slope_per_ns']:10.1f} atm/ns (t={dd['Press']['t_stat']:6.2f}) "
                          f"| PE slope={dd['PotEng']['slope_per_ns']:9.2f} kcal/ns (t={dd['PotEng']['t_stat']:6.2f}) "
                          f"| rho slope={dd['Density']['slope_per_ns']:9.5f} /ns (t={dd['Density']['t_stat']:6.2f})")
            print(f"   rho(eq_lo last NPT)={rho_eqlo_end:.6f}  rho(NVT frozen)={rho_nvt:.6f} "
                  f" gap={rec['rho_gap_pct']:+.3f}%   Press_NVT_mean={rec['press_nvt_mean_atm']:.1f} atm")
            print()
    p = Path(__file__).with_name("drift_stats.json")
    p.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
