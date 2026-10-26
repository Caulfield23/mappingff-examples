"""
Global bond-continuity test in the wrapped vs unwrapped gauges + Rg bias
quantification + independent reproduction of the reported Rg values.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
OUT = BASE / "analysis" / "_verify"
exec(open(OUT / "sanity_check.py").read().split("if __name__")[0])


def analyse(src, sysname):
    d = BASE / src / sysname
    mol, mass, natoms, bonds = read_topo(d / "filled.data")
    uniq = np.unique(mol)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol.tolist()], dtype=np.int64)
    nchain = len(uniq)
    w = np.bincount(chain, weights=mass, minlength=nchain)
    ba = np.array([a for a, b in bonds])
    bb = np.array([b for a, b in bonds])

    def com_of(pos):
        c = np.empty((nchain, 3))
        for k in range(3):
            c[:, k] = np.bincount(chain, weights=mass * pos[:, k], minlength=nchain)
        return c / w[:, None]

    def rg_of(pos, com):
        diff = pos - com[chain]
        r2 = np.einsum("ij,ij->i", diff, diff)
        return np.sqrt(np.bincount(chain, weights=mass * r2, minlength=nchain) / w)

    rows = []
    for lines in frames(d / "dump.lammpstrj"):
        step, lo, hi, cols, arr = parse(lines)
        if not (1_500_000 <= step <= 2_000_000):
            continue
        L = hi - lo
        xw = arr[:, [cols.index("x"), cols.index("y"), cols.index("z")]]
        xu = arr[:, [cols.index("xu"), cols.index("yu"), cols.index("zu")]]
        du = np.linalg.norm(xu[ba] - xu[bb], axis=1)
        dw = np.linalg.norm(xw[ba] - xw[bb], axis=1)
        # (a)
        com_a = com_of(xu)
        rg_a = rg_of(xu, com_a)
        # (b)
        rg_b = rg_of(xw, com_of(xw))
        # (c1) naive COM-MIC
        com_w = com_of(xw)
        d1 = xw - com_w[chain]
        for k in range(3):
            d1[:, k] -= L[k] * np.round(d1[:, k] / L[k])
        rg_c1 = rg_of(com_w[chain] + d1, com_w)
        # (c3) MIC relative to true COM
        d3 = xw - com_a[chain]
        for k in range(3):
            d3[:, k] -= L[k] * np.round(d3[:, k] / L[k])
        rg_c3 = rg_of(com_a[chain] + d3, com_a)
        rows.append(dict(step=step, L=float(L[0]),
                         rg_a_mean=float(rg_a.mean()), rg_a_std=float(rg_a.std(ddof=1)),
                         rg_b_mean=float(rg_b.mean()), rg_c1_mean=float(rg_c1.mean()),
                         rg_c3_mean=float(rg_c3.mean()),
                         bond_xu_max=float(du.max()), bond_xu_min=float(du.min()),
                         bond_xu_nbroken=int((du > 3.0).sum()),
                         bond_w_max=float(dw.max()), bond_w_min=float(dw.min()),
                         bond_w_nbroken=int((dw > 3.0).sum()),
                         nbonds=len(bonds),
                         dev_c3_max=float(np.abs(rg_a - rg_c3).max()),
                         dev_b_mean=float(np.abs(rg_a - rg_b).mean()),
                         dev_b_max=float(np.abs(rg_a - rg_b).max()),
                         dev_c1_mean=float(np.abs(rg_a - rg_c1).mean()),
                         dev_c1_max=float(np.abs(rg_a - rg_c1).max())))
    return rows


if __name__ == "__main__":
    runs = [("ligpargen", "9_pva"), ("mappingff", "9_pva"),
            ("ligpargen", "9_peo"), ("mappingff", "9_peo"),
            ("ligpargen", "9_ps"), ("mappingff", "9_ps"),
            ("ligpargen", "9_pmma"), ("mappingff", "9_pmma")]
    allr = {}
    for src, sysn in runs:
        r = analyse(src, sysn)
        allr[f"{src}/{sysn}"] = r
        rg_a = np.mean([x["rg_a_mean"] for x in r])
        rg_b = np.mean([x["rg_b_mean"] for x in r])
        rg_c1 = np.mean([x["rg_c1_mean"] for x in r])
        rg_c3 = np.mean([x["rg_c3_mean"] for x in r])
        nb = r[0]["nbonds"] * len(r)
        print(f"\n=== {src}/{sysn}  frames={len(r)}")
        print(f"  mean Rg  (a) unwrapped = {rg_a:.4f} A")
        print(f"          (b) wrapped   = {rg_b:.4f} A   (bias {100*(rg_b-rg_a)/rg_a:+.2f} %)")
        print(f"        c1 naive COM-MIC= {rg_c1:.4f} A   (bias {100*(rg_c1-rg_a)/rg_a:+.2f} %)")
        print(f"        c3 true-COM MIC = {rg_c3:.4f} A   (bias {100*(rg_c3-rg_a)/rg_a:+.2e} %)")
        print(f"  bonds in unwrapped gauge: len range "
              f"[{min(x['bond_xu_min'] for x in r):.3f}, {max(x['bond_xu_max'] for x in r):.3f}] "
              f" broken(>3A) = {sum(x['bond_xu_nbroken'] for x in r)} / {nb}")
        print(f"  bonds in wrapped   gauge: len range "
              f"[{min(x['bond_w_min'] for x in r):.3f}, {max(x['bond_w_max'] for x in r):.3f}] "
              f" broken(>3A) = {sum(x['bond_w_nbroken'] for x in r)} / {nb}")
        print(f"  |Rg(a)-Rg(c3)| max over all chains/frames = "
              f"{max(x['dev_c3_max'] for x in r):.3e} A")
        print(f"  |Rg(a)-Rg(b)| mean {np.mean([x['dev_b_mean'] for x in r]):.3f} A, "
              f"max {max(x['dev_b_max'] for x in r):.3f} A")
    with open(OUT / "bond_global.json", "w") as fh:
        json.dump(allr, fh, indent=1, default=float)

    # ---- compare with the script's own CSV -------------------------------
    p = BASE / "analysis" / "density_rg_per_run.csv"
    if p.exists():
        print("\n--- script CSV vs my independent (a) ---")
        mine = {k: float(np.mean([x["rg_a_mean"] for x in v])) for k, v in allr.items()}
        with open(p) as fh:
            for row in csv.DictReader(fh):
                k = f"{row['source']}/{row['system']}"
                m = mine.get(k)
                if m is not None:
                    print(f"  {k:24s} script rg_mean={float(row['rg_mean']):.4f}  "
                          f"mine={m:.4f}  diff={float(row['rg_mean'])-m:+.5f}  "
                          f"script rg_std={float(row['rg_std']):.4f} "
                          f"mine per-frame std={np.mean([x['rg_a_std'] for x in allr[k]]):.4f}")
