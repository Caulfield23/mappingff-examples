"""Final summary: straddle tables, image-offset integrality, and what the
ligpargen-vs-mappingff Rg comparison looks like under each PBC treatment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
OUT = BASE / "analysis" / "_verify"
exec(open(OUT / "sanity_check.py").read().split("if __name__")[0])


def frame_stats(src, sysn):
    d = BASE / src / sysn
    mol, mass, natoms, bonds = read_topo(d / "filled.data")
    uniq = np.unique(mol)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol.tolist()], dtype=np.int64)
    nchain = len(uniq)
    w = np.bincount(chain, weights=mass, minlength=nchain)
    ba = np.array([a for a, b in bonds]); bb = np.array([b for a, b in bonds])

    def com_of(pos):
        c = np.empty((nchain, 3))
        for k in range(3):
            c[:, k] = np.bincount(chain, weights=mass * pos[:, k], minlength=nchain)
        return c / w[:, None]

    def rg_of(pos, com):
        diff = pos - com[chain]
        r2 = np.einsum("ij,ij->i", diff, diff)
        return np.sqrt(np.bincount(chain, weights=mass * r2, minlength=nchain) / w)

    out = dict(steps=[], multimage=[], span=[], img_resid_max=[], rga=[],
               rgb=[], rgc1=[], rgc3=[], L=[])
    for lines in frames(d / "dump.lammpstrj"):
        step, lo, hi, cols, arr = parse(lines)
        if not (1_500_000 <= step <= 2_000_000):
            continue
        L = hi - lo
        xw = arr[:, [cols.index("x"), cols.index("y"), cols.index("z")]]
        xu = arr[:, [cols.index("xu"), cols.index("yu"), cols.index("zu")]]
        # image offsets: n = (xu-x)/L ; residual after rounding to integer
        resid = 0.0
        for k in range(3):
            kk = (xu[:, k] - xw[:, k]) / L[k]
            resid = max(resid, float(np.abs(kk - np.round(kk)).max()) * L[k])
        out["img_resid_max"].append(resid)
        img = np.zeros((natoms, 3), dtype=np.int64)
        for k in range(3):
            img[:, k] = np.round((xu[:, k] - xw[:, k]) / L[k]).astype(np.int64)
        multi = np.zeros(nchain, dtype=bool)
        span = np.zeros(nchain, dtype=bool)
        for k in range(3):
            mn = np.full(nchain, np.iinfo(np.int64).max)
            mx = np.full(nchain, np.iinfo(np.int64).min)
            np.minimum.at(mn, chain, img[:, k]); np.maximum.at(mx, chain, img[:, k])
            multi |= (mx - mn) > 0
            mn = np.full(nchain, np.inf); mx = np.full(nchain, -np.inf)
            np.minimum.at(mn, chain, xw[:, k]); np.maximum.at(mx, chain, xw[:, k])
            span |= (mx - mn) > 0.5 * L[k]
        out["multimage"].append(int(multi.sum()))
        out["span"].append(int(span.sum()))
        out["steps"].append(step)
        out["L"].append(float(L[0]))
        com_a = com_of(xu)
        rg_a = rg_of(xu, com_a)
        rg_b = rg_of(xw, com_of(xw))
        com_w = com_of(xw)
        d1 = xw - com_w[chain]
        for k in range(3):
            d1[:, k] -= L[k] * np.round(d1[:, k] / L[k])
        rg_c1 = rg_of(com_w[chain] + d1, com_w)
        d3 = xw - com_a[chain]
        for k in range(3):
            d3[:, k] -= L[k] * np.round(d3[:, k] / L[k])
        rg_c3 = rg_of(com_a[chain] + d3, com_a)
        out["rga"].append(float(rg_a.mean()))
        out["rgb"].append(float(rg_b.mean()))
        out["rgc1"].append(float(rg_c1.mean()))
        out["rgc3"].append(float(rg_c3.mean()))
    return out


if __name__ == "__main__":
    systems = ["9_pva", "9_peo", "9_pmma", "9_ps"]
    res = {}
    for sysn in systems:
        for src in ("ligpargen", "mappingff"):
            k = f"{src}/{sysn}"
            r = frame_stats(src, sysn)
            res[k] = r
            mi = np.array(r["multimage"]); sp = np.array(r["span"])
            print(f"\n=== {k}")
            print(f"  frames={len(r['steps'])}  L={min(r['L']):.3f}..{max(r['L']):.3f} A")
            print(f"  chains straddling (>1 image flag): min={mi.min()} max={mi.max()} "
                  f"mean={mi.mean():.1f} of 100   ({mi.mean():.0f}%)")
            print(f"  chains with wrapped span > L/2 : min={sp.min()} max={sp.max()} "
                  f"mean={sp.mean():.1f} of 100   ({sp.mean():.0f}%)")
            print(f"  max |(xu-x) - n*L| residual (image integrality): "
                  f"{max(r['img_resid_max']):.2e} A  (= dump 6-sig-digit round-off)")
            print(f"  mean Rg  (a)={np.mean(r['rga']):.4f}  (b)={np.mean(r['rgb']):.4f}  "
                  f"c1={np.mean(r['rgc1']):.4f}  c3={np.mean(r['rgc3']):.4f}")

    print("\n\n### ligpargen vs mappingff Rg deviation under each PBC treatment")
    print(f"{'system':>9} {'(a) xu  [CORRECT]':>24} {'(b) wrapped':>18} "
          f"{'c1 naive MIC':>18} {'c3 true-COM MIC':>18}")
    for sysn in systems:
        a = res[f"ligpargen/{sysn}"]; b = res[f"mappingff/{sysn}"]
        def dev(key):
            va, vb = np.mean(a[key]), np.mean(b[key])
            return 100 * (vb - va) / va
        print(f"{sysn:>9} {dev('rga'):+23.2f}% {dev('rgb'):+17.2f}% "
              f"{dev('rgc1'):+17.2f}% {dev('rgc3'):+17.2f}%")
    with open(OUT / "final_summary.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
