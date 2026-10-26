"""
Sanity checks on the unwrapped (xu) coordinates:
  * bond-length stability for fixed bonded pairs over the WHOLE run
  * box shrinkage over the run
  * whether xu is rescaled with the box (would corrupt intramolecular geometry)
  * out-of-box excursions of the wrapped coordinate
  * (a) vs (c3) residual magnitudes (precision vs real error)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
OUT = BASE / "analysis" / "_verify"


def read_topo(path):
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    natoms = None
    mass_of_type = {}
    mol_of_atom = typ_of_atom = None
    bonds = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        m = re.match(r"^(\d+)\s+atoms\b", line)
        if m and natoms is None:
            natoms = int(m.group(1))
        if line.startswith("Masses"):
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            while j < n and lines[j].strip():
                p = lines[j].split()
                mass_of_type[int(p[0])] = float(p[1])
                j += 1
            i = j
            continue
        if line.startswith("Atoms"):
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            mol = np.zeros(natoms, dtype=np.int64)
            typ = np.zeros(natoms, dtype=np.int64)
            while j < n and lines[j].strip():
                p = lines[j].split()
                mol[int(p[0]) - 1] = int(p[1])
                typ[int(p[0]) - 1] = int(p[2])
                j += 1
            mol_of_atom, typ_of_atom = mol, typ
            i = j
            continue
        if line.startswith("Bonds"):
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            while j < n and lines[j].strip():
                p = lines[j].split()
                bonds.append((int(p[2]) - 1, int(p[3]) - 1))
                j += 1
            i = j
            continue
        i += 1
    mass_of_atom = np.array([mass_of_type[t] for t in typ_of_atom], dtype=np.float64)
    return mol_of_atom, mass_of_atom, natoms, bonds


def frames(traj):
    import mmap
    import os
    with open(traj, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        starts = []
        pos = 0
        while True:
            hit = mm.find(b"ITEM: TIMESTEP\n", pos)
            if hit < 0:
                break
            starts.append(hit)
            pos = hit + len(b"ITEM: TIMESTEP\n")
        starts.append(size)
        for k in range(len(starts) - 1):
            lines = mm[starts[k]:starts[k + 1]].split(b"\n")
            yield lines
        mm.close()


def parse(lines):
    step = int(lines[1])
    ibox = next(i for i, l in enumerate(lines) if l.startswith(b"ITEM: BOX BOUNDS"))
    iatom = next(i for i, l in enumerate(lines) if l.startswith(b"ITEM: ATOMS"))
    lo = np.array([float(lines[ibox + 1 + k].split()[0]) for k in range(3)])
    hi = np.array([float(lines[ibox + 1 + k].split()[1]) for k in range(3)])
    a = iatom + 1
    while a < len(lines) and lines[a].strip():
        a += 1
    cols = [c.decode() for c in lines[iatom].split()[2:]]
    arr = np.array(b" ".join(lines[iatom + 1:a]).split(), dtype=np.float64).reshape(
        a - iatom - 1, len(cols))
    return step, lo, hi, cols, arr


def run(src, sysname):
    d = BASE / src / sysname
    mol, mass, natoms, bonds = read_topo(d / "filled.data")
    uniq = np.unique(mol)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol.tolist()], dtype=np.int64)
    nchain = len(uniq)
    w = np.bincount(chain, weights=mass, minlength=nchain)

    def com_of(pos):
        c = np.empty((nchain, 3))
        for k in range(3):
            c[:, k] = np.bincount(chain, weights=mass * pos[:, k], minlength=nchain)
        return c / w[:, None]

    def rg_of(pos, com):
        diff = pos - com[chain]
        r2 = np.einsum("ij,ij->i", diff, diff)
        return np.sqrt(np.bincount(chain, weights=mass * r2, minlength=nchain) / w)

    out = {"src": src, "sys": sysname}
    steps, Ls, bonds_u, bonds_w, oob, devs_c3, res_c3 = [], [], [], [], [], [], []
    # pick 8 bonds from 8 different chains (long chain bonds)
    first = True
    for lines in frames(d / "dump.lammpstrj"):
        step, lo, hi, cols, arr = parse(lines)
        L = hi - lo
        xw = arr[:, [cols.index("x"), cols.index("y"), cols.index("z")]]
        xu = arr[:, [cols.index("xu"), cols.index("yu"), cols.index("zu")]]
        if first:
            # choose bonds: from the bond list, take one per chain for chains 0..9
            sel = []
            seen = set()
            for a, b in bonds:
                c = chain[a]
                if c not in seen and c < 12:
                    seen.add(c)
                    sel.append((a, b))
            out["bond_pairs"] = sel
            out["bond_chains"] = [int(chain[a]) for a, b in sel]
            first = False
        bl_u = [float(np.linalg.norm(xu[a] - xu[b])) for a, b in out["bond_pairs"]]
        bl_w = [float(np.linalg.norm(xw[a] - xw[b])) for a, b in out["bond_pairs"]]
        bonds_u.append(bl_u)
        bonds_w.append(bl_w)
        steps.append(step)
        Ls.append(float(L[0]))
        # out-of-box
        for k in range(3):
            oob.append(float(np.maximum(lo[k] - xw[:, k], xw[:, k] - hi[k]).max()))
        # (a) vs (c3)
        com_a = com_of(xu)
        rg_a = rg_of(xu, com_a)
        d3 = xw - com_a[chain]
        for k in range(3):
            d3[:, k] -= L[k] * np.round(d3[:, k] / L[k])
        pos3 = com_a[chain] + d3
        rg_c3 = rg_of(pos3, com_a)
        devs_c3.append(float(np.abs(rg_a - rg_c3).max()))
        res_c3.append(float(np.abs(pos3 - xu).max()))

    bonds_u = np.array(bonds_u)
    bonds_w = np.array(bonds_w)
    out["steps"] = steps
    out["L"] = Ls
    out["bond_xu_min"] = bonds_u.min(axis=0).tolist()
    out["bond_xu_max"] = bonds_u.max(axis=0).tolist()
    out["bond_xu_first"] = bonds_u[0].tolist()
    out["bond_xu_relfluct"] = ((bonds_u.max(axis=0) - bonds_u.min(axis=0))
                               / bonds_u[0]).tolist()
    out["bond_wrapped_max"] = float(bonds_w.max())
    out["oob_max"] = float(np.max(oob))
    out["oob_frames"] = int((np.array(oob) > 1e-9).sum())
    out["dev_c3_max"] = float(np.max(devs_c3))
    out["dev_c3_window_max"] = float(np.max([d for s, d in zip(steps, devs_c3)
                                              if 1_500_000 <= s <= 2_000_000]))
    out["res_c3_max"] = float(np.max(res_c3))
    out["res_c3_window_max"] = float(np.max([r for s, r in zip(steps, res_c3)
                                              if 1_500_000 <= s <= 2_000_000]))
    # bond behaviour split by stage
    st = np.array(steps)
    for name, m in (("deform", st <= 401000), ("eq_hi", (st > 401000) & (st <= 501000)),
                    ("cool", (st > 501000) & (st <= 1001000)),
                    ("eq_lo", (st > 1001001) & (st <= 1501000)),
                    ("nvt", st > 1501000)):
        if m.sum():
            bb = bonds_u[m]
            out[f"bond_fluct_{name}"] = float(
                (bb.max(axis=0) - bb.min(axis=0)).max())
            out[f"L_{name}"] = [float(np.min(np.array(Ls)[m])), float(np.max(np.array(Ls)[m]))]
    return out


if __name__ == "__main__":
    runs = [("ligpargen", "9_pva"), ("mappingff", "9_pva"),
            ("ligpargen", "9_peo"), ("mappingff", "9_ps")]
    res = []
    for s, y in runs:
        o = run(s, y)
        res.append(o)
        print(f"\n=== {s}/{y}")
        print("  L over run:", "first", o["L"][0], "min", min(o["L"]),
              "max", max(o["L"]), "last", o["L"][-1])
        print("  bond(xu) first:", [round(v, 4) for v in o["bond_xu_first"]])
        print("  bond(xu) min  :", [round(v, 4) for v in o["bond_xu_min"]])
        print("  bond(xu) max  :", [round(v, 4) for v in o["bond_xu_max"]])
        print("  bond(xu) max rel fluct: %.3e" % max(o["bond_xu_relfluct"]))
        for st in ("deform", "eq_hi", "cool", "eq_lo", "nvt"):
            k = f"bond_fluct_{st}"
            if k in o:
                print(f"    stage {st:7s}: max bond fluct {o[k]:.4e} A   L in {o['L_'+st]}")
        print("  wrapped-bond max length:", round(o["bond_wrapped_max"], 3),
              " (a bond should be ~1-2 A)")
        print("  out-of-box excursion max:", o["oob_max"], "in", o["oob_frames"], "frames")
        print("  (a) vs (c3) max dev: all-run %.3e   window %.3e" %
              (o["dev_c3_max"], o["dev_c3_window_max"]))
        print("  |pos_c3 - xu| max: all-run %.3e  window %.3e" %
              (o["res_c3_max"], o["res_c3_window_max"]))
    with open(OUT / "sanity_deep.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
