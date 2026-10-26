"""
Brute-force cross-check of the two suspicious numbers:
  (a) minimum INTER-chain distance  (KD-tree vs O(N^2) min-image brute force)
  (b) minimum bond (1-2) length      (trajectory vs filled.data vs minimized.data)
"""
from __future__ import annotations

import mmap
import os
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
FRAME_MARK = b"ITEM: TIMESTEP\n"


def parse(path: Path, np=np, re=re):
    lines = path.read_text(errors="replace").splitlines()
    sec = {}
    i, n = 0, len(lines)
    box = []
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        mb = re.match(r"^(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+xlo\s+xhi", line)
        if mb:
            box.append((float(mb.group(1)), float(mb.group(2))))
        m = re.match(r"^(?:(\d+)\s+)?([A-Za-z][A-Za-z ]*?)\s*(#.*)?$", line)
        if m and m.group(2).strip() in ("Masses", "Atoms", "Bonds", "Angles",
                                        "Pair Coeffs", "Bond Coeffs"):
            s = m.group(2).strip()
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            rows, k = [], j
            while k < n and lines[k].strip():
                rows.append(lines[k].split())
                k += 1
            sec[s] = rows
            i = k
            continue
        i += 1
    mot = {int(r[0]): float(r[1]) for r in sec["Masses"]}
    at = sec["Atoms"]
    aid = np.array([int(r[0]) for r in at])
    mol = np.array([int(r[1]) for r in at])
    typ = np.array([int(r[2]) for r in at])
    o = np.argsort(aid)
    aid, mol, typ = aid[o], mol[o], typ[o]
    mass = np.array([mot[t] for t in typ])
    b1 = np.array([int(r[2]) for r in sec["Bonds"]]) - 1
    b2 = np.array([int(r[3]) for r in sec["Bonds"]]) - 1
    btype = np.array([int(r[1]) for r in sec["Bonds"]])
    pc = {int(r[0]): (float(r[1]), float(r[2])) for r in sec.get("Pair Coeffs", [])}
    bc = {int(r[0]): (float(r[1]), float(r[2])) for r in sec.get("Bond Coeffs", [])}
    return dict(aid=aid, mol=mol, typ=typ, mass=mass, b1=b1, b2=b2,
                btype=btype, pair=pc, bond=bc, box=box, n=len(aid))


def bond_lengths(x, b1, b2, L, np=np):
    d = x[b1] - x[b2]
    d = d - L * np.round(d / L)
    return np.linalg.norm(d, axis=1)


def data_file_bonds(path: Path, np=np):
    """Bond lengths straight out of a LAMMPS data file (x,y,z columns)."""
    d = parse(path)
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    coords = None
    while i < len(lines):
        if lines[i].strip().startswith("Atoms"):
            j = i + 1
            while not lines[j].strip():
                j += 1
            rows = []
            k = j
            while k < len(lines) and lines[k].strip():
                rows.append(lines[k].split())
                k += 1
            # atom_style full: id mol type q x y z
            coords = np.array([[float(r[4]), float(r[5]), float(r[6])]
                               for r in rows])
            break
        i += 1
    L = np.array([b[1] - b[0] for b in d["box"]])
    return d, bond_lengths(coords, d["b1"], d["b2"], L)


def main(np=np):
    for src, sysname, step in (("mappingff", "9_ps", 2_000_000),
                               ("ligpargen", "9_peo", 2_000_000)):
        run = BASE / src / sysname
        d = parse(run / "filled.data")
        heavy = d["mass"] >= 12.0
        chain = np.array([{m: k for k, m in
                           enumerate(np.unique(d["mol"]).tolist())}[m]
                          for m in d["mol"].tolist()])
        print(f"===== {src}/{sysname} =====")
        d0, bl0 = data_file_bonds(run / "filled.data")
        dm0, blm = data_file_bonds(run / "minimized.data")
        print(f"  filled.data   bond len min/max/mean = "
              f"{bl0.min():.3f}/{bl0.max():.3f}/{bl0.mean():.3f}")
        print(f"  minimized.data bond len min/max/mean = "
              f"{blm.min():.3f}/{blm.max():.3f}/{blm.mean():.3f}")
        # bond-length histogram per element pair in filled.data
        el = np.where(np.abs(d["mass"] - 12.011) < 1e-3, "C",
                      np.where(np.abs(d["mass"] - 15.999) < 1e-3, "O", "H"))
        pairname = np.array([el[a] + el[b] for a, b in zip(d["b1"], d["b2"])])
        for p in sorted(set(pairname.tolist())):
            sel = pairname == p
            print(f"    filled.data {p}: n={sel.sum():6d} "
                  f"min={bl0[sel].min():.3f} max={bl0[sel].max():.3f} "
                  f"mean={bl0[sel].mean():.3f}")

        # trajectory frame
        traj = run / "dump.lammpstrj"
        with open(traj, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            starts, pos = [], 0
            while True:
                hit = mm.find(FRAME_MARK, pos)
                if hit < 0:
                    break
                starts.append(hit)
                pos = hit + len(FRAME_MARK)
            starts.append(size)
            for k in range(len(starts) - 1):
                beg = starts[k] + len(FRAME_MARK)
                end = mm.find(b"\n", beg)
                st = int(mm[beg:end])
                if st != step:
                    continue
                blob = mm[starts[k]:starts[k + 1]]
                lines = blob.split(b"\n")
                nat = int(lines[3])
                bx = []
                for dd in (5, 6, 7):
                    p = lines[dd].split()
                    bx.append((float(p[0]), float(p[1])))
                L = np.array([b[1] - b[0] for b in bx])
                cols = [c.decode() for c in lines[8].split()[2:]]
                arr = np.fromstring(b" ".join(lines[9:9 + nat]), sep=" ")
                arr = arr.reshape(nat, len(cols))
                ix = (cols.index("xu"), cols.index("yu"), cols.index("zu"))
                xu = np.stack([arr[:, ix[0]], arr[:, ix[1]], arr[:, ix[2]],],
                              axis=1)
                bl = bond_lengths(xu, d["b1"], d["b2"], L)
                print(f"  trajectory step {st}: bond len min/max/mean = "
                      f"{bl.min():.3f}/{bl.max():.3f}/{bl.mean():.3f}")
                for p in sorted(set(pairname.tolist())):
                    sel = pairname == p
                    print(f"    traj {p}: n={sel.sum():6d} "
                          f"min={bl[sel].min():.3f} max={bl[sel].max():.3f} "
                          f"mean={bl[sel].mean():.3f}")

                # brute force min inter-chain heavy distance
                posw = xu[heavy].copy()
                for j in range(3):
                    posw[:, j] = (posw[:, j] - bx[j][0]) % L[j]
                ch = chain[heavy]
                n = posw.shape[0]
                best = (1e9, -1, -1)
                blk = 1500
                for a0 in range(0, n, blk):
                    a1 = min(n, a0 + blk)
                    diff = posw[a0:a1, None, :] - posw[None, :, :]
                    diff -= L * np.round(diff / L)
                    dm = np.linalg.norm(diff, axis=2)
                    same = ch[a0:a1, None] == ch[None, :]
                    dm[same] = 1e9
                    j = np.unravel_index(np.argmin(dm), dm.shape)
                    if dm[j] < best[0]:
                        best = (float(dm[j]), a0 + j[0], int(j[1]))
                gi = np.nonzero(heavy)[0][best[1]]
                gj = np.nonzero(heavy)[0][best[2]]
                print(f"  BRUTE FORCE min inter-chain heavy dist = {best[0]:.4f} A"
                      f"  atoms {d['aid'][gi]}({el[gi]},type {d['typ'][gi]},"
                      f"mol {d['mol'][gi]}) & {d['aid'][gj]}({el[gj]},"
                      f"type {d['typ'][gj]},mol {d['mol'][gj]})")
                ei, ej = d["pair"].get(int(d["typ"][gi])), d["pair"].get(int(d["typ"][gj]))
                print(f"    Pair Coeffs: {ei} (eps,sigma) and {ej}")
                if ei and ej:
                    eps = (ei[0] + ej[0]) / 2
                    sig = (ei[1] + ej[1]) / 2
                    r = best[0]
                    print(f"    mixed: eps={eps:.4f} sigma={sig:.3f} -> "
                          f"LJ(r) = {4*eps*((sig/r)**12 - 2*(sig/r)**6):.1f} "
                          f"kcal/mol  (kT(300K)=0.596)")
                print(f"    xu[gi]={xu[gi]}  xu[gj]={xu[gj]}")
            mm.close()
        print()


if __name__ == "__main__":
    main()
