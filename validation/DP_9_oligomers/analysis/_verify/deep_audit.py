"""
Deep PBC investigation: 4 variants of the "wrapped-coordinate" Rg vs the
unwrapped-coordinate Rg, definitive boundary-straddling counts, and unwrapped
coordinate sanity checks.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
OUT = BASE / "analysis" / "_verify"
OUT.mkdir(parents=True, exist_ok=True)

STEP_LO, STEP_HI = 1_500_000, 2_000_000


# --------------------------------------------------------------------------- #
def read_topology_with_bonds(path: Path):
    """Return mol_of_atom, mass_of_atom, natoms, bond list (1-based atom ids)."""
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


# --------------------------------------------------------------------------- #
def rg_from_pos(pos, chain, mass, nchain, com=None):
    """Rg from absolute positions (already in a consistent gauge)."""
    w = np.bincount(chain, weights=mass, minlength=nchain)
    if com is None:
        com = np.empty((nchain, 3))
        for d in range(3):
            com[:, d] = np.bincount(chain, weights=mass * pos[:, d], minlength=nchain)
        com /= w[:, None]
    diff = pos - com[chain]
    r2 = np.einsum("ij,ij->i", diff, diff)
    num = np.bincount(chain, weights=mass * r2, minlength=nchain)
    return np.sqrt(num / w)


def com_of(pos, chain, mass, nchain):
    w = np.bincount(chain, weights=mass, minlength=nchain)
    com = np.empty((nchain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass * pos[:, d], minlength=nchain)
    com /= w[:, None]
    return com


def mic_to(com_ref, xw, L):
    """Shift xw by whole lattice vectors so each atom is nearest to com_ref."""
    d = xw - com_ref
    for k in range(3):
        d[:, k] -= L[k] * np.round(d[:, k] / L[k])
    return d


# --------------------------------------------------------------------------- #
def unwrap_by_topology(xw, chain, bonds, L, nchain):
    """
    Gold-standard molecule unwrapping: BFS over the bond graph, each atom
    placed at the minimum-image position relative to its already-placed bond
    partner.  Uses ONLY wrapped coords + bond topology + box lengths.
    """
    pos = np.empty_like(xw)
    placed = np.zeros(len(xw), dtype=bool)
    # adjacency
    adj = [[] for _ in range(len(xw))]
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)

    # seed: first atom of each chain
    for c in range(nchain):
        idx = np.where(chain == c)[0]
        if len(idx) == 0:
            continue
        s = idx[0]
        pos[s] = xw[s]
        placed[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not placed[v]:
                    d = xw[v] - xw[u]
                    for k in range(3):
                        d[k] -= L[k] * np.round(d[k] / L[k])
                    pos[v] = pos[u] + d
                    placed[v] = True
                    stack.append(v)
    if not placed.all():
        raise RuntimeError("topology unwrap left atoms unplaced")
    return pos


# --------------------------------------------------------------------------- #
def iter_frames(traj: Path):
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


def parse_frame(lines):
    step = int(lines[1])
    ibox = next(i for i, l in enumerate(lines) if l.startswith(b"ITEM: BOX BOUNDS"))
    iatom = next(i for i, l in enumerate(lines) if l.startswith(b"ITEM: ATOMS"))
    lo = np.empty(3)
    hi = np.empty(3)
    tilt = None
    for k in range(3):
        p = lines[ibox + 1 + k].split()
        lo[k] = float(p[0])
        hi[k] = float(p[1])
        if len(p) == 4:
            if tilt is None:
                tilt = np.array([float(p[2]), 0.0, 0.0])
            else:
                tilt[1] = float(p[2]) if k == 1 else tilt[1]
                tilt[2] = float(p[2]) if k == 2 else tilt[2]
    a = iatom + 1
    while a < len(lines) and lines[a].strip():
        a += 1
    atom_lines = lines[iatom + 1:a]
    cols = [c.decode() for c in lines[iatom].split()[2:]]
    arr = np.array(b" ".join(atom_lines).split(), dtype=np.float64).reshape(
        len(atom_lines), len(cols))
    return step, lo, hi, tilt, cols, arr


# --------------------------------------------------------------------------- #
def deep_run(src, sysname, every=1):
    run = BASE / src / sysname
    mol_of_atom, mass_of_atom, natoms, bonds = read_topology_with_bonds(run / "filled.data")
    uniq = np.unique(mol_of_atom)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol_of_atom.tolist()], dtype=np.int64)
    nchain = len(uniq)
    # bonds within a chain?
    cb = np.array([(chain[a], chain[b]) for a, b in bonds])
    print(f"{src}/{sysname}: natoms={natoms} nchain={nchain} nbonds={len(bonds)} "
          f"bonds_intra_chain={int((cb[:,0]==cb[:,1]).sum())} "
          f"max_bond_len_est")

    R = dict(src=src, sys=sysname, frames=0, steps=[],
             dev=dict(c1=[], c2=[], c3=[], c4=[]),
             bc=[], n_straddle=[] , n_multimage=[],
             n_span_gt_halfL=[], extent_over_halfL=[], extent_max=[],
             box=[], out_of_box=0, img_noninteger=0, n_frames=0,
             worst=None)
    t0 = time.perf_counter()
    for lines in iter_frames(run / "dump.lammpstrj"):
        step, lo, hi, tilt, cols, arr = parse_frame(lines)
        if not (STEP_LO <= step <= STEP_HI):
            continue
        L = hi - lo
        xw = arr[:, [cols.index("x"), cols.index("y"), cols.index("z")]]
        xu = arr[:, [cols.index("xu"), cols.index("yu"), cols.index("zu")]]

        # --- is the "wrapped" coordinate actually inside the box? ---
        oob = 0
        for k in range(3):
            oob += int(((xw[:, k] < lo[k] - 1e-6) | (xw[:, k] > hi[k] + 1e-6)).sum())
        R["out_of_box"] += oob

        # --- image flag integrality: (xu-x)/L must be an integer -------------
        nonint = 0
        for k in range(3):
            kk = (xu[:, k] - xw[:, k]) / L[k]
            nonint += int((np.abs(kk - np.round(kk)) > 1e-6).sum())
        R["img_noninteger"] += nonint

        # --- (a) unwrapped ---
        rg_a = rg_from_pos(xu, chain, mass_of_atom, nchain)
        com_a = com_of(xu, chain, mass_of_atom, nchain)

        # --- (c1) naive: COM from wrapped coords, one MIC pass ---------------
        com_w = com_of(xw, chain, mass_of_atom, nchain)
        pos_c1 = com_w[chain] + mic_to(com_w[chain], xw, L)
        rg_c1 = rg_from_pos(pos_c1, chain, mass_of_atom, nchain)

        # --- (c2) iterative COM-MIC ------------------------------------------
        com = com_w.copy()
        for _ in range(50):
            pos = com[chain] + mic_to(com[chain], xw, L)
            nc = com_of(pos, chain, mass_of_atom, nchain)
            dl = np.max(np.abs(nc - com))
            com = nc
            if dl < 1e-12:
                break
        rg_c2 = rg_from_pos(com[chain] + mic_to(com[chain], xw, L),
                            chain, mass_of_atom, nchain)

        # --- (c3) MIC relative to the TRUE (unwrapped) COM -------------------
        pos_c3 = com_a[chain] + mic_to(com_a[chain], xw, L)
        rg_c3 = rg_from_pos(pos_c3, chain, mass_of_atom, nchain)

        # --- (c4) topology graph unwrap from wrapped coords only -------------
        pos_c4 = unwrap_by_topology(xw, chain, bonds, L, nchain)
        rg_c4 = rg_from_pos(pos_c4, chain, mass_of_atom, nchain)

        # --- (b) wrapped, no correction -------------------------------------
        rg_b = rg_from_pos(xw, chain, mass_of_atom, nchain)

        for key, val in (("c1", rg_c1), ("c2", rg_c2), ("c3", rg_c3), ("c4", rg_c4)):
            d = np.abs(rg_a - val)
            R["dev"][key].append(float(d.max()))
        R["bc"].append(float(np.abs(rg_a - rg_b).max()))
        if R["worst"] is None or np.abs(rg_a - rg_c4).max() > R["worst"][0]:
            c = int(np.argmax(np.abs(rg_a - rg_c4)))
            R["worst"] = (float(np.abs(rg_a - rg_c4)[c]), int(step), int(c),
                          float(rg_a[c]), float(rg_c4[c]))

        # --- straddling metrics ---------------------------------------------
        # (i) definitive: distinct LAMMPS image flags occupied per chain
        img = np.zeros((natoms, 3), dtype=np.int64)
        for k in range(3):
            img[:, k] = np.round((xu[:, k] - xw[:, k]) / L[k]).astype(np.int64)
        multi = np.zeros(nchain, dtype=bool)
        for k in range(3):
            for c in range(nchain):
                pass
        # vectorised: min/max image per chain
        for k in range(3):
            mn = np.full(nchain, np.iinfo(np.int64).max)
            mx = np.full(nchain, np.iinfo(np.int64).min)
            np.minimum.at(mn, chain, img[:, k])
            np.maximum.at(mx, chain, img[:, k])
            multi |= (mx - mn) > 0
        R["n_multimage"].append(int(multi.sum()))

        # (ii) wrapped span > L/2
        span = np.zeros(nchain, dtype=bool)
        for k in range(3):
            mn = np.full(nchain, np.inf)
            mx = np.full(nchain, -np.inf)
            np.minimum.at(mn, chain, xw[:, k])
            np.maximum.at(mx, chain, xw[:, k])
            span |= (mx - mn) > 0.5 * L[k]
        R["n_span_gt_halfL"].append(int(span.sum()))

        # (iii) true extent of each chain (max pairwise distance, xu gauge)
        ext = np.zeros(nchain)
        for c in range(nchain):
            p = xu[chain == c]
            d = p.max(axis=0) - p.min(axis=0)
            ext[c] = np.linalg.norm(d)
        R["extent_max"].append(float(ext.max()))
        R["extent_over_halfL"].append(int((ext > 0.5 * L[0]).sum()))

        R["steps"].append(int(step))
        R["box"].append((int(step), float(L[0])))
        R["n_frames"] += 1
    R["frames"] = R["n_frames"]
    R["wall_s"] = time.perf_counter() - t0
    for k in R["dev"]:
        R["dev"][k] = np.array(R["dev"][k])
    for k in ("n_multimage", "n_span_gt_halfL", "extent_over_halfL"):
        R[k] = np.array(R[k])
    R["extent_max"] = np.array(R["extent_max"])
    R["box"] = R["box"][::20] + R["box"][-2:]
    R.pop("n_frames")
    return R


def report(R):
    print(f"\n{'='*78}\n{R['src']}/{R['sys']}  frames={R['frames']}  "
          f"wall={R['wall_s']:.1f}s")
    print(f"  box Lx: first {R['box'][0]} last {R['box'][-1]}   "
          f"min {min(b[1] for b in R['box']):.3f} max {max(b[1] for b in R['box']):.3f}")
    print(f"  atoms with wrapped coord outside box (total over frames): {R['out_of_box']}")
    print(f"  atoms with non-integer image offset (xu-x)/L : {R['img_noninteger']}")
    print("  --- max |Rg(a)-Rg(c_k)| over all chains & frames ---")
    for k in ("c1", "c2", "c3", "c4"):
        d = R["dev"][k]
        print(f"    {k}: max={d.max():.6g}  mean-of-frame-max={d.mean():.6g}  "
              f"n_frames_with_dev>1e-8={int((d>1e-8).sum())}/{len(d)}")
    print(f"    b (wrapped, no MIC): max={max(R['bc']):.6g}")
    print("  --- straddling ---")
    print(f"    chains occupying >1 image flag (definitive): "
          f"min={R['n_multimage'].min()} max={R['n_multimage'].max()} "
          f"mean={R['n_multimage'].mean():.1f} / {100}")
    print(f"    chains with wrapped span > L/2: min={R['n_span_gt_halfL'].min()} "
          f"max={R['n_span_gt_halfL'].max()} mean={R['n_span_gt_halfL'].mean():.1f}")
    print(f"    chains whose TRUE extent > L/2: min={R['extent_over_halfL'].min()} "
          f"max={R['extent_over_halfL'].max()} mean={R['extent_over_halfL'].mean():.1f}")
    print(f"    max chain extent (A): min {R['extent_max'].min():.2f} "
          f"max {R['extent_max'].max():.2f}")


if __name__ == "__main__":
    runs = [("ligpargen", "9_pva"), ("mappingff", "9_pva"),
            ("ligpargen", "9_peo"), ("mappingff", "9_peo"),
            ("ligpargen", "9_ps"), ("mappingff", "9_ps")]
    out = []
    for src, s in runs:
        R = deep_run(src, s)
        report(R)
        out.append(R)
    with open(OUT / "deep_audit.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
