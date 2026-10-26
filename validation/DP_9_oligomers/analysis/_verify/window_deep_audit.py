"""
(1) Full-file audit split into 'inside window' vs 'outside window'.
(2) Decisive check that xu/yu/zu are genuinely UNWRAPPED:
        for every atom, (xu - x) must be an exact integer multiple of the
        corresponding box length (up to the print precision).
    Also compares the chain Rg computed from xu (unwrapped) vs from x (wrapped)
    to show the difference is real -- i.e. that using x would be wrong.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
MARK = b"ITEM: TIMESTEP\n"
EXPECT = b"ITEM: ATOMS id mol type q x y z xu yu zu"
LO, HI = 1_500_000, 2_000_000


def read_topology(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    natoms = None
    mass_of_type = {}
    mol = None
    typ = None
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
            rows = []
            while j < n and lines[j].strip():
                p = lines[j].split()
                rows.append((int(p[0]), int(p[1]), int(p[2])))
                j += 1
            mol = np.zeros(natoms, dtype=np.int64)
            typ = np.zeros(natoms, dtype=np.int64)
            for aid, m_, t_ in rows:
                mol[aid - 1] = m_
                typ[aid - 1] = t_
            i = j
            continue
        i += 1
    mass = np.array([mass_of_type[t] for t in typ], dtype=np.float64)
    return mol, typ, mass, natoms


import re  # noqa: E402


def chain_rg(xyz, chain, mass, nchain):
    w = np.bincount(chain, weights=mass, minlength=nchain)
    com = np.empty((nchain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass * xyz[:, d], minlength=nchain)
    com /= w[:, None]
    diff = xyz - com[chain]
    return np.sqrt(np.bincount(chain, weights=mass * np.einsum("ij,ij->i", diff, diff),
                               minlength=nchain) / w)


for sysname in SYSTEMS:
    for src in SOURCES:
        rd = BASE / src / sysname
        mol, typ, mass, natoms = read_topology(rd / "filled.data")
        uniq = np.unique(mol)
        chain = np.array([{m: k for k, m in enumerate(uniq.tolist())}[m]
                          for m in mol.tolist()], dtype=np.int64)
        traj = rd / "dump.lammpstrj"
        in_win, out_win = [], []
        unwrap_bad = []
        unwrap_bad_w = []
        rg_u, rg_w = [], []
        with open(traj, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            mm = __import__("mmap").mmap(fh.fileno(), 0, access=__import__("mmap").ACCESS_READ)
            starts, pos = [], 0
            while True:
                hit = mm.find(MARK, pos)
                if hit < 0:
                    break
                starts.append(hit)
                pos = hit + len(MARK)
            starts.append(size)
            for k in range(len(starts) - 1):
                beg = starts[k] + len(MARK)
                end = mm.find(b"\n", beg)
                step = int(mm[beg:end])
                lines = mm[starts[k]:starts[k + 1]].split(b"\n")
                na = int(lines[3])
                probs = []
                if lines[8] != EXPECT:
                    probs.append((step, "header"))
                for bi in (5, 6, 7):
                    v = [float(t) for t in lines[bi].split()]
                    if len(v) != 2 or not np.all(np.isfinite(v)):
                        probs.append((step, "box"))
                arr = np.array(b" ".join(lines[9:9 + na]).split()).reshape(na, 10)
                ids = arr[:, 0].astype(np.int64)
                if not np.array_equal(ids, np.arange(1, na + 1)):
                    probs.append((step, "ids"))
                m_ = arr[:, 1].astype(np.int64)
                if m_.min() != 1 or m_.max() != 100:
                    probs.append((step, "mol"))
                num = arr[:, 3:].astype(np.float64)
                if not np.all(np.isfinite(num)):
                    probs.append((step, "nonfinite"))

                # unwrap consistency: xu - x  ==  k * L
                L = np.array([float(lines[5].split()[1]) - float(lines[5].split()[0]),
                              float(lines[6].split()[1]) - float(lines[6].split()[0]),
                              float(lines[7].split()[1]) - float(lines[7].split()[0])])
                xw = arr[:, 4:7].astype(np.float64)
                xu = arr[:, 7:10].astype(np.float64)
                kk = (xu - xw) / L
                resid = np.abs(kk - np.round(kk)).max()
                if resid > 1e-4:
                    (ununwrap_bad_w if LO <= step <= HI else unwrap_bad).append((step, resid))
                if LO <= step <= HI:
                    rg_u.append(chain_rg(xu, chain, mass, len(uniq)).mean())
                    rg_w.append(chain_rg(xw, chain, mass, len(uniq)).mean())
                    in_win.extend(probs)
                else:
                    out_win.extend(probs)
            mm.close()
        print(f"{src}/{sysname}: problems IN window={len(in_win)}  outside window={len(out_win)}"
              f"  unwrap-resid>1e-4: in window={len(unwrap_bad_w)} outside={len(unwrap_bad)}")
        print(f"    mean Rg from xu={np.mean(rg_u):.4f} A   from wrapped x={np.mean(rg_w):.4f} A"
              f"   (ratio {np.mean(rg_w)/np.mean(rg_u):.3f})")
        if in_win:
            print("    IN-WINDOW PROBLEMS:", in_win[:5])
