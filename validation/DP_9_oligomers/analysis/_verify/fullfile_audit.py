"""Deep audit of EVERY frame (all 400) in all 8 dumps: header, natoms, atom-line
count, token count, id ordering, mol range, non-finite floats, box bounds."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
MARK = b"ITEM: TIMESTEP\n"
EXPECT = b"ITEM: ATOMS id mol type q x y z xu yu zu"

for sysname in SYSTEMS:
    for src in SOURCES:
        traj = BASE / src / sysname / "dump.lammpstrj"
        probs = []
        nframes = 0
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
            nframes = len(starts) - 1
            for k in range(nframes):
                beg = starts[k] + len(MARK)
                end = mm.find(b"\n", beg)
                step = int(mm[beg:end])
                lines = mm[starts[k]:starts[k + 1]].split(b"\n")
                if lines[0] != b"ITEM: TIMESTEP":
                    probs.append((step, "TIMESTEP marker"))
                if lines[2] != b"ITEM: NUMBER OF ATOMS":
                    probs.append((step, "NUMBER OF ATOMS marker"))
                if not lines[4].startswith(b"ITEM: BOX BOUNDS"):
                    probs.append((step, "BOX BOUNDS marker"))
                if lines[8] != EXPECT:
                    probs.append((step, "ATOMS header %r" % lines[8][:80]))
                na = int(lines[3])
                for bi in (5, 6, 7):
                    try:
                        v = np.array([float(t) for t in lines[bi].split()])
                    except Exception:
                        probs.append((step, "box line %d unparsable" % bi))
                        continue
                    if not np.all(np.isfinite(v)):
                        probs.append((step, "box line %d non-finite" % bi))
                    if v[1] - v[0] <= 0:
                        probs.append((step, "box line %d non-positive extent" % bi))
                al = lines[9:9 + na]
                if len(al) != na:
                    probs.append((step, "atom lines %d != %d" % (len(al), na)))
                    continue
                toks = np.array(b" ".join(al).split())
                if toks.size != na * 10:
                    probs.append((step, "token count %d != %d" % (toks.size, na * 10)))
                    continue
                arr = toks.reshape(na, 10)
                ids = arr[:, 0].astype(np.int64)
                if not np.array_equal(ids, np.arange(1, na + 1)):
                    probs.append((step, "atom ids not 1..N"))
                mol = arr[:, 1].astype(np.int64)
                if mol.min() != 1 or mol.max() != 100:
                    probs.append((step, "mol range %d..%d" % (mol.min(), mol.max())))
                num = arr[:, 3:].astype(np.float64)
                if not np.all(np.isfinite(num)):
                    probs.append((step, "non-finite numeric value"))
                # x/y/z must lie inside the box bounds; xu/yu/zu must be finite
                bx = [(float(t) for t in lines[5].split())]
                if bx:
                    lo, hi = bx[0]
                    xyz = arr[:, 4:7].astype(np.float64)
                    if xyz.min() < lo - 1e-3 or xyz.max() > hi + 1e-3:
                        probs.append((step, "wrapped coord outside box %g..%g" % (xyz.min(), xyz.max())))
            mm.close()
        print(f"{src}/{sysname}: frames={nframes}  problems={len(probs)}"
              + (f"  {probs[:5]}" if probs else ""))
