"""
(1) Shape of the density relaxation over the whole run (block-averaged, 25 ps
    blocks) so we can see whether the eq_lo NPT stage is converged at 1.5e6.
(2) Cross-check: the BOX BOUNDS in the dump vs the Lx in log.lammps at the
    same step (trajectory/log consistency).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
MARK = b"ITEM: TIMESTEP\n"


def log_rows(log: Path):
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
                out.append((step, vals[idx["Density"]], vals[idx["Lx"]]))
                j += 1
            i = j
            continue
        i += 1
    return out


for sysname in SYSTEMS:
    for src in SOURCES:
        run = BASE / src / sysname
        rows = log_rows(run / "log.lammps")
        # de-duplicate stage-boundary rows (same step printed twice, identical values)
        seen = {}
        for s, rho, Lx in rows:
            seen[s] = (rho, Lx)
        steps = np.array(sorted(seen))
        rho = np.array([seen[s][0] for s in steps])
        Lx = np.array([seen[s][1] for s in steps])

        # ---- dump box bounds vs log Lx ---------------------------------
        traj = run / "dump.lammpstrj"
        mism = []
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
                if not (1_500_000 <= step <= 2_000_000):
                    continue
                lines = mm[starts[k]:starts[k + 1]].split(b"\n")
                bx = [float(v) for v in lines[5].split()]
                if bx[1] - bx[0] != Lx[list(steps).index(step)]:
                    mism.append((step, bx[1] - bx[0], Lx[list(steps).index(step)]))
            mm.close()

        # ---- block-averaged density trajectory (25 ps blocks) -----------
        B = 25
        nb = len(rho) // B
        bm = rho[:nb * B].reshape(nb, B).mean(axis=1)
        bt = steps[:nb * B].reshape(nb, B).mean(axis=1)
        # print a compact trajectory
        traj_str = []
        for i in range(0, nb, 4):        # every 100 ps
            traj_str.append(f"{bt[i]/1e6:.3f}M:{bm[i]:.4f}")
        print(f"=== {src}/{sysname}   dump-vs-log Lx mismatches in window: {len(mism)}")
        print("    rho(25ps blocks, every 100 ps): " + "  ".join(traj_str))
        if mism:
            print("    MISMATCHES:", mism[:5])
    print()
