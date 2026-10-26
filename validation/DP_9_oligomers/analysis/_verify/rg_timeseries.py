"""Verifier part 10: full-trajectory Rg time series to test equilibration."""
import os, re, mmap
from pathlib import Path
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
SOURCES = ("ligpargen", "mappingff")
FRAME_MARK = b"ITEM: TIMESTEP\n"
ELEM = {15.999: "O", 12.011: "C", 1.008: "H"}


def read_topology(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    mt, atoms = {}, []
    i = 0
    while not lines[i].strip().startswith("Masses"):
        i += 1
    i += 1
    while not lines[i].strip():
        i += 1
    while lines[i].strip():
        p = lines[i].split(); mt[int(p[0])] = float(p[1]); i += 1
    while not lines[i].strip().startswith("Atoms"):
        i += 1
    i += 1
    while not lines[i].strip():
        i += 1
    while lines[i].strip():
        p = lines[i].split(); atoms.append((int(p[0]), int(p[1]), int(p[2]))); i += 1
    nat = len(atoms)
    mol = np.zeros(nat, np.int64); typ = np.zeros(nat, np.int64)
    for aid, m, t in atoms:
        mol[aid-1] = m; typ[aid-1] = t
    mass = np.array([mt[int(t)] for t in typ])
    return mol, mass, nat


def chain_rg(xyz, chain, mass, n_chain):
    w = np.bincount(chain, weights=mass, minlength=n_chain)
    com = np.empty((n_chain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass*xyz[:, d], minlength=n_chain)
    com /= w[:, None]
    diff = xyz - com[chain]
    r2 = np.einsum("ij,ij->i", diff, diff)
    num = np.bincount(chain, weights=mass*r2, minlength=n_chain)
    return np.sqrt(num/w)


for sysname in SYSTEMS:
    print("=" * 110)
    print(f"SYSTEM {sysname}   Rg(t) time series, mean over 100 chains")
    print("=" * 110)
    for src in SOURCES:
        run = BASE / src / sysname
        mol, mass, nat = read_topology(run / "filled.data")
        uniq = np.unique(mol); remap = {m: k for k, m in enumerate(uniq.tolist())}
        chain = np.array([remap[m] for m in mol.tolist()], np.int64)
        n_chain = len(uniq)
        traj = run / "dump.lammpstrj"
        steps, means = [], []
        with open(traj, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            starts = []
            pos = 0
            while True:
                hit = mm.find(FRAME_MARK, pos)
                if hit < 0: break
                starts.append(hit); pos = hit + len(FRAME_MARK)
            starts.append(size)
            for k in range(len(starts)-1):
                beg = starts[k] + len(FRAME_MARK)
                end = mm.find(b"\n", beg)
                step = int(mm[beg:end])
                lines = mm[starts[k]:starts[k+1]].split(b"\n")
                cols = [c.decode() for c in lines[8].split()[2:]]
                ix, iy, iz = cols.index("xu"), cols.index("yu"), cols.index("zu")
                raw = np.array([ln.split() for ln in lines[9:9+nat]])
                xyz = raw[:, [ix, iy, iz]].astype(np.float64)
                rg = chain_rg(xyz, chain, mass, n_chain)
                steps.append(step); means.append(rg.mean())
            mm.close()
        s = np.array(steps); m = np.array(means)
        # bins of 50 frames
        nb = len(m)//20
        print(f"  [{src}] nframes={len(m)}  first_step={s[0]}  last_step={s[-1]}")
        line = []
        for i in range(0, nb*20, 20):
            line.append(f"{s[i]:>8d}:{m[i:i+20].mean():.3f}")
        for i in range(0, len(line), 10):
            print("      " + "  ".join(line[i:i+10]))
        # last 101-frame window
        w = m[(s >= 1_500_000) & (s <= 2_000_000)]
        sw = s[(s >= 1_500_000) & (s <= 2_000_000)]
        print(f"      NVT window: n={len(w)} mean={w.mean():.4f} std_of_frames={w.std(ddof=1):.4f} "
              f"first5={w[:5].mean():.4f} last5={w[-5:].mean():.4f} "
              f"drift(last5-first5)={w[-5:].mean()-w[:5].mean():+.4f}")
    print()
