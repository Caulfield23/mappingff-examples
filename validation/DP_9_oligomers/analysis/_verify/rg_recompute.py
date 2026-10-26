"""
Independent Rg recomputation + per-frame topology consistency + Rg stationarity.

  * re-implements the topology parse and the chain-Rg computation from scratch
  * verifies, for EVERY selected frame (not just the first), that
        - atom ids are 1..N in order
        - the `mol` column of the dump matches the `mol` column of filled.data
        - the per-atom `type` column matches filled.data
  * produces the per-frame Rg time series so that stationarity / drift of Rg
    inside the analysis window can be quantified.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
STEP_LO, STEP_HI = 1_500_000, 2_000_000
MARK = b"ITEM: TIMESTEP\n"


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


def chain_rg(xyz, chain, mass, nchain):
    w = np.bincount(chain, weights=mass, minlength=nchain)
    com = np.empty((nchain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass * xyz[:, d], minlength=nchain)
    com /= w[:, None]
    diff = xyz - com[chain]
    r2 = np.einsum("ij,ij->i", diff, diff)
    return np.sqrt(np.bincount(chain, weights=mass * r2, minlength=nchain) / w)


def run(src, sysname):
    run_dir = BASE / src / sysname
    mol, typ, mass, natoms = read_topology(run_dir / "filled.data")
    uniq = np.unique(mol)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol.tolist()], dtype=np.int64)
    nchain = len(uniq)

    traj = run_dir / "dump.lammpstrj"
    steps, means, stds, mins, maxs = [], [], [], [], []
    mol_mismatch = []
    typ_mismatch = []
    id_mismatch = []
    chain_count_bad = []

    t0 = time.perf_counter()
    with open(traj, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        mm = __import__("mmap").mmap(fh.fileno(), 0, access=__import__("mmap").ACCESS_READ)
        starts = []
        pos = 0
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
            if not (STEP_LO <= step <= STEP_HI):
                continue
            lines = mm[starts[k]:starts[k + 1]].split(b"\n")
            na = int(lines[3])
            cols = [c.decode() for c in lines[8].split()[2:]]
            ia, imol, ityp = cols.index("id"), cols.index("mol"), cols.index("type")
            ix, iy, iz = cols.index("xu"), cols.index("yu"), cols.index("zu")
            arr = np.array(b" ".join(lines[9:9 + na]).split()).reshape(na, 10)
            ids = arr[:, ia].astype(np.int64)
            if not np.array_equal(ids, np.arange(1, na + 1)):
                id_mismatch.append(step)
            dm = arr[:, imol].astype(np.int64)
            dt = arr[:, ityp].astype(np.int64)
            if not np.array_equal(dm, mol):
                mol_mismatch.append(step)
            if not np.array_equal(dt, typ):
                typ_mismatch.append(step)
            # per-chain atom counts must be constant
            cnt = np.bincount(dm, minlength=nchain + 1)
            if len(set(cnt[1:].tolist())) != 1 or cnt[1:].sum() != na:
                chain_count_bad.append(step)
            xyz = arr[:, [ix, iy, iz]].astype(np.float64)
            rg = chain_rg(xyz, chain, mass, nchain)
            steps.append(step)
            means.append(rg.mean())
            stds.append(rg.std(ddof=1))
            mins.append(rg.min())
            maxs.append(rg.max())
        mm.close()

    steps = np.array(steps)
    means = np.array(means)
    stds = np.array(stds)
    n = len(steps)
    t = (steps - steps[0]).astype(np.float64) / 1e6      # ns since window start
    A = np.vstack([t, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, means, rcond=None)
    slope_ns = float(coef[0])
    # block-averaged (10-frame) SEM of the slope
    nb = n // 10
    bm = means[:nb * 10].reshape(nb, 10).mean(axis=1)
    bt = t[:nb * 10].reshape(nb, 10).mean(axis=1)
    Ab = np.vstack([bt, np.ones(nb)]).T
    cb, *_ = np.linalg.lstsq(Ab, bm, rcond=None)
    slope_block = float(cb[0])
    sem_block = float(np.std(bm, ddof=1) / np.sqrt(nb))

    return {
        "source": src, "system": sysname,
        "n_frames_selected": n,
        "steps_first": int(steps[0]), "steps_last": int(steps[-1]),
        "step_diffs": sorted(set(np.diff(steps).tolist())),
        "n_chain": int(nchain),
        "atoms_per_chain_from_dump": sorted(set(np.bincount(mol).tolist())),
        "rg_mean_over_window": float(means.mean()),
        "rg_std_mean_over_frames": float(stds.mean()),
        "rg_first_frame": float(means[0]),
        "rg_mean_excl_first_frame": float(means[1:].mean()),
        "rg_first_frame_z": float((means[0] - means[1:].mean()) / (stds[1:].mean() / np.sqrt(nchain))),
        "rg_min": float(means.min()), "rg_max": float(means.max()),
        "rg_slope_per_ns": slope_ns,
        "rg_slope_per_ns_blockavg": slope_block,
        "rg_sem_block_mean": sem_block,
        "rg_first_half_mean": float(means[:n // 2].mean()),
        "rg_second_half_mean": float(means[n // 2:].mean()),
        "rg_half_delta": float(means[n // 2:].mean() - means[:n // 2].mean()),
        "rg_last5_mean": float(means[-5:].mean()),
        "rg_first5_mean": float(means[:5].mean()),
        "id_mismatch_frames": id_mismatch,
        "mol_mismatch_frames": mol_mismatch,
        "typ_mismatch_frames": typ_mismatch,
        "chain_count_bad_frames": chain_count_bad,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main():
    out = []
    for sysname in SYSTEMS:
        for src in SOURCES:
            r = run(src, sysname)
            out.append(r)
            print(f"{src}/{sysname}: n={r['n_frames_selected']} steps {r['steps_first']}..{r['steps_last']} "
                  f"diffs={r['step_diffs']} chain={r['n_chain']} "
                  f"apc={r['atoms_per_chain_from_dump']} "
                  f"Rg={r['rg_mean_over_window']:.4f}+-{r['rg_std_mean_over_frames']:.4f} "
                  f"first_frame={r['rg_first_frame']:.4f} "
                  f"excl_first={r['rg_mean_excl_first_frame']:.4f} "
                  f"slope/ns={r['rg_slope_per_ns']:.4f} (blk {r['rg_slope_per_ns_blockavg']:.4f}) "
                  f"halfdelta={r['rg_half_delta']:.4f} "
                  f"idbad={len(r['id_mismatch_frames'])} molbad={len(r['mol_mismatch_frames'])} "
                  f"typbad={len(r['typ_mismatch_frames'])}", flush=True)
    p = Path(__file__).with_name("rg_audit.json")
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
