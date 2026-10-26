"""
Independent re-computation + physics sanity checks for the Rg values produced
by analyze_density_rg.py.

Re-implements the LAMMPS data-file parser, dump parser and Rg from scratch so
that a shared bug cannot hide.

NOTE on style: every function binds the modules it needs as DEFAULT
ARGUMENTS.  The conda CPython 3.10.21 in this environment intermittently
resolves a module-level global name to "not defined" inside a function body
(spurious NameError for a name that is demonstrably present in globals()).
Binding via defaults turns the lookups into LOAD_FAST and side-steps it.

For every run it computes, on the SAME window [1.5e6, 2.0e6]:
    R1  mass-weighted, all atoms          (must reproduce the published number)
    R2  geometric (unweighted), all atoms
    R3  mass-weighted, HEAVY atoms only   (mass >= 12)
    R4  geometric, HEAVY atoms only
    R5  mass-weighted, backbone skeleton only
plus end-to-end distance, min INTER-chain distance (PBC) and the contrast
between within-frame across-chain std and temporal std.
"""
from __future__ import annotations

import json
import mmap
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
RUNS = [(src, s) for s in ("9_peo", "9_pmma", "9_ps", "9_pva")
        for src in ("ligpargen", "mappingff")]
STEP_LO, STEP_HI = 1_500_000, 2_000_000
FRAME_MARK = b"ITEM: TIMESTEP\n"
PROBE_STEPS = (1_500_000, 1_600_000, 1_700_000, 1_800_000, 1_900_000, 2_000_000)


def parse_data(path: Path, np=np, re=re):
    lines = path.read_text(errors="replace").splitlines()
    sec, box = {}, []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        mb = re.match(r"^(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+xlo\s+xhi", line)
        if mb:
            box.append((float(mb.group(1)), float(mb.group(2))))
        m = re.match(r"^(?:(\d+)\s+)?([A-Za-z][A-Za-z ]*?)\s*(#.*)?$", line)
        if m and m.group(2).strip() in ("Masses", "Atoms", "Bonds"):
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
    return aid, mol, typ, mass, b1, b2, box


def chain_rg(xyz, chain, w, n_chain, np=np):
    """Generic weighted Rg.  w = per-atom weight (mass, or 1 for geometric)."""
    wsum = np.bincount(chain, weights=w, minlength=n_chain)
    com = np.empty((n_chain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=w * xyz[:, d], minlength=n_chain)
    com /= wsum[:, None]
    diff = xyz - com[chain]
    r2 = np.einsum("ij,ij->i", diff, diff)
    return np.sqrt(np.bincount(chain, weights=w * r2, minlength=n_chain) / wsum)


def read_frames(path: Path, os=os, mmap=mmap):
    """Yield (step, blob) for every frame of a LAMMPS dump."""
    with open(path, "rb") as fh:
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
            step = int(mm[beg:end])
            yield step, mm[starts[k]:starts[k + 1]]
        mm.close()


def parse_frame(blob, natoms, np=np):
    lines = blob.split(b"\n")
    nat = int(lines[3])
    assert nat == natoms, (nat, natoms)
    bx = []
    for d in (5, 6, 7):
        p = lines[d].split()
        bx.append((float(p[0]), float(p[1])))
    cols = [c.decode() for c in lines[8].split()[2:]]
    flat = np.fromstring(b" ".join(lines[9:9 + nat]), sep=" ")
    arr = flat.reshape(nat, len(cols))
    return arr, cols, bx


def get_xyz(arr, cols, names, np=np):
    ia = (cols.index(names[0]), cols.index(names[1]), cols.index(names[2]))
    return np.stack([arr[:, ia[0]], arr[:, ia[1]], arr[:, ia[2]]], axis=1)


def longest_heavy_path(idx, heavy, adj, np=np):
    """Longest simple path (in bonds) in the heavy subgraph of one chain."""
    sub = {i: [j for j in adj[i] if heavy[j]] for i in idx if heavy[i]}
    best = [0, None]

    def dfs(u, seen, length):
        if length > best[0]:
            best[0] = length
            best[1] = list(seen)
        for v in sub.get(u, ()):
            if v not in seen:
                seen.add(v)
                dfs(v, seen, length + 1)
                seen.discard(v)

    starts = [i for i in sub if len(sub[i]) <= 2] or list(sub)
    for s in starts:
        dfs(s, {s}, 0)
    return best[0], best[1]


def min_inter_chain(pos, ch, L, np=np, cKDTree=cKDTree):
    tree = cKDTree(pos, boxsize=np.array(L))
    pairs = tree.query_pairs(r=6.0, output_type="ndarray")
    if not len(pairs):
        return float("nan"), float("nan")
    dm = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    same = ch[pairs[:, 0]] == ch[pairs[:, 1]]
    return (float(dm[~same].min()) if (~same).any() else float("nan"),
            float(dm[same].min()) if same.any() else float("nan"))


def main(np=np, os=os, time=time, json=json, Path=Path):
    out = {}
    for src, sysname in RUNS:
        run = BASE / src / sysname
        t0 = time.perf_counter()
        aid, mol, typ, mass, b1, b2, box = parse_data(run / "filled.data")
        natoms = len(aid)
        heavy = mass >= 12.0

        uniq = np.unique(mol)
        remap = {m: k for k, m in enumerate(uniq.tolist())}
        chain = np.array([remap[m] for m in mol.tolist()], dtype=np.int64)
        n_chain = len(uniq)

        adj = [[] for _ in range(natoms)]
        for a, b in zip(b1, b2):
            adj[a].append(b)
            adj[b].append(a)

        idx1 = np.where(chain == 0)[0]
        nb_bonds, path1 = longest_heavy_path(idx1, heavy, adj)
        path1_set = set(path1)

        traj = run / "dump.lammpstrj"
        R1l, R2l, R3l, R4l, R5l = [], [], [], [], []
        e2el, dminl, outbox = [], [], []
        steps, all_steps = [], []
        n_bad_mol = n_bad_id = 0
        mol_ok = True

        # backbone mask for all chains (chains are contiguous blocks)
        chain0 = np.where(chain == 0)[0]
        contiguous = (chain0[0] == 0 and
                      chain0[-1] - chain0[0] + 1 == len(chain0))
        bb_mask = np.zeros(natoms, dtype=bool)
        if contiguous:
            nper = len(chain0)
            for c in range(n_chain):
                for o in path1:
                    bb_mask[c * nper + o] = True

        for step, blob in read_frames(traj):
            all_steps.append(step)
            if not (STEP_LO <= step <= STEP_HI):
                continue
            arr, cols, bx = parse_frame(blob, natoms)
            ids = arr[:, cols.index("id")].astype(np.int64)
            mols = arr[:, cols.index("mol")].astype(np.int64)
            if not np.array_equal(ids, np.arange(1, natoms + 1)):
                n_bad_id += 1
            if not np.array_equal(mols, mol):
                n_bad_mol += 1
                mol_ok = False
            xu = get_xyz(arr, cols, ("xu","yu","zu"))
            xw = get_xyz(arr, cols, ("x","y","z"))
            L = [b[1] - b[0] for b in bx]

            fo = 0
            for d in range(3):
                lo, hi = bx[d]
                fo += int(np.count_nonzero((xu[:, d] < lo - 1e-6) |
                                           (xu[:, d] > hi + 1e-6)))
            outbox.append(fo / natoms)

            w_one = np.ones(natoms)
            R1l.append(chain_rg(xu, chain, mass, n_chain))
            R2l.append(chain_rg(xu, chain, w_one, n_chain))
            R3l.append(chain_rg(xu[heavy], chain[heavy], mass[heavy], n_chain))
            R4l.append(chain_rg(xu[heavy], chain[heavy], w_one[heavy], n_chain))
            if contiguous:
                R5l.append(chain_rg(xu[bb_mask], chain[bb_mask],
                                   mass[bb_mask], n_chain))

            # end-to-end distance of the chain-1 backbone
            if path1:
                p = xu[np.array(sorted(path1_set))]
                e2el.append(float(np.linalg.norm(p[-1] - p[0])))

            if step in PROBE_STEPS:
                pos = xu[heavy].copy()
                for d in range(3):
                    pos[:, d] = (pos[:, d] - bx[d][0]) % L[d]
                dminl.append(min_inter_chain(pos, chain[heavy], L))
            steps.append(step)

        R1 = np.array(R1l); R2 = np.array(R2l)
        R3 = np.array(R3l); R4 = np.array(R4l)
        R5 = np.array(R5l) if R5l else None
        fmean = np.array([r.mean() for r in R1])
        fstd = np.array([r.std(ddof=1) for r in R1])

        res = dict(
            system=sysname, source=src, natoms=natoms, n_chain=n_chain,
            atoms_per_chain=natoms // n_chain,
            n_frames_total=len(all_steps),
            step_min=int(min(all_steps)), step_max=int(max(all_steps)),
            frames_analyzed=len(steps),
            steps_first=int(steps[0]), steps_last=int(steps[-1]),
            n_bad_mol_frames=n_bad_mol, n_bad_id_frames=n_bad_id,
            R1_mass_all_mean=float(R1.mean()),
            R1_mean_of_frame_means=float(fmean.mean()),
            R1_within_frame_std_mean=float(fstd.mean()),
            R1_temporal_std=float(fmean.std(ddof=1)),
            R2_geom_all_mean=float(R2.mean()),
            R3_mass_heavy_mean=float(R3.mean()),
            R4_geom_heavy_mean=float(R4.mean()),
            R5_mass_backbone_mean=float(R5.mean()) if R5 is not None else None,
            R1_min=float(R1.min()), R1_max=float(R1.max()),
            R3_min=float(R3.min()), R3_max=float(R3.max()),
            e2e_backbone_mean=float(np.mean(e2el)) if e2el else None,
            e2e_backbone_max=float(np.max(e2el)) if e2el else None,
            heavy_frac_outside_box_max=float(max(outbox)),
            heavy_frac_outside_box_mean=float(np.mean(outbox)),
            backbone_bonds=int(nb_bonds),
            backbone_natoms=int(len(path1)),
            dmin_inter=dminl,
            contiguous_chains=bool(contiguous),
            mol_ids_consistent=bool(mol_ok),
        )
        out[f"{src}/{sysname}"] = res
        print(f"[{src}/{sysname}] frames={res['frames_analyzed']}/"
              f"{res['n_frames_total']} steps {res['steps_first']}.."
              f"{res['steps_last']}  ({time.perf_counter()-t0:.0f}s)")
        print(f"   R1 mass/all={res['R1_mass_all_mean']:.4f}  "
              f"R2 geom/all={res['R2_geom_all_mean']:.4f}  "
              f"R3 mass/heavy={res['R3_mass_heavy_mean']:.4f}  "
              f"R4 geom/heavy={res['R4_geom_heavy_mean']:.4f}  "
              f"R5 backbone={('%.4f' % res['R5_mass_backbone_mean']) if R5 is not None else '-'}")
        print(f"   within-frame across-chain std="
              f"{res['R1_within_frame_std_mean']:.4f}   "
              f"temporal std of frame means={res['R1_temporal_std']:.4f}")
        print(f"   backbone {res['backbone_natoms']} heavy atoms / "
              f"{res['backbone_bonds']} bonds; e2e mean="
              f"{res['e2e_backbone_mean']:.2f} max={res['e2e_backbone_max']:.2f}")
        print(f"   min inter-chain / intra-chain (6 probes)="
              f"{[f'{a:.3f}/{b:.3f}' for a, b in dminl]}")
        print(f"   frac heavy atoms outside box: max="
              f"{res['heavy_frac_outside_box_max']:.4f} mean="
              f"{res['heavy_frac_outside_box_mean']:.4f};  "
              f"mol-id mismatches={n_bad_mol}; id mismatches={n_bad_id}")
        print()

    (Path(__file__).parent / "verify_rg3.json").write_text(
        json.dumps(out, indent=1))
    print("wrote verify_rg3.json")


if __name__ == "__main__":
    main()
