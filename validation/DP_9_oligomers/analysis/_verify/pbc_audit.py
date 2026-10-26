"""
Independent PBC audit of analyze_density_rg.py's Rg computation.

For each selected system, for every frame in [1.5e6, 2.0e6]:
  (a) Rg from UNWRAPPED xu,yu,zu
  (b) Rg from WRAPPED x,y,z, no minimum-image correction
  (c) Rg from WRAPPED x,y,z WITH minimum-image correction relative to the chain COM
      (box bounds taken from that frame's own ITEM: BOX BOUNDS header)

Also: per-frame chain boundary-straddling counts, sanity checks on xu
(bond-length stability, box shrinkage, xu-vs-x image offsets).
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

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
OUT = BASE / "analysis" / "_verify"
OUT.mkdir(parents=True, exist_ok=True)

STEP_LO, STEP_HI = 1_500_000, 2_000_000
FRAME_MARK = b"ITEM: TIMESTEP\n"


# --------------------------------------------------------------------------- #
def read_topology(path: Path):
    """Return mol_of_atom, mass_of_atom, natoms (1-indexed arrays)."""
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    natoms = None
    mass_of_type = {}
    mol_of_atom = None
    typ_of_atom = None

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
        i += 1

    mass_of_atom = np.array([mass_of_type[t] for t in typ_of_atom], dtype=np.float64)
    return mol_of_atom, mass_of_atom, natoms


# --------------------------------------------------------------------------- #
def chain_rg_unwrapped(xyz, chain, mass, nchain):
    w = np.bincount(chain, weights=mass, minlength=nchain)
    com = np.empty((nchain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass * xyz[:, d], minlength=nchain)
    com /= w[:, None]
    diff = xyz - com[chain]
    r2 = np.einsum("ij,ij->i", diff, diff)
    num = np.bincount(chain, weights=mass * r2, minlength=nchain)
    return np.sqrt(num / w)


def chain_rg_mic(xw, chain, mass, nchain, L):
    """MIC-corrected Rg from wrapped coords.  Returns (rg, pos_unwrapped_like, com)."""
    w = np.bincount(chain, weights=mass, minlength=nchain)
    com = np.empty((nchain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass * xw[:, d], minlength=nchain)
    com /= w[:, None]

    pos = xw.copy()
    for _ in range(50):
        d = pos - com[chain]
        for k in range(3):
            d[:, k] -= L[k] * np.round(d[:, k] / L[k])
        pos = com[chain] + d
        newcom = np.empty((nchain, 3))
        for k in range(3):
            newcom[:, k] = np.bincount(chain, weights=mass * pos[:, k], minlength=nchain)
        newcom /= w[:, None]
        delta = np.max(np.abs(newcom - com))
        com = newcom
        if delta < 1e-12:
            break
    # final consistency: re-apply MIC relative to converged COM
    d = xw - com[chain]
    for k in range(3):
        d[:, k] -= L[k] * np.round(d[:, k] / L[k])
    pos = com[chain] + d
    diff = pos - com[chain]
    r2 = np.einsum("ij,ij->i", diff, diff)
    num = np.bincount(chain, weights=mass * r2, minlength=nchain)
    rg = np.sqrt(num / w)
    # integer image shift applied to each atom relative to the COM
    kimg = np.zeros((len(xw), 3), dtype=np.int64)
    for k in range(3):
        kimg[:, k] = np.round((xw[:, k] - com[chain][:, k]) / L[k]).astype(np.int64)
    return rg, pos, com, kimg


def chain_com_unwrapped(xyu, chain, mass, nchain):
    w = np.bincount(chain, weights=mass, minlength=nchain)
    com = np.empty((nchain, 3))
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass * xyu[:, d], minlength=nchain)
    com /= w[:, None]
    return com


# --------------------------------------------------------------------------- #
def iter_frames(traj: Path):
    """Yield (step, bounds_lines, header_line, atom_lines) for every frame."""
    with open(traj, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        starts = []
        pos = 0
        while True:
            hit = mm.find(FRAME_MARK, pos)
            if hit < 0:
                break
            starts.append(hit)
            pos = hit + len(FRAME_MARK)
        starts.append(size)
        n_frames = len(starts) - 1

        for k in range(n_frames):
            blob = mm[starts[k]:starts[k + 1]]
            lines = blob.split(b"\n")
            step = int(lines[1])
            yield step, lines, starts[k], starts[k + 1]
        mm.close()
        yield None, None, None, None


def parse_frame(lines):
    """Return (step, lo[3], hi[3], tilt(3 or None), cols, atom_block_bytes)."""
    step = int(lines[1])
    ibox = next(i for i, l in enumerate(lines) if l.startswith(b"ITEM: BOX BOUNDS"))
    iatom = next(i for i, l in enumerate(lines) if l.startswith(b"ITEM: ATOMS"))
    box_hdr = lines[ibox].split()
    bound_lines = lines[ibox + 1:ibox + 4]
    lo = np.empty(3)
    hi = np.empty(3)
    for k in range(3):
        p = bound_lines[k].split()
        lo[k] = float(p[0])
        hi[k] = float(p[1])
        extra = len(p) - 2
    tilt = None
    # triclinic dump: "ITEM: BOX BOUNDS xy xz yz pp pp pp" -> 3 extra floats
    if len(bound_lines[0].split()) == 4:
        tilt = np.array([float(bound_lines[0].split()[2]),
                         float(bound_lines[1].split()[2]),
                         float(bound_lines[2].split()[2])])
    ncols = len(lines[iatom].split()) - 2
    natoms = len(lines[iatom + 1:iatom + 1 + 100000])
    # count atom lines properly: consecutive non-empty lines after header
    a = iatom + 1
    while a < len(lines) and lines[a].strip():
        a += 1
    atom_lines = lines[iatom + 1:a]
    natoms = len(atom_lines)
    cols = [c.decode() for c in lines[iatom].split()[2:]]
    return step, lo, hi, tilt, cols, atom_lines, box_hdr


# --------------------------------------------------------------------------- #
def audit_run(src, sysname, do_all_frames=False):
    run = BASE / src / sysname
    mol_of_atom, mass_of_atom, natoms = read_topology(run / "filled.data")
    uniq = np.unique(mol_of_atom)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol_of_atom.tolist()], dtype=np.int64)
    nchain = len(uniq)

    res = {
        "system": sysname, "source": src, "natoms": int(natoms),
        "nchain": int(nchain), "frames": 0,
        "max_abs_dev_ac": 0.0, "n_mismatch_ac": 0,
        "max_rel_dev_ac": 0.0,
        "max_abs_dev_bc": 0.0, "n_mismatch_bc": 0,
        "b_mean_abs_dev": 0.0, "b_max_abs_dev": 0.0,
        "b_n_over_1pct": 0, "b_n_over_10pct": 0, "b_n_over_50pct": 0,
        "straddle_frames": [], "straddle_chain_frames": 0, "total_chain_frames": 0,
        "n_straddle_span": [], "n_straddle_image": [],
        "box": [], "tilt_found": False, "box_hdrs": set(),
        "worst_cases": [],
        "per_frame_rg_a_mean": [],
    }
    bond_pairs = []
    t0 = time.perf_counter()

    for step, lines, _s, _e in iter_frames(run / "dump.lammpstrj"):
        if lines is None:
            break
        if not (do_all_frames or (STEP_LO <= step <= STEP_HI)):
            continue
        step, lo, hi, tilt, cols, atom_lines, box_hdr = parse_frame(lines)
        res["box_hdrs"].add(b" ".join(box_hdr).decode())
        if tilt is not None:
            res["tilt_found"] = True
        L = hi - lo
        res["box"].append((step, float(L[0]), float(L[1]), float(L[2])))

        blob = b" ".join(atom_lines)
        arr = np.array(blob.split(), dtype=np.float64).reshape(len(atom_lines), len(cols))
        imol = arr[:, cols.index("mol")].astype(np.int64)
        ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
        ixu, iyu, izu = cols.index("xu"), cols.index("yu"), cols.index("zu")
        xw = arr[:, [ix, iy, iz]]
        xu = arr[:, [ixu, iyu, izu]]

        rg_a = chain_rg_unwrapped(xu, chain, mass_of_atom, nchain)
        rg_b = chain_rg_unwrapped(xw, chain, mass_of_atom, nchain)
        rg_c, pos_c, com_c, kimg = chain_rg_mic(xw, chain, mass_of_atom, nchain, L)

        dev_ac = np.abs(rg_a - rg_c)
        dev_bc = np.abs(rg_a - rg_b)
        res["max_abs_dev_ac"] = max(res["max_abs_dev_ac"], float(dev_ac.max()))
        res["max_abs_dev_bc"] = max(res["max_abs_dev_bc"], float(dev_bc.max()))
        res["n_mismatch_ac"] += int((dev_ac > 1e-8).sum())
        res["n_mismatch_bc"] += int((dev_bc > 1e-8).sum())
        relac = np.divide(dev_ac, rg_a, out=np.zeros_like(rg_a), where=rg_a > 0)
        res["max_rel_dev_ac"] = max(res["max_rel_dev_ac"], float(relac.max()))
        res["b_mean_abs_dev"] += float(dev_bc.mean())
        res["b_max_abs_dev"] = max(res["b_max_abs_dev"], float(dev_bc.max()))
        res["b_n_over_1pct"] += int((dev_bc > 0.01 * rg_a).sum())
        res["b_n_over_10pct"] += int((dev_bc > 0.10 * rg_a).sum())
        res["b_n_over_50pct"] += int((dev_bc > 0.50 * rg_a).sum())

        # --- straddling metrics -------------------------------------------
        # (i) span > L/2 on wrapped coords
        span_straddle = np.zeros(nchain, dtype=bool)
        for k in range(3):
            mn = np.full(nchain, np.inf)
            mx = np.full(nchain, -np.inf)
            np.minimum.at(mn, chain, xw[:, k])
            np.maximum.at(mx, chain, xw[:, k])
            span_straddle |= ((mx - mn) > 0.5 * L[k])
        # (ii) atoms in more than one image relative to the chain COM (from xu gauge)
        com_u = chain_com_unwrapped(xu, chain, mass_of_atom, nchain)
        kimg_u = np.zeros((natoms, 3), dtype=np.int64)
        for k in range(3):
            kimg_u[:, k] = np.round((xu[:, k] - com_u[chain][:, k]) / L[k]).astype(np.int64)
        img_straddle = np.zeros(nchain, dtype=bool)
        for k in range(3):
            anyk = np.zeros(nchain, dtype=bool)
            np.logical_or.at(anyk, chain, kimg_u[:, k] != 0)
            img_straddle |= anyk

        res["n_straddle_span"].append(int(span_straddle.sum()))
        res["n_straddle_image"].append(int(img_straddle.sum()))
        res["straddle_chain_frames"] += int((span_straddle | img_straddle).sum())
        res["total_chain_frames"] += nchain
        res["straddle_frames"].append((step, int(span_straddle.sum()),
                                       int(img_straddle.sum())))
        res["per_frame_rg_a_mean"].append(float(rg_a.mean()))

        # worst (a) vs (c) mismatch bookkeeping
        if dev_ac.max() > 1e-8:
            c = int(np.argmax(dev_ac))
            res["worst_cases"].append({
                "step": int(step), "chain": int(c),
                "rg_a": float(rg_a[c]), "rg_c": float(rg_c[c]),
                "dev": float(dev_ac[c])})

        # pick two bonded-looking atoms of chain 0 for the xu sanity check
        if not bond_pairs:
            m0 = chain == 0
            ids0 = np.where(m0)[0]
            # choose the closest pair of atoms in chain 0 (bonded)
            p = xu[ids0]
            dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            np.fill_diagonal(dd, np.inf)
            a, b = np.unravel_index(np.argmin(dd), dd.shape)
            bond_pairs = [(int(ids0[a]), int(ids0[b]))]
            res["bond_pair"] = bond_pairs[0]
            res["bond_ref_dist"] = float(dd[a, b])

    res["frames"] = len(res["per_frame_rg_a_mean"])
    res["rg_a_mean"] = float(np.mean(res["per_frame_rg_a_mean"]))
    res["b_mean_abs_dev"] = res["b_mean_abs_dev"] / max(1, res["frames"])
    res["wall_s"] = time.perf_counter() - t0
    res["box_hdrs"] = sorted(res["box_hdrs"])
    res.pop("per_frame_rg_a_mean")
    return res


def xu_sanity(src, sysname):
    """Bond-length stability + box shrinkage + xu/x image-offset consistency."""
    run = BASE / src / sysname
    mol_of_atom, mass_of_atom, natoms = read_topology(run / "filled.data")
    uniq = np.unique(mol_of_atom)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol_of_atom.tolist()], dtype=np.int64)

    # candidate bonds: consecutive atom ids within a chain (topology adjacency
    # is unknown -> use the closest pair per chain from the first frame)
    first = None
    for step, lines, _s, _e in iter_frames(run / "dump.lammpstrj"):
        if lines is None:
            break
        first = parse_frame(lines)
        break
    step, lo, hi, tilt, cols, atom_lines, box_hdr = first
    blob = b" ".join(atom_lines)
    arr = np.array(blob.split(), dtype=np.float64).reshape(len(atom_lines), len(cols))
    ixu, iyu, izu = cols.index("xu"), cols.index("yu"), cols.index("zu")
    ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
    xu0 = arr[:, [ixu, iyu, izu]]
    xw0 = arr[:, [ix, iy, iz]]

    # for the first 5 chains pick the closest atom pair (bonded)
    pairs = []
    for c in range(5):
        ids = np.where(chain == c)[0]
        p = xu0[ids]
        dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        a, b = np.unravel_index(np.argmin(dd), dd.shape)
        pairs.append((int(ids[a]), int(ids[b]), float(dd[a, b])))

    out = {"pairs": pairs, "frames": 0, "box": [], "bond": [], "resid": [],
           "n_nonint_offset": 0, "n_frames_offset_bad": 0}
    for step, lines, _s, _e in iter_frames(run / "dump.lammpstrj"):
        if lines is None:
            break
        step, lo, hi, tilt, cols, atom_lines, box_hdr = parse_frame(lines)
        L = hi - lo
        blob = b" ".join(atom_lines)
        arr = np.array(blob.split(), dtype=np.float64).reshape(len(atom_lines), len(cols))
        xu = arr[:, [ixu, iyu, izu]]
        xw = arr[:, [ix, iy, iz]]
        d = []
        for (a, b, ref) in pairs:
            d.append(float(np.linalg.norm(xu[a] - xu[b])))
        out["bond"].append(d)
        out["box"].append((int(step), float(L[0])))
        # image offset consistency: xu - x should be an integer multiple of L
        resid = np.empty((natoms, 3))
        bad = 0
        for k in range(3):
            kk = (xu[:, k] - xw[:, k]) / L[k]
            r = kk - np.round(kk)
            resid[:, k] = r
            bad += int((np.abs(r) > 1e-4).sum())
        out["resid"].append(float(np.abs(resid).max()))
        out["n_nonint_offset"] += bad
        out["n_frames_offset_bad"] += int(bad > 0)
        out["frames"] += 1
    out["bond"] = np.array(out["bond"])
    out["bond_min"] = out["bond"].min(axis=0).tolist()
    out["bond_max"] = out["bond"].max(axis=0).tolist()
    out["bond_rel_fluct"] = ((out["bond"].max(axis=0) - out["bond"].min(axis=0))
                             / out["bond"][0]).tolist()
    out["bond"] = out["bond"].tolist()
    out["box"] = out["box"][::20] + out["box"][-3:]
    out["resid"] = out["resid"][::20] + out["resid"][-3:]
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if which == "audit":
        runs = [("ligpargen", "9_pva"), ("mappingff", "9_pva"),
                ("ligpargen", "9_peo"), ("mappingff", "9_peo"),
                ("ligpargen", "9_ps"), ("mappingff", "9_ps")]
        allres = []
        for src, sysname in runs:
            r = audit_run(src, sysname)
            allres.append(r)
            print(f"\n### {src}/{sysname}")
            print(json.dumps({k: v for k, v in r.items()
                              if k not in ("worst_cases", "straddle_frames", "box")},
                             indent=1, default=float))
            print("  box (step,Lx):", [f"{s}:{l:.2f}" for s, l in r["box"][:3]],
                  "...", [f"{s}:{l:.2f}" for s, l in r["box"][-3:]])
            sf = r["straddle_frames"]
            ns = np.array([x[1] for x in sf])
            ni = np.array([x[2] for x in sf])
            print(f"  straddle span> L/2 : min={ns.min()} max={ns.max()} mean={ns.mean():.2f}")
            print(f"  straddle image!=0  : min={ni.min()} max={ni.max()} mean={ni.mean():.2f}")
            print(f"  worst (a)-(c) cases: {r['worst_cases'][:5]}")
            with open(OUT / f"audit_{src}_{sysname}.json", "w") as fh:
                json.dump(r, fh, indent=1, default=float)
        with open(OUT / "audit_all.json", "w") as fh:
            json.dump(allres, fh, indent=1, default=float)
    elif which == "sanity":
        for src, sysname in [("ligpargen", "9_pva"), ("mappingff", "9_pva"),
                             ("ligpargen", "9_peo"), ("mappingff", "9_ps")]:
            o = xu_sanity(src, sysname)
            print(f"\n### sanity {src}/{sysname}")
            print("  pairs:", o["pairs"])
            print("  bond min:", o["bond_min"])
            print("  bond max:", o["bond_max"])
            print("  bond rel fluct:", o["bond_rel_fluct"])
            print("  n frames offset not integer:", o["n_frames_offset_bad"], "/", o["frames"])
            print("  max |resid| sampled:", max(o["resid"]))
            print("  box:", [f"{s}:{l:.2f}" for s, l in o["box"]])
            with open(OUT / f"sanity_{src}_{sysname}.json", "w") as fh:
                json.dump(o, fh, indent=1, default=float)
