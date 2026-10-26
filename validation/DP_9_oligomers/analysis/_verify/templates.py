"""Verifier part 4: single-chain template comparison + initial packing structure."""
import re, sys
from pathlib import Path
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")


def grab_sections(path):
    txt = Path(path).read_text(errors="replace")
    lines = txt.splitlines()
    i, n = 0, len(lines)
    while i < n and not re.match(r"^[\d\.\-]+\s+[\-\d\.eE]+\s+xlo xhi", lines[i].strip()):
        i += 1
    def grab(name):
        nonlocal i, n
        while i < n:
            if lines[i].strip().startswith(name):
                break
            i += 1
        else:
            return []
        i += 1
        while i < n and (not lines[i].strip() or lines[i].strip().startswith("#")):
            i += 1
        rows = []
        while i < n and lines[i].strip():
            rows.append([float(x) for x in lines[i].split()])
            i += 1
        return rows
    out = {}
    out["mass_of_type"] = {int(r[0]): float(r[1]) for r in grab("Masses")}
    out["atoms"] = grab("Atoms")
    out["bonds"] = grab("Bonds")
    return out


def chain_geom(d):
    A = d["atoms"]
    ids = np.array([int(a[0]) for a in A]); typ = np.array([int(a[2]) for a in A])
    xyz = np.array([[a[4], a[5], a[6]] for a in A])
    mt = d["mass_of_type"]
    mass = np.array([mt[int(t)] for t in typ])
    return ids, typ, xyz, mass


for sysname in SYSTEMS:
    print("=" * 95)
    print(f"SYSTEM {sysname}")
    lg_tpl = BASE / "ligpargen" / sysname / "build" / f"{sysname}.lmp"
    mf_tpl = BASE / "mappingff" / sysname / "build" / f"{sysname}.data"
    print(f"  ligpargen template: {lg_tpl.name} exists={lg_tpl.exists()}  "
          f"mappingff template: {mf_tpl.name} exists={mf_tpl.exists()}")
    if not (lg_tpl.exists() and mf_tpl.exists()):
        continue
    la = grab_sections(lg_tpl)
    ma = grab_sections(mf_tpl)
    li, lt, lxyz, lm = chain_geom(la)
    mi, mt_, mxyz, mm = chain_geom(ma)
    print(f"  template natoms: ligpargen={len(li)} mappingff={len(mi)}")
    if len(li) == len(mi):
        print(f"  template atom-id order identical: {np.array_equal(li, mi)}")
        print(f"  template TYPE vector identical: {np.array_equal(lt, mt_)}")
        print(f"  template MASS vector identical: {np.array_equal(lm, mm)}")
        dev = np.abs(lxyz - mxyz)
        print(f"  template XYZ max|dev| = {dev.max():.6f} A   rms = {np.sqrt((dev**2).mean()):.6f} A")
        # check for pure rigid motion: compare internal distance matrices
        def dmat(x):
            n = len(x)
            return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        da, db = dmat(lxyz), dmat(mxyz)
        print(f"  template intra-chain distance matrix max|dev| = {np.abs(da-db).max():.6f} A")
        # Rg
        for tag, x, m in (("ligpargen", lxyz, lm), ("mappingff", mxyz, mm)):
            com = (m[:, None] * x).sum(0) / m.sum()
            rg = np.sqrt((m * ((x - com) ** 2).sum(1)).sum() / m.sum())
            print(f"  [{tag}] single-chain Rg = {rg:.4f} A, COM = {com}")
    print()


# --- initial packing structure of the actual run files ---
print("### INITIAL PACKING OF RUN FILES (filled.data) ###")
for sysname in SYSTEMS:
    print("=" * 95)
    print(f"SYSTEM {sysname}")
    for src in ("ligpargen", "mappingff"):
        d = grab_sections(BASE / src / sysname / "filled.data")
        ids, typ, xyz, mass = chain_geom(d)
        mol = np.array([int(a[1]) for a in d["atoms"]])
        # per-molecule COM
        coms = []
        for m in np.unique(mol):
            sel = mol == m
            coms.append((mass[sel][:, None] * xyz[sel]).sum(0) / mass[sel].sum())
        coms = np.array(coms)
        # per-molecule Rg
        rgs = []
        for m in np.unique(mol):
            sel = mol == m
            c = coms[m - 1]
            rgs.append(np.sqrt((mass[sel] * ((xyz[sel] - c) ** 2).sum(1)).sum() / mass[sel].sum()))
        rgs = np.array(rgs)
        # nearest-neighbour distance between chain COMs
        D = np.linalg.norm(coms[:, None, :] - coms[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        print(f"  [{src}] per-chain Rg: mean={rgs.mean():.4f} std={rgs.std():.4f} "
              f"min={rgs.min():.4f} max={rgs.max():.4f}")
        print(f"  [{src}] chain-COM nearest-neighbour dist: min={D.min():.3f} mean={D[D<1e30].mean():.3f} A")
        globals().setdefault("_coms", {})[(src, sysname)] = coms
    ca = _coms[("ligpargen", sysname)]
    cb = _coms[("mappingff", sysname)]
    # is it a permutation?
    from scipy.optimize import linear_sum_assignment
    try:
        C = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
        r, c = linear_sum_assignment(C)
        print(f"  COM-set matching (Hungarian): max residual = {C[r, c].max():.4f} A "
              f"(0 => identical packing up to chain relabelling)")
    except Exception as e:
        print("  (scipy unavailable:", e, ")")
    print()
