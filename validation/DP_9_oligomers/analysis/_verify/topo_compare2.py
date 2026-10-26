"""Verifier part 2: element-level consistency, angle/dihedral topology, charges, initial coords."""
import sys, re
from pathlib import Path
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
SOURCES = ("ligpargen", "mappingff")

ELEM = {15.999: "O", 12.011: "C", 1.008: "H"}


def read_data(path):
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    i, n = 0, len(lines)
    out = {}
    while i < n:
        s = lines[i].strip()
        if re.match(r"^[\d\.\-]+\s+[\-\d\.eE]+\s+xlo xhi", s):
            break
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
    masses = grab("Masses")
    out["mass_of_type"] = {int(r[0]): float(r[1]) for r in masses}
    out["atoms"] = grab("Atoms")
    out["bonds"] = grab("Bonds")
    out["angles"] = grab("Angles")
    out["dihedrals"] = grab("Dihedrals")
    return out


def summarize(path):
    d = read_data(path)
    A = d["atoms"]
    ids = np.array([int(a[0]) for a in A]); mol = np.array([int(a[1]) for a in A])
    typ = np.array([int(a[2]) for a in A]); q = np.array([float(a[3]) for a in A])
    xyz = np.array([[a[4], a[5], a[6]] for a in A])
    mt = d["mass_of_type"]
    mass = np.array([mt[int(t)] for t in typ])
    elem = np.array([ELEM.get(round(m, 3), "?") for m in mass])
    B = d["bonds"]; ANG = d["angles"]; DIH = d["dihedrals"]
    return dict(ids=ids, mol=mol, typ=typ, q=q, xyz=xyz, mass=mass, elem=elem,
                mt=mt, B=B, ANG=ANG, DIH=DIH)


def key(rows, idx):
    """canonical sorted tuple key for rows"""
    return tuple(sorted(tuple(int(r[i]) for i in idx) for r in rows))


for sysname in SYSTEMS:
    print("=" * 100)
    print(f"SYSTEM {sysname}")
    print("=" * 100)
    S = {s: summarize(BASE / s / sysname / "filled.data") for s in SOURCES}
    a, b = S["ligpargen"], S["mappingff"]

    # ---- bonds as element pairs (chemistry invariant of the FF) ----
    def bond_elem(r):
        return sorted(tuple(sorted((r["elem"][i-1], r["elem"][j-1])))
                      for i, j in ((int(x[2]), int(x[3])) for x in r["B"]))
    ea, eb = bond_elem(a), bond_elem(b)
    print(f"[BOND element-pair multiset identical]: {ea == eb}")
    if ea != eb:
        from collections import Counter
        ca, cb = Counter(map(tuple, ea)), Counter(map(tuple, eb))
        print("   ligpargen only:", (ca - cb).most_common(10))
        print("   mappingff only:", (cb - ca).most_common(10))

    # ---- is bond-type -> element-pair a function (chemically consistent typing)? ----
    def type_elem_map(r):
        m = {}
        for x in r["B"]:
            t = int(x[1]); i, j = int(x[2]), int(x[3])
            ep = tuple(sorted((r["elem"][i-1], r["elem"][j-1])))
            m.setdefault(t, set()).add(ep)
        return m
    for tag, r in (("ligpargen", a), ("mappingff", b)):
        m = type_elem_map(r)
        multi = {t: v for t, v in m.items() if len(v) > 1}
        print(f"[{tag}] bond types: {len(m)}; types mapping to >1 element pair: {len(multi)}")
        if multi:
            for t, v in list(multi.items())[:6]:
                print(f"      type {t}: {sorted(v)}")
        # show the mapping table for mappingff (small)
        if tag == "mappingff":
            print("      mappingff bond type -> element pair table:")
            for t in sorted(m):
                print(f"        type {t:3d} -> {sorted(m[t])}")

    # ---- bond connectivity identity (already known true) - confirm type->pair consistency across runs ----
    # Does the ELEMENT PAIR imply the same bond type in both runs? (i.e. same chem -> same params)
    def pair_to_types(r):
        m = {}
        for x in r["B"]:
            t = int(x[1]); i, j = int(x[2]), int(x[3])
            ep = tuple(sorted((r["elem"][i-1], r["elem"][j-1])))
            m.setdefault(ep, set()).add(t)
        return m
    for tag, r in (("ligpargen", a), ("mappingff", b)):
        m = pair_to_types(r)
        multi = {k: v for k, v in m.items() if len(v) > 1}
        print(f"[{tag}] distinct element pairs: {len(m)}; pairs mapped to >1 bond type: {len(multi)}")

    # ---- angles / dihedrals connectivity ----
    if a["ANG"] and b["ANG"]:
        ka = key(a["ANG"], [2, 3, 4]); kb = key(b["ANG"], [2, 3, 4])
        print(f"[ANGLE (a,b,c) connectivity identical]: {ka == kb}")
        ea = sorted(tuple(sorted((a['elem'][i-1], a['elem'][j-1], a['elem'][k-1])))
                    for i, j, k in ((int(x[2]), int(x[3]), int(x[4])) for x in a["ANG"]))
        eb = sorted(tuple(sorted((b['elem'][i-1], b['elem'][j-1], b['elem'][k-1])))
                    for i, j, k in ((int(x[2]), int(x[3]), int(x[4])) for x in b["ANG"]))
        print(f"[ANGLE element-triple multiset identical]: {ea == eb}")
        # angle type -> element triple function
        def atype_map(r):
            m = {}
            for x in r["ANG"]:
                t = int(x[1]); i, j, k = int(x[2]), int(x[3]), int(x[4])
                ep = tuple(sorted((r["elem"][i-1], r["elem"][j-1], r["elem"][k-1])))
                m.setdefault(t, set()).add(ep)
            return m
        for tag, r in (("ligpargen", a), ("mappingff", b)):
            m = atype_map(r)
            multi = {t: v for t, v in m.items() if len(v) > 1}
            print(f"[{tag}] angle types={len(m)}; >1 element-triple: {len(multi)}")

    if a["DIH"] and b["DIH"]:
        ka = key(a["DIH"], [2, 3, 4, 5]); kb = key(b["DIH"], [2, 3, 4, 5])
        print(f"[DIHEDRAL (a,b,c,d) connectivity identical]: {ka == kb}")
        ea = sorted(tuple(sorted((a['elem'][i-1], a['elem'][j-1], a['elem'][k-1], a['elem'][l-1])))
                    for i, j, k, l in ((int(x[2]), int(x[3]), int(x[4]), int(x[5])) for x in a["DIH"]))
        eb = sorted(tuple(sorted((b['elem'][i-1], b['elem'][j-1], b['elem'][k-1], b['elem'][l-1])))
                    for i, j, k, l in ((int(x[2]), int(x[3]), int(x[4]), int(x[5])) for x in b["DIH"]))
        print(f"[DIHEDRAL element-quad multiset identical]: {ea == eb}")
        def dtype_map(r):
            m = {}
            for x in r["DIH"]:
                t = int(x[1]); i, j, k, l = int(x[2]), int(x[3]), int(x[4]), int(x[5])
                ep = tuple(sorted((r["elem"][i-1], r["elem"][j-1], r["elem"][k-1], r["elem"][l-1])))
                m.setdefault(t, set()).add(ep)
            return m
        for tag, r in (("ligpargen", a), ("mappingff", b)):
            m = dtype_map(r)
            multi = {t: v for t, v in m.items() if len(v) > 1}
            print(f"[{tag}] dihedral types={len(m)}; >1 element-quad: {len(multi)}")

    # ---- charges ----
    print(f"[CHARGE] ligpargen sum={a['q'].sum():.6f}  mappingff sum={b['q'].sum():.6f}")
    print(f"[CHARGE] per-atom identical: {np.allclose(a['q'], b['q'], atol=1e-6)}  "
          f"max|dq|={np.abs(a['q']-b['q']).max():.4f}  "
          f"MAE={np.abs(a['q']-b['q']).mean():.4f}  RMS={np.sqrt((a['q']-b['q'])**2).mean():.4f}")
    print(f"[CHARGE] |q|>0.05 fraction: ligpargen={np.mean(np.abs(a['q'])>0.05):.3f} "
          f"mappingff={np.mean(np.abs(b['q'])>0.05):.3f}")

    # ---- initial coordinates in filled.data ----
    print(f"[INITIAL XYZ in filled.data] identical: {np.allclose(a['xyz'], b['xyz'], atol=1e-6)}  "
          f"max|dev|={np.abs(a['xyz']-b['xyz']).max():.6e} A")
    print(f"[INITIAL XYZ] identical exactly: {np.array_equal(a['xyz'], b['xyz'])}")
    print()


