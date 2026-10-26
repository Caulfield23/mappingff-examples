"""Verifier part 3: canonical angle/dihedral connectivity + first-frame coordinate comparison."""
import re
from pathlib import Path
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
SOURCES = ("ligpargen", "mappingff")
ELEM = {15.999: "O", 12.011: "C", 1.008: "H"}


def read_sections(path):
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
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
    out["angles"] = grab("Angles")
    out["dihedrals"] = grab("Dihedrals")
    return out


def first_frame(path, want_step=None):
    """Read the first frame of a dump; return (step, xyz, cols, atom_ids, mol, type)."""
    with open(path, "r", errors="replace") as fh:
        lines = []
        for _ in range(9):
            lines.append(fh.readline())
        hdr = lines[8].split()
        cols = hdr[2:]
        natoms = int(lines[3].split()[0])
        data = []
        for _ in range(natoms):
            data.append(fh.readline().split())
        raw = np.array(data, dtype=float)
        step = int(lines[1].split()[0])
        box = [lines[5], lines[6], lines[7]]
        return step, raw, cols, box


print("### FIRST-FRAME COORDINATE COMPARISON (dump.lammpstrj) ###")
for sysname in SYSTEMS:
    print("=" * 90)
    print(f"SYSTEM {sysname}")
    res = {}
    for src in SOURCES:
        step, raw, cols, hdr = first_frame(BASE / src / sysname / "dump.lammpstrj")
        ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
        ixu, iyu, izu = cols.index("xu"), cols.index("yu"), cols.index("zu")
        iid, imol, ityp = cols.index("id"), cols.index("mol"), cols.index("type")
        res[src] = dict(step=step, raw=raw, cols=cols,
                        xyz=raw[:, [ix, iy, iz]], xyz_u=raw[:, [ixu, iyu, izu]],
                        ids=raw[:, iid].astype(int), mol=raw[:, imol].astype(int),
                        typ=raw[:, ityp].astype(int))
        print(f"  [{src}] first frame step={step}  natoms={raw.shape[0]}  cols={cols}")
    a, b = res["ligpargen"], res["mappingff"]
    print(f"  steps: ligpargen={a['step']} mappingff={b['step']}")
    print(f"  atom id order identical: {np.array_equal(a['ids'], b['ids'])}")
    print(f"  mol vector identical: {np.array_equal(a['mol'], b['mol'])}")
    print(f"  TYPE vector identical: {np.array_equal(a['typ'], b['typ'])}")
    dev_w = np.abs(a["xyz"] - b["xyz"])
    dev_u = np.abs(a["xyz_u"] - b["xyz_u"])
    print(f"  WRAPPED  (x,y,z) max|dev| = {dev_w.max():.6f} A   rms = {np.sqrt((dev_w**2).mean()):.6f} A")
    print(f"  UNWRAPPED(xu,yu,zu) max|dev| = {dev_u.max():.6f} A   rms = {np.sqrt((dev_u**2).mean()):.6f} A")
    # per-atom deviation distribution
    per_atom = dev_u.max(axis=1)
    print(f"  per-atom max|dev| (unwrapped): min={per_atom.min():.6f} "
          f"median={np.median(per_atom):.6f} max={per_atom.max():.6f}")
    # per-molecule Rg of the first frame for each
    def per_mol_rms(src):
        r = res[src]
        return r
    # how many atoms are within 1e-6
    nclose = int((per_atom < 1e-6).sum())
    print(f"  atoms with |dev| < 1e-6: {nclose}/{len(per_atom)}")
    # check for rigid translation
    shift = (a["xyz_u"] - b["xyz_u"]).mean(axis=0)
    print(f"  mean per-axis shift = {shift}")
    print()


print("### CANONICAL ANGLE / DIHEDRAL CONNECTIVITY ###")
for sysname in SYSTEMS:
    print(f"--- {sysname} ---")
    S = {s: read_sections(BASE / s / sysname / "filled.data") for s in SOURCES}
    for s in SOURCES:
        d = S[s]
        mt = d["mass_of_type"]
        atoms = d["atoms"]
        typ = np.array([int(a[2]) for a in atoms])
        elem = np.array([ELEM.get(round(mt[int(t)], 3), "?") for t in typ])
        ang = d["angles"]
        dih = d["dihedrals"]
        # canonical angle key: (min(a,c), b, max(a,c))
        ka = sorted((min(int(x[2]), int(x[4])), int(x[3]), max(int(x[2]), int(x[4]))) for x in ang)
        # canonical dihedral key: (min(a,d), b, c, max(a,d))
        kd = sorted((min(int(x[2]), int(x[5])), int(x[3]), int(x[4]), max(int(x[2]), int(x[5]))) for x in dih)
        S[s]["ka"] = ka; S[s]["kd"] = kd
        S[s]["elem"] = elem
        print(f"  [{s}] angles={len(ang)} canonical unique={len(set(ka))}  "
              f"dihedrals={len(dih)} canonical unique={len(set(kd))}")
    print(f"  ANGLE connectivity identical (canonical): {S['ligpargen']['ka'] == S['mappingff']['ka']}")
    print(f"  DIHEDRAL connectivity identical (canonical): {S['ligpargen']['kd'] == S['mappingff']['kd']}")
    if S['ligpargen']['ka'] != S['mappingff']['ka']:
        from collections import Counter
        ca, cb = Counter(S['ligpargen']['ka']), Counter(S['mappingff']['ka'])
        print("   ligpargen-only:", list((ca - cb).items())[:5])
        print("   mappingff-only:", list((cb - ca).items())[:5])
    if S['ligpargen']['kd'] != S['mappingff']['kd']:
        from collections import Counter
        ca, cb = Counter(S['ligpargen']['kd']), Counter(S['mappingff']['kd'])
        print("   ligpargen-only:", list((ca - cb).items())[:5])
        print("   mappingff-only:", list((cb - ca).items())[:5])
    print()
