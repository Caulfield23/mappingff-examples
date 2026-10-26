"""Verifier part 8: dihedral sense reversal (numeric), improper counts, n_min, first-frame box."""
import re
from pathlib import Path
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
ELEM = {15.999: "O", 12.011: "C", 1.008: "H"}


def read_data(path):
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
    masses = grab("Masses")
    atoms = grab("Atoms")
    bonds = grab("Bonds")
    dihed = grab("Dihedrals")
    improps = grab("Impropers")
    return masses, atoms, bonds, dihed, improps


def torsion(p1, p2, p3, p4):
    """LAMMPS/IUPAC dihedral in degrees."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    c1 = np.cross(b1, b2)
    c2 = np.cross(b2, b3)
    b2u = b2 / np.linalg.norm(b2)
    x = np.dot(c1, c2)
    y = np.dot(np.cross(c1, c2), b2u) * -1.0   # sign convention (see note below)
    return np.degrees(np.arctan2(y, x))


def torsion_ref(p1, p2, p3, p4):
    """Standard IUPAC dihedral: sign of atan2((n1 x n2).b2hat, n1.n2)."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m, n2), np.dot(n1, n2)))


print("### Numeric check: is (a,b,c,d) vs (a,c,b,d) a sign flip of the torsion? ###")
rng = np.random.default_rng(7)
maxdiff = 0.0
for _ in range(2000):
    p = rng.normal(size=(4, 3))
    t1 = torsion_ref(p[0], p[1], p[2], p[3])
    t2 = torsion_ref(p[0], p[2], p[1], p[3])
    # compare energy for OPLS-style (even) potential
    for K in (1.0, 0.3, -0.5, 2.0):
        for n in (1, 2, 3, 4):
            e1 = np.cos(np.radians(n * t1))
            e2 = np.cos(np.radians(n * t2))
            maxdiff = max(maxdiff, abs(e1 - e2))
print(f"  max |cos(n*phi1) - cos(n*phi2)| over 2000 random quadruples, n=1..4: {maxdiff:.3e}")
print("  (0 => the OPLS dihedral functional form is EVEN in phi, so a reversed")
print("   dihedral atom ordering gives an IDENTICAL energy.)")

# --- quantify the reversal in the actual files ---
print()
print("### Dihedral atom-ordering: ligpargen vs mappingff ###")
for sysname in SYSTEMS:
    res = {}
    for src in ("ligpargen", "mappingff"):
        masses, atoms, bonds, dihed, improps = read_data(BASE / src / sysname / "filled.data")
        mt = {int(r[0]): float(r[1]) for r in masses}
        typ = np.array([int(a[2]) for a in atoms])
        el = np.array([ELEM.get(round(mt[int(t)], 3), "?") for t in typ])
        quads = [(int(d[2]), int(d[3]), int(d[4]), int(d[5])) for d in dihed]
        res[src] = (quads, el, improps)
    qa, el, imp_a = res["ligpargen"]
    qb, elb, imp_b = res["mappingff"]
    # map quad -> quad
    mb = {}
    for q in qb:
        mb.setdefault(q, 0); mb[q] += 1
    same = rev = other = 0
    for q in qa:
        if q in mb:
            same += 1
        elif (q[3], q[2], q[1], q[0]) in mb:
            rev += 1
        else:
            other += 1
    print(f"  {sysname}: n_dih={len(qa)}  identical-order={same}  "
          f"reversed-order={rev}  unmatched={other}")
    # element quadruple identity
    ea = sorted(tuple(sorted(el[i-1] for i in q)) for q in qa)
    eb = sorted(tuple(sorted(elb[i-1] for i in q)) for q in qb)
    print(f"     element-quad multiset identical: {ea == eb}")
    # SAME atom quadruple set?
    sa = sorted(tuple(sorted(q)) for q in qa)
    sb = sorted(tuple(sorted(q)) for q in qb)
    print(f"     atom-quad multiset identical: {sa == sb}")
    print(f"     impropers in filled.data: ligpargen={len(imp_a)} mappingff={len(imp_b)}")


print()
print("### Impropers in the single-chain templates (which fill_box.py discards) ###")
for sysname in SYSTEMS:
    for src, fn in (("ligpargen", f"{sysname}.lmp"), ("mappingff", f"{sysname}.data")):
        p = BASE / src / sysname / "build" / fn
        if not p.exists():
            continue
        txt = p.read_text(errors="replace")
        m = re.search(r"^(\d+)\s+impropers\s*$", txt, re.M)
        n_imp = 0
        if m:
            lines = txt.splitlines()
            i = lines.index(m.group(0))
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and lines[j].strip():
                n_imp += 1; j += 1
        print(f"  {src}/{sysname}/{fn}: header says {m.group(1) if m else 0} impropers, "
              f"section has {n_imp} rows")


print()
print("### n_min (minimiser steps) and first-frame box ###")
for sysname in SYSTEMS:
    for src in ("ligpargen", "mappingff"):
        log = BASE / src / sysname / "log.lammps"
        txt = log.read_text(errors="replace")
        m = re.search(r"Loop time of [\d.eE+-]+ on \d+ procs for (\d+) steps", txt)
        # first frame box
        with open(BASE / src / sysname / "dump.lammpstrj", "r", errors="replace") as fh:
            hdr = [fh.readline() for _ in range(9)]
        box = [hdr[5].split(), hdr[6].split(), hdr[7].split()]
        L = [float(b[1]) - float(b[0]) for b in box]
        print(f"  {sysname:8s} {src:9s} n_min={m.group(1) if m else '?':>5s}  "
              f"frame1 L = {L[0]:.6f} {L[1]:.6f} {L[2]:.6f}  "
              f"lo={box[0][0]} hi={box[0][1]}")
