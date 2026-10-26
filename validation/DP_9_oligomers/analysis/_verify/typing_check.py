"""Verifier part 11: does mappingff's reduced typing preserve distinct chemical environments?"""
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
ELEM = {15.999: "O", 12.011: "C", 1.008: "H"}


def rd(p):
    lines = Path(p).read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines) and not re.match(r"^[\d\.\-]+\s+[\-\d\.eE]+\s+xlo xhi", lines[i].strip()):
        i += 1
    def grab(name):
        nonlocal i
        while i < len(lines):
            if lines[i].strip().startswith(name): break
            i += 1
        else: return []
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("#")): i += 1
        rows = []
        while i < len(lines) and lines[i].strip():
            rows.append([float(x) for x in lines[i].split()]); i += 1
        return rows
    mt = {int(r[0]): float(r[1]) for r in grab("Masses")}
    atoms = grab("Atoms"); bonds = grab("Bonds"); angles = grab("Angles")
    dihed = grab("Dihedrals")
    return mt, atoms, bonds, angles, dihed


for sysname in SYSTEMS:
    print("=" * 100)
    print(f"SYSTEM {sysname}")
    print("=" * 100)
    data = {}
    for src in ("ligpargen", "mappingff"):
        mt, atoms, bonds, angles, dihed = rd(BASE/src/sysname/"filled.data")
        typ = np.array([int(a[2]) for a in atoms])
        el = np.array([ELEM.get(round(mt[int(t)],3),"?") for t in typ])
        mol = np.array([int(a[1]) for a in atoms])
        data[src] = (typ, el, mol, bonds, angles, dihed)
    ta, ea, ma, ba, aa, da = data["ligpargen"]
    tb, eb, mb, bb, ab, db = data["mappingff"]

    # ---- per-atom-type chemical signature in each run ----
    def atom_sig(src):
        typ, el, mol, bonds, angles, dihed = data[src]
        nper = len(mol)//100
        # local (within-chain) neighbour element multiset + degree
        nbr_el = defaultdict(Counter)
        for b in bonds:
            x, y = int(b[2]), int(b[3])
            nbr_el[x][el[y-1]] += 1
            nbr_el[y][el[x-1]] += 1
        sig = {}
        for i in range(1, nper+1):
            key = (el[i-1], tuple(sorted(nbr_el[i].items())))
            sig.setdefault(key, set()).add(typ[i-1])
        return sig
    sa, sb = atom_sig("ligpargen"), atom_sig("mappingff")
    print(f"  distinct (element, neighbour-element-multiset) environments in chain 1: "
          f"ligpargen={len(sa)}  mappingff={len(sb)}")
    # how many environments does each run split into >1 type?
    def multi(sig):
        return {k: v for k, v in sig.items() if len(v) > 1}
    print(f"     ligpargen: environments split into >1 atom TYPE: {len(multi(sa))}")
    print(f"     mappingff: environments split into >1 atom TYPE: {len(multi(sb))}")
    # conversely: mappingff types that merge distinct environments
    t2env = defaultdict(set)
    for k, v in sb.items():
        for t in v:
            t2env[t].add(k)
    merged = {t: v for t, v in t2env.items() if len(v) > 1}
    print(f"     mappingff atom TYPES that merge >1 distinct environment: {len(merged)}")
    for t, v in list(merged.items())[:6]:
        print(f"        type {t}: {sorted(v)}")

    # ---- bond environment: (elem pair, neighbour signatures) -> type ----
    def bond_env(src):
        typ, el, mol, bonds, angles, dihed = data[src]
        nper = len(mol)//100
        nbr_el = defaultdict(Counter)
        for b in bonds:
            x, y = int(b[2]), int(b[3])
            nbr_el[x][el[y-1]] += 1
            nbr_el[y][el[x-1]] += 1
        env2type = defaultdict(set)
        for b in bonds:
            x, y = int(b[2]), int(b[3])
            if x > nper: continue
            key = (tuple(sorted((el[x-1], el[y-1]))),
                   tuple(sorted(nbr_el[x].items())), tuple(sorted(nbr_el[y].items())))
            env2type[key].add(int(b[1]))
        return env2type
    for src in ("ligpargen", "mappingff"):
        e2t = bond_env(src)
        multi_e = {k: v for k, v in e2t.items() if len(v) > 1}
        merged = 0
        t2e = defaultdict(set)
        for k, v in e2t.items():
            for t in v: t2e[t].add(k)
        for t, v in t2e.items():
            if len(v) > 1: merged += 1
        print(f"  [{src}] distinct bond environments={len(e2t)}; "
              f"env split into >1 bond type={len(multi_e)}; "
              f"bond types merging >1 env={merged}")
    print()
