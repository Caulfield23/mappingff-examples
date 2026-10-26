"""Verifier part 5: oligomer chemistry from .mol files + backbone topology reconstruction."""
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
ELEM = {15.999: "O", 12.011: "C", 1.008: "H"}


def parse_mol(path):
    """Parse a V2000 .mol file: returns (elements[], bonds[(a,b)], charges[])."""
    lines = Path(path).read_text(errors="replace").splitlines()
    counts = lines[3]
    natoms = int(counts[0:3]); nbonds = int(counts[3:6])
    elems, xs = [], []
    for i in range(natoms):
        ln = lines[4 + i]
        x = float(ln[0:10]); y = float(ln[10:20]); z = float(ln[20:30])
        e = ln[31:34].strip()
        elems.append(e)
        xs.append((x, y, z))
    bonds = []
    off = 4 + natoms
    for i in range(nbonds):
        ln = lines[off + i]
        a = int(ln[0:3]); b = int(ln[3:6]); t = int(ln[6:9])
        bonds.append((a, b, t))
    return elems, bonds, np.array(xs)


def parse_data_bonds(path):
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
    grab("Masses")
    atoms = grab("Atoms")
    bonds = grab("Bonds")
    return atoms, bonds


for sysname in SYSTEMS:
    print("=" * 100)
    print(f"SYSTEM {sysname}")
    print("=" * 100)
    molp = BASE / "ligpargen" / sysname / "build" / f"{sysname}.mol"
    elems, molbonds, xs = parse_mol(molp)
    print(f"  .mol: {sysname}.mol  natoms={len(elems)}  nbonds={len(molbonds)}")
    print(f"  element counts: {dict(Counter(elems))}")
    # bond orders present
    print(f"  bond order histogram: {dict(Counter(b[2] for b in molbonds))}")
    print(f"  .mol charges (col 9): "
          f"{[lines for lines in []] or ''}")
    # charges from mol file
    lines = Path(molp).read_text(errors="replace").splitlines()
    natoms = int(lines[3][0:3])
    q = []
    for i in range(natoms):
        ln = lines[4 + i]
        try:
            q.append(float(ln[36:45]))
        except ValueError:
            q.append(float("nan"))
    q = np.array(q)
    print(f"  .mol total charge = {np.nansum(q):.6f}")

    # reconstruct graph from the RUN's filled.data (authoritative)
    atoms, bonds = parse_data_bonds(BASE / "ligpargen" / sysname / "filled.data")
    mt = {}
    # rebuild masses
    with open(BASE / "ligpargen" / sysname / "filled.data", "r", errors="replace") as fh:
        txt = fh.read().splitlines()
    ii = 0
    while ii < len(txt) and not txt[ii].strip().startswith("Masses"):
        ii += 1
    ii += 1
    while ii < len(txt) and not txt[ii].strip():
        ii += 1
    while ii < len(txt) and txt[ii].strip():
        p = txt[ii].split(); mt[int(p[0])] = float(p[1]); ii += 1
    nat = len(atoms)
    typ = np.array([int(a[2]) for a in atoms])
    mol = np.array([int(a[1]) for a in atoms])
    mass = np.array([mt[int(t)] for t in typ])
    el = np.array([ELEM.get(round(m, 3), "?") for m in mass])
    # chain 1 = atoms 1..nat/100
    nper = nat // 100
    sel = np.arange(nper)
    # bonds within chain 1
    adj = defaultdict(set)
    for b in bonds:
        a, c = int(b[2]), int(b[3])
        if a <= nper and c <= nper:
            adj[a].add(c); adj[c].add(a)
    heavy_deg = Counter()
    for i in range(1, nper + 1):
        d = sum(1 for j in adj[i] if el[j-1] != "H")
        heavy_deg[d] += 1
    print(f"  chain-1 heavy-atom degree histogram: {dict(sorted(heavy_deg.items()))}")
    # backbone path: start from a heavy atom with 1 heavy neighbour
    heavy = [i for i in range(1, nper + 1) if el[i-1] != "H"]
    ends = [i for i in heavy if len([j for j in adj[i] if el[j-1] != "H"]) == 1]
    print(f"  heavy atoms per chain: {len(heavy)}  chain ends (1 heavy nbr): {ends}")
    # walk backbone
    if len(ends) >= 2:
        path = [ends[0]]
        prev = None
        cur = ends[0]
        while True:
            nxts = [j for j in adj[cur] if el[j-1] != "H" and j != prev]
            if not nxts:
                break
            nxt = nxts[0]
            path.append(nxt)
            prev, cur = cur, nxt
        print(f"  backbone heavy-atom path length: {len(path)} atoms "
              f"(linear chain -> len == n_heavy)")
        seq = "".join(el[i-1] for i in path)
        print(f"  backbone element sequence: {seq}")
        # which atoms are off-backbone substituents
        branches = [i for i in heavy if i not in path]
        print(f"  heavy atoms NOT on backbone: {len(branches)}")
        # end groups: element + neighbours of the two end atoms
        for e in ends:
            nb = sorted((el[j-1], j) for j in adj[e])
            print(f"    end atom {e} ({el[e-1]}): neighbours {nb}")
    # formula
    c = Counter(el[:nper])
    print(f"  chain-1 formula: C{c['C']} H{c['H']} O{c['O']}   (MW={mass[:nper].sum():.3f})")
    print()


print("### mappingff build-dir filled.data vs RUN filled.data ###")
for sysname in SYSTEMS:
    p_run = BASE / "mappingff" / sysname / "filled.data"
    p_bld = BASE / "mappingff" / sysname / "build" / "filled.data"
    if not p_bld.exists():
        print(f"  {sysname}: no build/filled.data")
        continue
    a1, b1 = parse_data_bonds(p_run)
    a2, b2 = parse_data_bonds(p_bld)
    xa = np.array([[a[4], a[5], a[6]] for a in a1])
    xb = np.array([[a[4], a[5], a[6]] for a in a2])
    ta = np.array([int(a[2]) for a in a1]); tb = np.array([int(a[2]) for a in a2])
    same_n = len(a1) == len(a2)
    dev = np.abs(xa - xb).max() if same_n else float("nan")
    print(f"  {sysname}: run natoms={len(a1)} build natoms={len(a2)} | "
          f"types identical={np.array_equal(ta, tb) if same_n else 'n/a'} | "
          f"max|dx|={dev:.6f} A | "
          f"mtime run={p_run.stat().st_mtime:.0f} build={p_bld.stat().st_mtime:.0f}")
