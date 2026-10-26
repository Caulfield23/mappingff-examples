"""Verifier part 7: readable connectivity dump for chain 1 (PMMA, PS) + ring detection."""
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
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
    return masses, atoms, bonds


for sysname in ("9_pmma", "9_ps", "9_pva"):
    masses, atoms, bonds = read_data(BASE / "ligpargen" / sysname / "filled.data")
    mt = {int(r[0]): float(r[1]) for r in masses}
    nper = len(atoms) // 100
    typ = np.array([int(a[2]) for a in atoms])
    mass = np.array([mt[int(t)] for t in typ])
    el = np.array([ELEM.get(round(m, 3), "?") for m in mass])
    adj = defaultdict(set)
    for b in bonds:
        a, c = int(b[2]), int(b[3])
        if a <= nper and c <= nper:
            adj[a].add(c); adj[c].add(a)
    # hydrogens per atom
    nh = Counter()
    for i in range(1, nper + 1):
        nh[i] = sum(1 for j in adj[i] if el[j-1] == "H")
    print("=" * 100)
    print(f"{sysname}: chain-1 heavy-atom connectivity (id: element, #H, heavy neighbours)")
    print("=" * 100)
    for i in range(1, nper + 1):
        if el[i-1] == "H":
            continue
        hn = sorted((j, el[j-1]) for j in adj[i] if el[j-1] != "H")
        print(f"  {i:4d} {el[i-1]} H{nh[i]}  ->  " +
              ", ".join(f"{j}({e})" for j, e in hn))
    print()

# PS ring detection: find 6-cycles of carbon
print("### PS ring count ###")
masses, atoms, bonds = read_data(BASE / "ligpargen" / "9_ps" / "filled.data")
mt = {int(r[0]): float(r[1]) for r in masses}
nper = len(atoms) // 100
typ = np.array([int(a[2]) for a in atoms])
mass = np.array([mt[int(t)] for t in typ])
el = np.array([ELEM.get(round(m, 3), "?") for m in mass])
adj = defaultdict(set)
for b in bonds:
    a, c = int(b[2]), int(b[3])
    if a <= nper and c <= nper:
        adj[a].add(c); adj[c].add(a)
heavy = [i for i in range(1, nper + 1) if el[i-1] == "C"]
hadj = {i: sorted(j for j in adj[i] if el[j-1] != "H") for i in heavy}
# count carbons with exactly 2 carbon heavy-nbrs that lie in a 6-cycle
cycles = []
seen = set()
for start in heavy:
    for nb in hadj[start]:
        path = [start, nb]
        prev = start
        cur = nb
        while len(path) < 7:
            nxt = [j for j in hadj[cur] if j != prev and j != path[0]]
            if not nxt:
                break
            nxt = nxt[0]
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == start:
                if len(path) == 6:
                    key = tuple(sorted(path))
                    if key not in seen:
                        seen.add(key); cycles.append(path)
                break
print(f"  6-membered carbon rings found: {len(cycles)}")
for c in cycles[:12]:
    print(f"    ring atoms: {c}")
