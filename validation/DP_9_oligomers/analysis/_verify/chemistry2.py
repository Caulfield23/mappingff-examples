"""Verifier part 6: definitive backbone extraction + repeat-unit counting."""
import re
from pathlib import Path
from collections import Counter, defaultdict
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
    angles = grab("Angles")
    dihed = grab("Dihedrals")
    imp = grab("Impropers")
    return masses, atoms, bonds, angles, dihed, imp


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def ring_components(nodes, adj):
    """2-core leaf stripping -> connected components = ring systems (pendant rings)."""
    deg = {n: len(adj[n]) for n in nodes}
    alive = set(nodes)
    changed = True
    while changed:
        changed = False
        for n in list(alive):
            if deg[n] <= 1:
                alive.discard(n)
                changed = True
                for m in adj[n]:
                    if m in alive:
                        deg[m] -= 1
    groups = defaultdict(list)
    seen = set()
    for n in sorted(alive):
        if n in seen:
            continue
        comp = []
        stack = [n]; seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y in alive and y not in seen:
                    seen.add(y); stack.append(y)
        groups[min(comp)] = sorted(comp)
    # every non-core atom is its own component
    for n in nodes:
        if n not in alive:
            groups[n] = [n]
    return groups


for sysname in SYSTEMS:
    print("=" * 100)
    print(f"SYSTEM {sysname}")
    print("=" * 100)
    masses, atoms, bonds, angles, dihed, imp = read_data(
        BASE / "ligpargen" / sysname / "filled.data")
    mt = {int(r[0]): float(r[1]) for r in masses}
    nat = len(atoms)
    nper = nat // 100
    typ = np.array([int(a[2]) for a in atoms])
    mass = np.array([mt[int(t)] for t in typ])
    el = np.array([ELEM.get(round(m, 3), "?") for m in mass])

    adj = defaultdict(set)
    for b in bonds:
        a, c = int(b[2]), int(b[3])
        if a <= nper and c <= nper:
            adj[a].add(c); adj[c].add(a)
    heavy = sorted(i for i in range(1, nper + 1) if el[i - 1] != "H")
    hadj = {i: sorted(j for j in adj[i] if el[j - 1] != "H") for i in heavy}
    groups = ring_components(heavy, {i: list(hadj[i]) for i in heavy})
    rings = {k: v for k, v in groups.items() if len(v) > 1}
    # quotient tree
    comp_of = {}
    for k, v in groups.items():
        for n in v:
            comp_of[n] = k
    qadj = defaultdict(set)
    for i in heavy:
        for j in hadj[i]:
            if comp_of[i] != comp_of[j]:
                qadj[comp_of[i]].add(comp_of[j])
                qadj[comp_of[j]].add(comp_of[i])
    # BFS diameter
    def bfs(src):
        dist = {src: 0}; par = {src: None}; q = [src]
        while q:
            nq = []
            for x in q:
                for y in qadj[x]:
                    if y not in dist:
                        dist[y] = dist[x] + 1; par[y] = x; nq.append(y)
            q = nq
        far = max(dist, key=lambda k: dist[k])
        return far, dist, par
    s = next(iter(groups))
    e, dist, par = bfs(s)
    e2, dist2, par2 = bfs(e)
    path_comps = []
    c = e2
    while c is not None:
        path_comps.append(c); c = par2[c]
    path_comps.reverse()
    backbone = []
    for c in path_comps:
        backbone.extend(sorted(groups[c]))
    print(f"  chain-1 formula: C{Counter(el[:nper])['C']} H{Counter(el[:nper])['H']} "
          f"O{Counter(el[:nper])['O']}   MW={mass[:nper].sum():.3f}  "
          f"(x100 chains -> system mass {mass.sum():.1f} amu)")
    print(f"  heavy atoms/chain = {len(heavy)};  ring systems found = {len(rings)} "
          f"(sizes {sorted(len(v) for v in rings.values())})")
    print(f"  BACKBONE path: {len(backbone)} heavy atoms; "
          f"element sequence = {''.join(el[i-1] for i in backbone)}")
    print(f"  backbone atoms not on the longest path: "
          f"{len(set(heavy) - set(backbone))} (side groups / ring atoms)")

    # ---- motif counting ----
    if sysname == "9_peo":
        ether = sum(1 for i in heavy if el[i-1] == "O" and len(hadj[i]) == 2)
        oh = sum(1 for i in heavy if el[i-1] == "O" and len(hadj[i]) == 1)
        print(f"  ether O (C-O-C, 2 heavy nbrs) = {ether};  hydroxyl O = {oh}")
        print(f"  => HO-(CH2CH2O)9-H : {ether} ether O + 2 OH; "
              f"EO repeat units = {ether+1}")
    if sysname == "9_pmma":
        ester = 0; methoxy = 0
        for i in heavy:
            if el[i-1] == "C" and len(hadj[i]) == 3:
                nb = sorted(el[j-1] for j in hadj[i])
                if nb == ["C", "O", "O"]:
                    ester += 1
        # side group: O bonded to a C that has only heavy nbr O
        for i in heavy:
            if el[i-1] == "O" and len(hadj[i]) == 1:
                c = hadj[i][0]
                if el[c-1] == "C" and all(el[j-1] == "H" for j in adj[c] if el[j-1] != "H"):
                    pass
        # find the O-alkyl: O with 1 heavy nbr (the C) whose C has 1 heavy nbr
        alkoxy = []
        for i in heavy:
            if el[i-1] == "O" and len(hadj[i]) == 1:
                c = hadj[i][0]
                if el[c-1] == "C":
                    alkyl = [j for j in hadj[c] if el[j-1] != "H"]
                    alkoxy.append((i, c, len(alkyl)))
        print(f"  ester groups C(=O)(O) with 3 heavy nbrs = {ester}")
        print(f"  ester O-CHx groups: {len(alkoxy)}; alkyl heavy-atom counts "
              f"= {Counter(a[2] for a in alkoxy)}")
        # alpha methyls: C with 3 heavy nbrs where nbrs are C,C,C and one is the ester C
        alpha = 0
        for i in heavy:
            if el[i-1] == "C" and len(hadj[i]) == 3:
                nb = sorted(el[j-1] for j in hadj[i])
                if nb == ["C", "C", "C"]:
                    alpha += 1
        print(f"  quaternary-ish C with 3 C heavy-nbrs (C-alpha) = {alpha}")
    if sysname == "9_ps":
        print(f"  phenyl rings = {len(rings)} (size {sorted(set(len(v) for v in rings.values()))})")
        # ipso carbons
        ipso = 0
        for k, v in rings.items():
            for i in v:
                if len(hadj[i]) == 3: ipso += 1
        print(f"  ipso carbons (3 heavy nbrs) = {ipso}")
    if sysname == "9_pva":
        oh = sum(1 for i in heavy if el[i-1] == "O" and len(hadj[i]) == 1)
        print(f"  hydroxyl O = {oh}  => {oh} vinyl-alcohol repeat units")
        # CH2 in backbone
        ch2 = sum(1 for i in backbone if el[i-1] == "C" and len(hadj[i]) == 2)
        print(f"  backbone CH2 carbons = {ch2}")
    # end groups
    ends = [i for i in heavy if len(hadj[i]) == 1]
    print(f"  heavy atoms with exactly 1 heavy neighbour (= candidate termini): {len(ends)}")
    for e_ in ends:
        nb = sorted((el[j-1], j) for j in adj[e_])
        print(f"     atom {e_} ({el[e_-1]}): bonded to {nb}")
    print()


# impropers in templates
print("### Impropers present in the single-chain TEMPLATES (dropped by fill_box.py?) ###")
for sysname in SYSTEMS:
    for src, fn in (("ligpargen", f"{sysname}.lmp"), ("mappingff", f"{sysname}.data")):
        p = BASE / src / sysname / "build" / fn
        if not p.exists():
            print(f"  {src}/{sysname}: {fn} MISSING")
            continue
        txt = p.read_text(errors="replace")
        m = re.search(r"^(\d+)\s+impropers\s*$", txt, re.M)
        has = "Impropers" in txt
        print(f"  {src}/{sysname} {fn}: impropers-count-line="
              f"{m.group(1) if m else '-'}  'Impropers' section present={has}")
