"""
Independent topology / connectivity audit of the 8 melt run directories.

Does NOT reuse analyze_density_rg.py: re-implements the LAMMPS data-file
parser from scratch so that a bug in the original parser cannot hide.

Checks
  1. atom/molecule counts, atoms per molecule
  2. bond graph -> connected components; is every molecule id ONE component?
     is every component exactly one molecule id?  component size == expected?
  3. backbone identification (for the Rg theory check): degree-2 carbon/oxygen
     skeleton of each chain
  4. mass census (heavy vs H)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
RUNS = [(src, sysname) for sysname in ("9_peo", "9_pmma", "9_ps", "9_pva")
        for src in ("ligpargen", "mappingff")]


def parse_data(path: Path):
    """Minimal but strict LAMMPS data-file parser."""
    text = path.read_text(errors="replace").splitlines()
    out = {"header": {}}
    i = 0
    n = len(text)
    natoms = None
    while i < n:
        line = text[i].strip()
        if not line:
            i += 1
            continue
        m = re.match(r"^(\d+)\s+atoms\b", line)
        if m:
            natoms = int(m.group(1))
        # section header: "<count> <section>" possibly followed by " # comment"
        m = re.match(r"^(?:(\d+)\s+)?([A-Za-z][A-Za-z ]*?)\s*(#.*)?$", line)
        if m and m.group(2).strip() in ("Masses", "Atoms", "Bonds", "Angles",
                                        "Dihedrals", "Impropers", "Velocities",
                                        "Pair Coeffs", "PairIJ Coeffs",
                                        "Bond Coeffs", "Angle Coeffs",
                                        "Dihedral Coeffs", "Atom Type Labels"):
            section = m.group(2).strip()
            count = int(m.group(1)) if m.group(1) else None
            j = i + 1
            while j < n and not text[j].strip():
                j += 1
            rows = []
            k = j
            while k < n and text[k].strip():
                rows.append(text[k].split())
                k += 1
            if count is not None and len(rows) != count and section not in ("Atom Type Labels",):
                raise RuntimeError(f"{path}: section {section} expected {count} "
                                   f"rows, found {len(rows)}")
            out[section] = rows
            i = k
            continue
        i += 1

    # box
    bb = []
    for line in text[:40]:
        mm = re.match(r"^(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+xlo\s+xhi", line)
        if mm:
            bb.append((float(mm.group(1)), float(mm.group(2))))
    out["box"] = bb
    out["natoms"] = natoms
    return out


def main():
    for src, sysname in RUNS:
        run = BASE / src / sysname
        d = parse_data(run / "filled.data")
        natoms = d["natoms"]
        mass_of_type = {int(r[0]): float(r[1]) for r in d["Masses"]}

        atoms = d["Atoms"]
        aid = np.array([int(r[0]) for r in atoms])
        mol = np.array([int(r[1]) for r in atoms])
        typ = np.array([int(r[2]) for r in atoms])
        order = np.argsort(aid)
        aid, mol, typ = aid[order], mol[order], typ[order]
        assert np.array_equal(aid, np.arange(1, natoms + 1)), "atom ids not 1..N"
        mass = np.array([mass_of_type[t] for t in typ])

        bonds = d["Bonds"]
        b1 = np.array([int(r[2]) for r in bonds])
        b2 = np.array([int(r[3]) for r in bonds])

        # --- union-find over bonds -------------------------------------- #
        parent = np.arange(natoms)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for a, b in zip(b1, b2):
            union(a - 1, b - 1)

        comp = np.array([find(i) for i in range(natoms)])
        ncomp = len(np.unique(comp))
        comp_ids = np.unique(comp)
        comp_size = np.array([np.sum(comp == c) for c in comp_ids])

        mol_ids = np.unique(mol)
        # map molecule -> set of components
        mol_to_comps = {}
        for c, size in zip(comp_ids, comp_size):
            mols_in = np.unique(mol[comp == c])
            assert len(mols_in) == 1, (
                f"{src}/{sysname}: component {c} contains molecule ids "
                f"{mols_in.tolist()} -> chains are SPLIT/merged!")
            mol_to_comps.setdefault(int(mols_in[0]), []).append(int(c))

        n_comp_per_mol = [len(v) for v in mol_to_comps.values()]
        sizes_per_mol = {m: int(np.sum(mol == m)) for m in mol_ids}

        # degree of each atom inside the bond graph (heavy atoms only)
        deg = np.zeros(natoms, dtype=int)
        for a, b in zip(b1, b2):
            deg[a - 1] += 1
            deg[b - 1] += 1

        heavy = mass >= 12.0
        nH = int((~heavy).sum())
        nheavy = int(heavy.sum())

        # ---- backbone skeleton of chain 1: heavy atoms with degree<=2 ---- #
        # (side-group atoms are attached to the backbone and have deg 1-3+)
        m1 = np.where(mol == mol_ids[0])[0]
        deg1 = deg[m1]
        heavy1 = heavy[m1]
        # A backbone atom has >=2 heavy-atom neighbours within the chain.
        hnbr = np.zeros(natoms, dtype=int)
        for a, b in zip(b1, b2):
            if mass[a - 1] >= 12 and mass[b - 1] >= 12:
                hnbr[a - 1] += 1
                hnbr[b - 1] += 1
        bb_mask = heavy & (hnbr >= 2)
        n_bb_chain1 = int(bb_mask[m1].sum())
        # longest heavy-atom path length in chain 1 via BFS on the heavy graph
        adj = {}
        for a, b in zip(b1, b2):
            if mass[a - 1] >= 12 and mass[b - 1] >= 12:
                adj.setdefault(a - 1, []).append(b - 1)
                adj.setdefault(b - 1, []).append(a - 1)
        # BFS from an end of the heavy skeleton
        start = min(m1[bb_mask[m1]])
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        # distance from start to the farthest heavy atom of chain 1
        dist = {start: 0}
        q = [start]
        while q:
            u = q.pop(0)
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        max_bb_hops = max(dist.values())

        print(f"[{src}/{sysname}]")
        print(f"   natoms={natoms} n_mol_ids={len(mol_ids)} "
              f"atoms/mol={sorted(set(sizes_per_mol.values()))} "
              f"n_bonds={len(bonds)}")
        print(f"   components={ncomp}  comps-per-mol: "
              f"min={min(n_comp_per_mol)} max={max(n_comp_per_mol)} "
              f"-> {'OK (1 per chain)' if max(n_comp_per_mol)==1 else 'SPLIT CHAINS!'}")
        print(f"   comp sizes (unique)={sorted(set(comp_size.tolist()))}")
        print(f"   heavy={nheavy}  H={nH}  "
              f"masses(unique)={sorted(set(np.round(mass,4).tolist()))}")
        print(f"   chain1: heavy atoms with >=2 heavy nbrs (backbone-ish)="
              f"{n_bb_chain1}   max heavy-atom hop distance={max_bb_hops}")
        # end atoms (deg 1) of the heavy skeleton -> 2 chain ends expected
        ends = [a for a in m1 if hnbr[a] <= 1]
        print(f"   chain1 heavy-skeleton end atoms (deg<=1): {len(ends)}")
        print()


if __name__ == "__main__":
    main()
