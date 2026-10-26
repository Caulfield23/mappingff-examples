"""Backbone skeleton walk: longest heavy-atom path per chain + contour length."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
RUNS = [(src, s) for s in ("9_peo", "9_pmma", "9_ps", "9_pva")
        for src in ("ligpargen", "mappingff")]


def parse(path):
    lines = path.read_text(errors="replace").splitlines()
    sec = {}
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        m = re.match(r"^(?:(\d+)\s+)?([A-Za-z][A-Za-z ]*?)\s*(#.*)?$", line)
        if m and m.group(2).strip() in ("Masses", "Atoms", "Bonds"):
            s = m.group(2).strip()
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            rows = []
            k = j
            while k < n and lines[k].strip():
                rows.append(lines[k].split())
                k += 1
            sec[s] = rows
            i = k
            continue
        i += 1
    return sec


for src, sysname in RUNS:
    d = parse(BASE / src / sysname / "filled.data")
    mot = {int(r[0]): float(r[1]) for r in d["Masses"]}
    at = d["Atoms"]
    aid = np.array([int(r[0]) for r in at])
    mol = np.array([int(r[1]) for r in at])
    typ = np.array([int(r[2]) for r in at])
    o = np.argsort(aid)
    aid, mol, typ = aid[o], mol[o], typ[o]
    mass = np.array([mot[t] for t in typ])
    n = len(aid)
    b1 = np.array([int(r[2]) for r in d["Bonds"]]) - 1
    b2 = np.array([int(r[3]) for r in d["Bonds"]]) - 1

    adj = [[] for _ in range(n)]
    for a, b in zip(b1, b2):
        adj[a].append(b)
        adj[b].append(a)

    m0 = 1
    idx = np.where(mol == m0)[0]
    heavy = mass >= 12.0
    # heavy subgraph of chain 1
    hadj = {i: [j for j in adj[i] if heavy[j]] for i in idx if heavy[i]}
    deg = {i: len(v) for i, v in hadj.items()}
    ends = [i for i in hadj if deg[i] <= 1]
    # walk from an end
    start = ends[0] if ends else sorted(hadj)[0]
    path = [start]
    prev = -1
    cur = start
    while True:
        nxt = [j for j in hadj[cur] if j != prev]
        # prefer continuing along degree<=2 nodes
        nxt2 = [j for j in nxt if deg[j] <= 2] or nxt
        if not nxt2:
            break
        prev, cur = cur, nxt2[0]
        path.append(cur)
        if len(path) > 500:
            break
    # for ring-containing backbones (PS) the walk stops at the ring; handle by
    # reporting the ring separately
    nheavy_chain = sum(1 for i in idx if heavy[i])
    print(f"[{src}/{sysname}] chain1 heavy={nheavy_chain} "
          f"heavy-graph ends(deg<=1)={len(ends)} walk_len={len(path)} "
          f"walk_bonds={len(path)-1}")
    print("   heavy-atom path (atom ids):",
          " ".join(str(aid[p]) for p in path[:40]), "..." if len(path) > 40 else "")
    # element labels
    el = ["C" if abs(mass[i] - 12.011) < 1e-3 else
          ("O" if abs(mass[i] - 15.999) < 1e-3 else "H") for i in idx]
    print("   elements of chain1 (atom-id order):",
          "".join(el[i] for i in range(len(idx)))[:200])
    print()
