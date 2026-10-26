"""Element composition per chain + physical-density plausibility context."""
import re
from pathlib import Path

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
ELEM = {15.999: "O", 12.011: "C", 1.008: "H", 14.007: "N", 32.06: "S", 35.45: "Cl"}


def parse(path):
    lines = path.read_text(errors="replace").splitlines()
    natoms = None
    mass_of_type = {}
    types = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if natoms is None:
            m = re.match(r"^(\d+)\s+atoms\b", s)
            if m:
                natoms = int(m.group(1))
        if s.startswith("Masses"):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip():
                p = lines[i].split()
                mass_of_type[int(p[0])] = float(p[1])
                i += 1
            continue
        if s.startswith("Atoms"):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip():
                types.append(int(lines[i].split()[2]))
                i += 1
            continue
        i += 1
    return natoms, mass_of_type, types


print(f"{'run':22s} {'natoms':>7s} {'nchain':>6s} {'per-chain formula':>24s} {'M_chain':>9s}")
print("-" * 78)
for sysname in ("9_peo", "9_pmma", "9_ps", "9_pva"):
    for src in ("ligpargen", "mappingff"):
        natoms, mot, types = parse(BASE / src / sysname / "filled.data")
        nchain = 100
        cnt = {}
        mt = 0.0
        for t in types:
            m = mot[t]
            e = ELEM.get(round(m, 3), f"?{m}")
            cnt[e] = cnt.get(e, 0) + 1
            mt += m
        per = {e: c / nchain for e, c in sorted(cnt.items())}
        formula = "".join(f"{e}{v:g}" for e, v in per.items())
        print(f"{src + '/' + sysname:22s} {natoms:7d} {nchain:6d} {formula:>24s} "
              f"{mt / nchain:9.3f}")
