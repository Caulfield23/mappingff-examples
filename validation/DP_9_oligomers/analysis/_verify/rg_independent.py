"""
Independent, from-scratch re-implementation of the per-chain radius of
gyration (Rg) analysis, written to cross-check
`analysis/analyze_density_rg.py`.

Deliberate differences from the script under test:
  * NOTHING is imported from analyze_density_rg.py.
  * The LAMMPS trajectory is read with a plain line-by-line iterator
    (`for line in fh` + itertools.islice), NOT with mmap / bytes.find
    frame scanning.
  * The topology is parsed with my own section scanner.
  * The Rg itself is computed with a PLAIN PYTHON LOOP over atoms
    (no numpy bincount, no einsum), using the UNWRAPPED xu/yu/zu.

Rg definition used (identical to the one under test):
    Rg_c^2 = sum_{i in chain c} m_i |r_i - r_com,c|^2 / sum_{i in chain c} m_i
where r_com,c is the MASS-weighted centre of mass of chain c.

Per-frame statistics:
    frame_mean = mean over the 100 chains of Rg_c
    frame_std  = sample std (ddof = 1) over the 100 chains of Rg_c
Per-run statistics (as written to density_rg_per_run.csv):
    rg_mean = mean over frames of frame_mean
    rg_std  = mean over frames of frame_std
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import time
from collections import deque
from itertools import islice
from pathlib import Path

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
OUT = Path(r"C:\schoolwork\Projects\mappingff\validation\longer\analysis\_verify")

SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")

STEP_LO, STEP_HI = 1_500_000, 2_000_000

PER_RUN_CSV = BASE / "analysis" / "density_rg_per_run.csv"


# --------------------------------------------------------------------------- #
# Topology (my own parser)
# --------------------------------------------------------------------------- #
def parse_topology(path: Path):
    """Return (natoms, atom_type[1..N], atom_mol[1..N], mass_of_type{})."""
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    natoms = None
    for ln in lines[:80]:
        m = re.match(r"^\s*(\d+)\s+atoms\b", ln)
        if m:
            natoms = int(m.group(1))
            break
    if natoms is None:
        raise RuntimeError(f"{path}: no 'N atoms' header found")

    mass_of_type: dict[int, float] = {}
    atom_type: list[int] = [0] * (natoms + 1)
    atom_mol: list[int] = [0] * (natoms + 1)

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        tok = s.split()[0].lower()

        if tok in ("masses", "atoms"):
            # skip the blank line LAMMPS puts after the section header
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if tok == "masses":
                while j < n and lines[j].strip():
                    p = lines[j].split()
                    mass_of_type[int(p[0])] = float(p[1])
                    j += 1
            else:
                while j < n and lines[j].strip():
                    p = lines[j].split()
                    aid, mol, typ = int(p[0]), int(p[1]), int(p[2])
                    if aid < 1 or aid > natoms:
                        raise RuntimeError(f"{path}: atom id {aid} out of range")
                    atom_type[aid] = typ
                    atom_mol[aid] = mol
                    j += 1
            i = j
            continue

        i += 1

    if not mass_of_type:
        raise RuntimeError(f"{path}: no Masses section")
    if 0 in atom_type[1:]:
        raise RuntimeError(f"{path}: incomplete Atoms section")

    return natoms, atom_type, atom_mol, mass_of_type


# --------------------------------------------------------------------------- #
# Trajectory reader + Rg  (plain python loops)
# --------------------------------------------------------------------------- #
def scan_rg(traj: Path, natoms, atom_type, atom_mol, mass_of_type,
            step_lo, step_hi, verbose=True):
    mass_of_atom = [0.0] * (natoms + 1)
    for aid in range(1, natoms + 1):
        t = atom_type[aid]
        if t not in mass_of_type:
            raise RuntimeError(f"{traj.name}: no mass for atom type {t}")
        mass_of_atom[aid] = mass_of_type[t]

    # chain (molecule) index of each atom, 0-based, ordered by molecule id
    mol_ids = sorted(set(atom_mol[1:]))
    chain_of_atom = [0] * (natoms + 1)
    mol2chain = {m: k for k, m in enumerate(mol_ids)}
    for aid in range(1, natoms + 1):
        chain_of_atom[aid] = mol2chain[atom_mol[aid]]
    n_chain = len(mol_ids)

    steps: list[int] = []
    f_mean: list[float] = []
    f_std: list[float] = []
    f_min: list[float] = []
    f_max: list[float] = []
    f_mean_wrapped: list[float] = []     # control: same calc using wrapped x,y,z
    n_frames_total = 0
    n_id_mismatch = 0
    n_mol_mismatch = 0
    n_type_mismatch = 0

    t0 = time.perf_counter()
    with open(traj, "r", errors="replace") as fh:
        lines = iter(fh)
        for line in lines:
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            n_frames_total += 1
            step = int(next(lines))
            next(lines)                       # ITEM: NUMBER OF ATOMS
            na = int(next(lines))
            next(lines)                       # ITEM: BOX BOUNDS ...
            next(lines)                       # xlo xhi
            next(lines)                       # ylo yhi
            next(lines)                       # zlo zhi
            hdr = next(lines)                 # ITEM: ATOMS ...
            cols = hdr.split()[2:]
            i_id = cols.index("id")
            i_mol = cols.index("mol")
            i_typ = cols.index("type")
            i_x = cols.index("x")
            i_y = cols.index("y")
            i_z = cols.index("z")
            i_xu = cols.index("xu")
            i_yu = cols.index("yu")
            i_zu = cols.index("zu")

            if na != natoms:
                raise RuntimeError(f"{traj.name}: frame natoms {na} != {natoms}")

            if not (step_lo <= step <= step_hi):
                # skip the atom block of out-of-window frames without parsing
                deque(islice(lines, na), maxlen=0)
                continue

            # ---------------- pass 1: parse + accumulate ------------------ #
            chains: list[list] = [[] for _ in range(n_chain)]
            sm = [0.0] * n_chain
            smx = [0.0] * n_chain
            smy = [0.0] * n_chain
            smz = [0.0] * n_chain

            for ln in islice(lines, na):
                p = ln.split()
                aid = int(p[i_id])
                mol = int(p[i_mol])
                typ = int(p[i_typ])
                if aid < 1 or aid > natoms:
                    raise RuntimeError(f"{traj.name}: atom id {aid} out of range")
                if mol != atom_mol[aid]:
                    n_mol_mismatch += 1
                if typ != atom_type[aid]:
                    n_type_mismatch += 1

                c = chain_of_atom[aid]
                m = mass_of_atom[aid]
                x = float(p[i_xu])
                y = float(p[i_yu])
                z = float(p[i_zu])
                chains[c].append((m, x, y, z,
                                  float(p[i_x]), float(p[i_y]), float(p[i_z])))
                sm[c] += m
                smx[c] += m * x
                smy[c] += m * y
                smz[c] += m * z

            # ---------------- centre of mass ---------------------------- #
            com = []
            for c in range(n_chain):
                if sm[c] <= 0.0:
                    raise RuntimeError(f"{traj.name}: chain {c} has no atoms")
                com.append((smx[c] / sm[c], smy[c] / sm[c], smz[c] / sm[c]))

            # ---------------- pass 2: plain loop for Rg^2 --------------- #
            rg = [0.0] * n_chain
            rg_wrapped = [0.0] * n_chain
            for c in range(n_chain):
                cx, cy, cz = com[c]
                tot = 0.0
                totw = 0.0
                for (m, x, y, z, wx, wy, wz) in chains[c]:
                    dx = x - cx
                    dy = y - cy
                    dz = z - cz
                    tot += m * (dx * dx + dy * dy + dz * dz)
                    dxw = wx - cx
                    dyw = wy - cy
                    dzw = wz - cz
                    totw += m * (dxw * dxw + dyw * dyw + dzw * dzw)
                rg[c] = math.sqrt(tot / sm[c])
                rg_wrapped[c] = math.sqrt(totw / sm[c])

            # ---------------- frame statistics -------------------------- #
            n = len(rg)
            mean = 0.0
            for v in rg:
                mean += v
            mean /= n
            var = 0.0
            for v in rg:
                var += (v - mean) * (v - mean)
            std = math.sqrt(var / (n - 1))          # ddof = 1, like numpy
            meanw = 0.0
            for v in rg_wrapped:
                meanw += v
            meanw /= n

            steps.append(step)
            f_mean.append(mean)
            f_std.append(std)
            f_min.append(min(rg))
            f_max.append(max(rg))
            f_mean_wrapped.append(meanw)

    if verbose:
        print(f"    frames total={n_frames_total}  selected={len(steps)}"
              f"  steps {min(steps) if steps else '-'}..{max(steps) if steps else '-'}"
              f"  mol_mismatch={n_mol_mismatch} type_mismatch={n_type_mismatch}"
              f"  ({time.perf_counter() - t0:.1f} s)")

    # per-run aggregates, matching the column definitions of the script
    def avg(v):
        if not v:
            return float("nan")
        s = 0.0
        for x in v:
            s += x
        return s / len(v)

    return {
        "n_frames_total": n_frames_total,
        "steps": steps,
        "n_chain": n_chain,
        "rg_mean": avg(f_mean),
        "rg_std": avg(f_std),
        "rg_min": min(f_min) if f_min else float("nan"),
        "rg_max": max(f_max) if f_max else float("nan"),
        "rg_mean_wrapped": avg(f_mean_wrapped),
        "mol_mismatch": n_mol_mismatch,
        "type_mismatch": n_type_mismatch,
        "per_frame_mean": f_mean,
        "per_frame_std": f_std,
    }


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    ref = {}
    if PER_RUN_CSV.exists():
        with open(PER_RUN_CSV, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ref[(row["system"], row["source"])] = row

    results = []
    for sysname in SYSTEMS:
        for src in SOURCES:
            run = BASE / src / sysname
            print(f"[{src}/{sysname}]")
            natoms, atom_type, atom_mol, mass_of_type = \
                parse_topology(run / "filled.data")
            r = scan_rg(run / "dump.lammpstrj", natoms, atom_type, atom_mol,
                        mass_of_type, STEP_LO, STEP_HI)
            r["system"] = sysname
            r["source"] = src
            r["natoms"] = natoms
            results.append(r)

    # ---------------- comparison table ---------------------------------- #
    print()
    hdr = (f"{'system':>8} {'source':>9} {'nframes':>7} "
           f"{'rg_mean(mine)':>16} {'rg_mean(csv)':>16} {'absdiff':>12} {'reldiff':>12} "
           f"{'rg_std(mine)':>15} {'rg_std(csv)':>15} {'absdiff':>12} {'reldiff':>12}")
    print(hdr)
    print("-" * len(hdr))

    worst_rel_mean = 0.0
    worst_rel_std = 0.0
    rows_out = []
    for r in results:
        key = (r["system"], r["source"])
        row = ref.get(key)
        if row is None:
            print(f"{r['system']:>8} {r['source']:>9}  NO REFERENCE ROW")
            continue
        csv_mean = float(row["rg_mean"])
        csv_std = float(row["rg_std"])
        d_mean = r["rg_mean"] - csv_mean
        rel_mean = abs(d_mean) / abs(csv_mean) if csv_mean else float("nan")
        d_std = r["rg_std"] - csv_std
        rel_std = abs(d_std) / abs(csv_std) if csv_std else float("nan")
        worst_rel_mean = max(worst_rel_mean, rel_mean)
        worst_rel_std = max(worst_rel_std, rel_std)
        print(f"{r['system']:>8} {r['source']:>9} {len(r['steps']):>7} "
              f"{r['rg_mean']:16.12f} {csv_mean:16.12f} {d_mean:12.3e} {rel_mean:12.3e} "
              f"{r['rg_std']:15.12f} {csv_std:15.12f} {d_std:12.3e} {rel_std:12.3e}")
        rows_out.append({
            "system": r["system"], "source": r["source"],
            "nframes": len(r["steps"]),
            "steps_min": min(r["steps"]), "steps_max": max(r["steps"]),
            "n_chain": r["n_chain"], "natoms": r["natoms"],
            "rg_mean_mine": repr(r["rg_mean"]), "rg_mean_csv": repr(csv_mean),
            "abs_diff_mean": repr(d_mean), "rel_diff_mean": repr(rel_mean),
            "rg_std_mine": repr(r["rg_std"]), "rg_std_csv": repr(csv_std),
            "abs_diff_std": repr(d_std), "rel_diff_std": repr(rel_std),
            "rg_min_mine": repr(r["rg_min"]), "rg_max_mine": repr(r["rg_max"]),
            "rg_min_csv": repr(float(row["rg_min"])),
            "rg_max_csv": repr(float(row["rg_max"])),
            "rg_mean_wrapped_mine": repr(r["rg_mean_wrapped"]),
            "mol_mismatch": r["mol_mismatch"], "type_mismatch": r["type_mismatch"],
            "frames_total": r["n_frames_total"],
            "frames_analyzed_csv": row["frames_analyzed"],
        })

    print()
    print(f"worst relative difference, rg_mean : {worst_rel_mean:.3e}")
    print(f"worst relative difference, rg_std  : {worst_rel_std:.3e}")
    print(f"tolerance                           : 1.0e-09")
    print(f"VERDICT rg_mean : {'PASS' if worst_rel_mean < 1e-9 else 'FAIL'}")
    print(f"VERDICT rg_std  : {'PASS' if worst_rel_std < 1e-9 else 'FAIL'}")

    p = OUT / "rg_independent_comparison.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
