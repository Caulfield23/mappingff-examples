"""
Density & Rg comparison between ligpargen and mappingff for 4 oligomers.

Systems (pure oligomer melts, 100 identical chains each):
    9_peo / 9_pmma / 9_ps / 9_pva

Production window: steps 1,500,000 - 2,000,000 (dt = 1 fs, i.e. the final
500 ps NVT stage at 300 K).  NOTE: `minimize` advances the timestep, so the
NVT stage actually starts at 1,500,000 + n_min (n_min <= 1000 = 1 ps); the
same window is applied to all 8 runs so the comparison stays consistent.

Rg
    mass-weighted radius of gyration of each individual chain
        Rg^2 = sum_i m_i |r_i - r_com|^2 / sum_i m_i
    computed from the UNWRAPPED coordinates (xu yu zu) so that chains
    straddling the periodic boundary are handled correctly.
    All 100 chains are evaluated per frame, then averaged over the window.

Density
    Read from log.lammps (thermo column `Density`, g/cm^3).
      * window 1.5M-2.0M  -> the production NVT stage, box frozen => density is
        essentially CONSTANT.  It is not exactly 0 std because the single row at
        step 1,500,000 still belongs to the preceding NPT stage (the NVT stage
        really starts at 1,500,000 + n_min), where the box was still moving; that
        one point gives std ~1e-4 (4th significant digit).
      * window 1.3M-1.5M  -> tail of the eq_lo NPT stage (already at 300 K, box
        equilibrated), where the density genuinely fluctuates; reported as a
        supplementary column.
"""

from __future__ import annotations

import csv
import mmap
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
OUT_DIR = BASE / "analysis" / "_verify" / "rerun_current_script"

SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")

STEP_LO, STEP_HI = 1_500_000, 2_000_000   # production NVT window (inclusive)
NPT_LO, NPT_HI = 1300_000, 1_500_000       # NPT window, half-open [lo, hi)

# Relative-deviation tolerance used to flag "consistent" / "inconsistent".
DEV_TOLERANCE_PCT = 3.0

FRAME_MARK = b"ITEM: TIMESTEP\n"


# --------------------------------------------------------------------------- #
# Topology (masses, molecule ids) from the LAMMPS data file
# --------------------------------------------------------------------------- #
def read_topology(path: Path):
    """Return (mol_of_atom, mass_of_atom, n_atoms) indexed by atom id - 1."""
    natoms = None
    mass_of_type: dict[int, float] = {}
    mol_of_atom = None

    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()

        m = re.match(r"^(\d+)\s+atoms\b", line)
        if m and natoms is None:
            natoms = int(m.group(1))

        if line.startswith("Masses"):
            j = i + 1
            while j < n and not lines[j].strip():   # skip blank line after header
                j += 1
            while j < n and lines[j].strip():
                p = lines[j].split()
                mass_of_type[int(p[0])] = float(p[1])
                j += 1
            i = j
            continue

        if line.startswith("Atoms"):
            j = i + 1
            while j < n and not lines[j].strip():   # skip blank line after header
                j += 1
            rows = []
            while j < n and lines[j].strip():
                p = lines[j].split()
                rows.append((int(p[0]), int(p[1]), int(p[2])))  # id, mol, type
                j += 1
            mol_of_atom = np.zeros(natoms, dtype=np.int64)
            typ_of_atom = np.zeros(natoms, dtype=np.int64)
            for aid, mol, typ in rows:
                mol_of_atom[aid - 1] = mol
                typ_of_atom[aid - 1] = typ
            i = j
            continue

        i += 1

    if natoms is None or mol_of_atom is None:
        raise RuntimeError(f"could not parse Atoms section of {path}")

    missing = sorted(set(typ_of_atom.tolist()) - set(mass_of_type))
    if missing:
        raise RuntimeError(f"{path}: no mass for atom type(s) {missing[:10]}")

    mass_of_atom = np.array([mass_of_type[t] for t in typ_of_atom], dtype=np.float64)
    return mol_of_atom, mass_of_atom, natoms


# --------------------------------------------------------------------------- #
# Rg
# --------------------------------------------------------------------------- #
def chain_rg(xyz: np.ndarray, chain: np.ndarray, mass: np.ndarray, n_chain: int):
    """Mass-weighted Rg for every chain. xyz (N,3), chain (N,) 0-based, mass (N,)."""
    w_sum = np.bincount(chain, weights=mass, minlength=n_chain)
    com = np.empty((n_chain, 3), dtype=np.float64)
    for d in range(3):
        com[:, d] = np.bincount(chain, weights=mass * xyz[:, d], minlength=n_chain)
    com /= w_sum[:, None]

    diff = xyz - com[chain]
    r2 = np.einsum("ij,ij->i", diff, diff)
    num = np.bincount(chain, weights=mass * r2, minlength=n_chain)
    return np.sqrt(num / w_sum)


# --------------------------------------------------------------------------- #
# Trajectory reader
# --------------------------------------------------------------------------- #
def scan_rg(traj: Path, mol_of_atom, mass_of_atom, natoms, step_lo, step_hi,
            verbose=True):
    """Stream a LAMMPS dump, return per-frame chain-Rg statistics in the window."""
    # Chain index (0-based) and per-atom mass are fixed for the whole run.
    uniq = np.unique(mol_of_atom)
    remap = {m: k for k, m in enumerate(uniq.tolist())}
    chain = np.array([remap[m] for m in mol_of_atom.tolist()], dtype=np.int64)
    n_chain = len(uniq)

    steps, rg_mean, rg_std, rg_min, rg_max = [], [], [], [], []
    n_frames_total = 0
    n_checked = 0

    t0 = time.perf_counter()
    with open(traj, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

        # Locate every frame by scanning for the header marker (C-level memmem).
        starts = []
        pos = 0
        while True:
            hit = mm.find(FRAME_MARK, pos)
            if hit < 0:
                break
            starts.append(hit)
            pos = hit + len(FRAME_MARK)
        starts.append(size)
        n_frames_total = len(starts) - 1

        for k in range(n_frames_total):
            beg = starts[k] + len(FRAME_MARK)
            end = mm.find(b"\n", beg)
            step = int(mm[beg:end])

            if not (step_lo <= step <= step_hi):
                continue

            # 9 header lines precede the atom block:
            # 0 TIMESTEP / 1 step / 2 NUMBER OF ATOMS / 3 natoms /
            # 4 BOX BOUNDS / 5-7 bounds / 8 ATOMS ...
            lines = mm[starts[k]:starts[k + 1]].split(b"\n")
            frame_natoms = int(lines[3])
            if frame_natoms != natoms:
                raise RuntimeError(
                    f"{traj.name}: natoms {frame_natoms} != topology {natoms}"
                )

            cols = [c.decode() for c in lines[8].split()[2:]]  # drop "ITEM ATOMS"
            for need in ("id", "mol", "xu", "yu", "zu"):
                if need not in cols:
                    raise RuntimeError(f"{traj.name}: dump has no '{need}' column")
            ia, imol = cols.index("id"), cols.index("mol")
            ix, iy, iz = cols.index("xu"), cols.index("yu"), cols.index("zu")

            raw = np.array([ln.split() for ln in lines[9:9 + frame_natoms]])
            # Cross-check the dump against the data file on the first frame.
            if n_checked == 0:
                ids = raw[:, ia].astype(np.int64)
                if not np.array_equal(ids, np.arange(1, natoms + 1)):
                    raise RuntimeError(f"{traj.name}: atom ids are not 1..N sorted")
                if not np.array_equal(raw[:, imol].astype(np.int64), mol_of_atom):
                    raise RuntimeError(f"{traj.name}: mol ids disagree with data file")
                n_checked += 1

            xyz = raw[:, [ix, iy, iz]].astype(np.float64)
            rg = chain_rg(xyz, chain, mass_of_atom, n_chain)

            steps.append(step)
            rg_mean.append(rg.mean())
            rg_std.append(rg.std(ddof=1))
            rg_min.append(rg.min())
            rg_max.append(rg.max())

        mm.close()

    if verbose:
        print(f"    frames total={n_frames_total}  selected={len(steps)}"
              f"  steps {min(steps) if steps else '-'}..{max(steps) if steps else '-'}"
              f"  ({time.perf_counter() - t0:.1f} s)")

    return {
        "n_frames_total": n_frames_total,
        "steps": np.array(steps, dtype=np.int64),
        "rg_mean": np.array(rg_mean),
        "rg_std": np.array(rg_std),
        "rg_min": np.array(rg_min),
        "rg_max": np.array(rg_max),
        "n_chain": n_chain,
    }


# --------------------------------------------------------------------------- #
# Thermo / density
# --------------------------------------------------------------------------- #
def parse_thermo(path: Path):
    """Collect (step, density, Lx) from every thermo block in a LAMMPS log.

    Returns (rows, block_first_steps).  Blocks are delimited by the thermo
    table header, which LAMMPS prints once per run stage.  Do NOT try to split
    blocks by step gaps: because `minimize` advanced the timestep by n_min
    (567/593/644/1000), the thermo rows of a stage are offset from multiples of
    the thermo interval and the gaps are irregular.
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    rows = []
    block_first = []
    i, n = 0, len(lines)
    while i < n:
        if re.match(r"^\s*Step\b", lines[i]) and "Density" in lines[i]:
            cols = lines[i].split()
            di, li = cols.index("Density"), cols.index("Lx")
            j = i + 1
            first = None
            while j < n:
                p = lines[j].split()
                if not p:
                    j += 1
                    continue
                try:
                    step = int(p[0])
                except ValueError:
                    break                      # "Loop time ...", "# ===", ...
                if len(p) <= max(di, li):
                    break
                if first is None:
                    first = step
                rows.append((step, float(p[di]), float(p[li])))
                j += 1
            if first is not None:
                block_first.append(first)
            i = j
            continue
        i += 1
    return rows, block_first


def parse_min_steps(path: Path):
    """Number of minimisation iterations (= steps the minimiser advanced).

    The FIRST "Loop time of ... for N steps" line in a LAMMPS log belongs to the
    `minimize` command; `minimize` advances the timestep by one per iteration,
    which is why every run starts its production stages at n_min, not 0.
    """
    pat = re.compile(r"Loop time of [\d.eE+-]+ on \d+ procs for (\d+) steps")
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                return int(m.group(1))
    return 0


def stage_structure(block_first):
    """Return (n_min, nvt_stage_start) from the thermo block boundaries.

    The program runs 2,000,000 steps in total (400k deform + 100k eq_hi +
    500k cool + 500k eq_lo + 500k eq_lo_nvt) after the minimiser advanced the
    timestep by n_min, so the final NVT stage starts at 1,500,000 + n_min.
    """
    nvt_start = block_first[-1]
    return nvt_start - 1_500_000, nvt_start


def density_stats(rows, lo, hi, include_hi=True):
    """Aggregate density over [lo, hi] (inclusive unless include_hi is False)."""
    if include_hi:
        sel = [r for r in rows if lo <= r[0] <= hi]
    else:
        sel = [r for r in rows if lo <= r[0] < hi]
    if not sel:
        return dict(n=0, mean=float("nan"), std=float("nan"), sem=float("nan"),
                    L=float("nan"))
    d = np.array([r[1] for r in sel])
    box = np.array([r[2] for r in sel])
    return dict(n=len(d), mean=float(d.mean()), std=float(d.std(ddof=1)),
                sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                L=float(box.mean()))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run = []

    for sysname in SYSTEMS:
        for src in SOURCES:
            run = BASE / src / sysname
            print(f"[{src}/{sysname}]")

            mol_of_atom, mass_of_atom, natoms = read_topology(run / "filled.data")
            rg = scan_rg(run / "dump.lammpstrj", mol_of_atom, mass_of_atom,
                         natoms, STEP_LO, STEP_HI)

            thermo, block_first = parse_thermo(run / "log.lammps")
            n_min, nvt_start = stage_structure(block_first)
            n_min_log = parse_min_steps(run / "log.lammps")
            d_nvt = density_stats(thermo, STEP_LO, STEP_HI, include_hi=True)
            d_npt = density_stats(thermo, NPT_LO, NPT_HI, include_hi=False)
            # density over the *exact* NVT stage (box truly frozen) -> std == 0
            d_nvt_exact = density_stats(thermo, nvt_start, STEP_HI, include_hi=True)

            # Autocorrelation-aware error on <Rg>: mean of 10-frame block means.
            nb = max(1, len(rg["rg_mean"]) // 10)
            blocks = [rg["rg_mean"][i:i + 10].mean() for i in range(0, nb * 10, 10)]
            sem_block = float(np.std(blocks, ddof=1) / np.sqrt(len(blocks))) \
                if len(blocks) > 1 else float("nan")

            per_run.append({
                "system": sysname,
                "source": src,
                "natoms": natoms,
                "n_chain": rg["n_chain"],
                "atoms_per_chain": natoms // rg["n_chain"],
                "frames_total": rg["n_frames_total"],
                "frames_analyzed": len(rg["steps"]),
                "chain_samples": len(rg["steps"]) * rg["n_chain"],
                "rho_nvt_mean": d_nvt["mean"],
                "rho_nvt_std": d_nvt["std"],
                "rho_nvt_n": d_nvt["n"],
                "L_nvt": d_nvt["L"],
                "n_min": n_min,
                "n_min_from_log": n_min_log,
                "nvt_stage_start": nvt_start,
                "rho_nvt_exact_mean": d_nvt_exact["mean"],
                "rho_nvt_exact_std": d_nvt_exact["std"],
                "rho_nvt_exact_n": d_nvt_exact["n"],
                "rho_npt_mean": d_npt["mean"],
                "rho_npt_std": d_npt["std"],
                "rho_npt_sem": d_npt["sem"],
                "rho_npt_n": d_npt["n"],
                "rg_mean": float(rg["rg_mean"].mean()),
                "rg_std": float(rg["rg_std"].mean()),
                "rg_min": float(rg["rg_min"].min()),
                "rg_max": float(rg["rg_max"].max()),
                "rg2_mean": float((rg["rg_mean"] ** 2).mean()),
                "rg_sem_block": sem_block,
            })
            r = per_run[-1]
            print(f"    n_min={n_min} (log: {n_min_log})  NVT stage starts {nvt_start}"
                  f"  rho(NVT)={r['rho_nvt_mean']:.5f}  "
                  f"rho(NPT)={r['rho_npt_mean']:.5f}+-{r['rho_npt_std']:.5f}  "
                  f"Rg={r['rg_mean']:.3f}+-{r['rg_std']:.3f} A")

    # ---------------- write per-run summary ------------------------------- #
    cols = list(per_run[0].keys())
    p1 = OUT_DIR / "density_rg_per_run.csv"
    with open(p1, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(per_run)
    print(f"\nwrote {p1}")

    # ---------------- write comparison table ------------------------------- #
    def rel(a, b):
        return (b - a) / a * 100.0

    comparison = []
    for sysname in SYSTEMS:
        a = next(r for r in per_run if r["system"] == sysname and r["source"] == "ligpargen")
        b = next(r for r in per_run if r["system"] == sysname and r["source"] == "mappingff")

        # z-score using the correlation-aware SEM of each run's <Rg>
        se = np.hypot(a["rg_sem_block"], b["rg_sem_block"])
        z_rg = (b["rg_mean"] - a["rg_mean"]) / se if se > 0 else float("nan")

        comparison.append({
            "system": sysname,
            "atoms_per_chain": a["atoms_per_chain"],
            "n_chain": a["n_chain"],
            "rho_nvt_ligpargen": a["rho_nvt_mean"],
            "rho_nvt_mappingff": b["rho_nvt_mean"],
            "rho_nvt_delta": b["rho_nvt_mean"] - a["rho_nvt_mean"],
            "rho_nvt_dev_pct": rel(a["rho_nvt_mean"], b["rho_nvt_mean"]),
            "rho_nvt_dev_pct_abs": abs(rel(a["rho_nvt_mean"], b["rho_nvt_mean"])),
            "rho_npt_ligpargen": a["rho_npt_mean"],
            "rho_npt_mappingff": b["rho_npt_mean"],
            "rho_npt_std_ligpargen": a["rho_npt_std"],
            "rho_npt_std_mappingff": b["rho_npt_std"],
            "rho_npt_delta": b["rho_npt_mean"] - a["rho_npt_mean"],
            "rho_npt_dev_pct": rel(a["rho_npt_mean"], b["rho_npt_mean"]),
            "rho_npt_dev_pct_abs": abs(rel(a["rho_npt_mean"], b["rho_npt_mean"])),
            "rg_ligpargen": a["rg_mean"],
            "rg_mappingff": b["rg_mean"],
            "rg_std_ligpargen": a["rg_std"],
            "rg_std_mappingff": b["rg_std"],
            "rg_delta": b["rg_mean"] - a["rg_mean"],
            "rg_dev_pct": rel(a["rg_mean"], b["rg_mean"]),
            "rg_dev_pct_abs": abs(rel(a["rg_mean"], b["rg_mean"])),
            "rg_zscore": z_rg,
            "verdict_rg": ("consistent" if abs(rel(a["rg_mean"], b["rg_mean"])) <= DEV_TOLERANCE_PCT
                           else "INCONSISTENT"),
            "verdict_rho": ("consistent" if abs(rel(a["rho_nvt_mean"], b["rho_nvt_mean"])) <= DEV_TOLERANCE_PCT
                            else "INCONSISTENT"),
        })

    cols2 = list(comparison[0].keys())
    p2 = OUT_DIR / "density_rg_comparison.csv"
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols2)
        w.writeheader()
        w.writerows(comparison)
    print(f"wrote {p2}")

    # ---------------- console summary -------------------------------------- #
    print("\n" + "=" * 108)
    print(f"{'system':>9} {'rho_NVT (1.5-2.0M)':>18} {'dev%':>7}   "
          f"{'rho_NPT (1.3-1.5M)':>20} "
          f"{'dev%':>7}   {'Rg (A)':>20} {'dev%':>7} {'z':>6}")
    print("-" * 108)
    for c in comparison:
        print(f"{c['system']:>9} "
              f"{c['rho_nvt_ligpargen']:8.4f}/{c['rho_nvt_mappingff']:<8.4f} "
              f"{c['rho_nvt_dev_pct']:7.2f}   "
              f"{c['rho_npt_ligpargen']:8.4f}/{c['rho_npt_mappingff']:<8.4f} "
              f"{c['rho_npt_dev_pct']:7.2f}   "
              f"{c['rg_ligpargen']:8.3f}/{c['rg_mappingff']:<8.3f} "
              f"{c['rg_dev_pct']:7.2f} {c['rg_zscore']:6.2f}")
    print("=" * 108)
    print(f"tolerance for 'consistent': |deviation| <= {DEV_TOLERANCE_PCT:.1f}%")


if __name__ == "__main__":
    main()