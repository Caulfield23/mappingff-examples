"""
Independent density audit (verifier: density-log-audit).

Deliberately does NOT import or reuse analyze_density_rg.py.
Parsing here is done with a whole-file regex sweep, and the mass/volume
end-to-end check is done with a separate minimal data-file reader.

Outputs a full comparison table.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
ANA = BASE / "analysis"
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")

STEP_LO, STEP_HI = 1_500_000, 2_000_000
NPT_LO, NPT_HI = 1_300_000, 1_500_000

# amu -> g  and  A^3 -> cm^3
AMU_TO_G = 1.66053906660e-24
A3_TO_CM3 = 1.0e-24

HDR = re.compile(
    r"(?m)^\s*Step\s+[^\n]*\bDensity\b[^\n]*$\n"          # header line
    r"(?P<rows>(?:[ \t]*\d+[^\n]*\n)+)"                     # numeric rows
)


def parse_log(text: str):
    """Return list of blocks; each block = list of (step, density, Lx)."""
    blocks = []
    for m in HDR.finditer(text):
        header = text[m.start():m.start() + text[m.start():].index("\n") + 1]
        cols = header.split()
        di = cols.index("Density")
        li = cols.index("Lx")
        rows = []
        for line in m.group("rows").splitlines():
            p = line.split()
            rows.append((int(p[0]), float(p[di]), float(p[li])))
        blocks.append(rows)
    return blocks


def read_masses_and_types(data_path: Path):
    """Minimal reader: returns (natoms, mass_of_type dict, type_of_atom array)."""
    lines = data_path.read_text(errors="replace").splitlines()
    natoms = None
    mass_of_type = {}
    types = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if natoms is None:
            mm = re.match(r"^(\d+)\s+atoms\b", s)
            if mm:
                natoms = int(mm.group(1))
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
                p = lines[i].split()
                types.append(int(p[2]))
                i += 1
            continue
        i += 1
    return natoms, mass_of_type, np.asarray(types, dtype=np.int64)


def min_steps(text: str):
    m = re.search(r"Loop time of [\d.eE+-]+ on \d+ procs for (\d+) steps", text)
    return int(m.group(1)) if m else None


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return len(v), float(v.mean()), float(v.std(ddof=1))


def main():
    per_run_csv = pd.read_csv(ANA / "density_rg_per_run.csv")

    rows = []
    for sysname in SYSTEMS:
        for src in SOURCES:
            run = BASE / src / sysname
            text = (run / "log.lammps").read_text(errors="replace")
            blocks = parse_log(text)

            flat = [(s, d, l, bi) for bi, b in enumerate(blocks) for (s, d, l) in b]
            steps = np.array([f[0] for f in flat])
            dens = np.array([f[1] for f in flat])
            lxs = np.array([f[2] for f in flat])

            n_min = min_steps(text)
            nvt_start = blocks[-1][0][0]          # first step of the last block
            first_row_of_block1 = blocks[0][0][0]

            # --- window A: [1.5e6, 2.0e6] inclusive -------------------------
            mA = (steps >= STEP_LO) & (steps <= STEP_HI)
            nA, mA_mean, mA_std = stats(dens[mA])
            LA_min, LA_max = float(lxs[mA].min()), float(lxs[mA].max())

            # --- window B: [1.3e6, 1.5e6) half-open -------------------------
            mB = (steps >= NPT_LO) & (steps < NPT_HI)
            nB, mB_mean, mB_std = stats(dens[mB])

            # --- exact NVT window: [nvt_start, 2.0e6] -----------------------
            mE = (steps >= nvt_start) & (steps <= STEP_HI)
            nE, mE_mean, mE_std = stats(dens[mE])

            # --- contamination analysis ------------------------------------
            # rows of the "exact" NVT window that are constant
            nvt_vals = np.unique(np.round(dens[mE], 12))
            # the offending rows: in window A but in the last-but-one block
            prev_block_steps = {s for (s, d, l, bi) in flat
                                if bi == len(blocks) - 2}
            off = [f for f, keep in zip(flat, mA)
                   if keep and f[3] == len(blocks) - 2]
            off_vals = [f[1] for f in off]
            # duplicate steps inside window A (same step in >1 block)
            seen = {}
            dup_steps = []
            for (s, d, l, bi) in flat:
                if not (STEP_LO <= s <= STEP_HI):
                    continue
                if s in seen:
                    dup_steps.append((s, seen[s], bi, d == seen[s][1]))
                else:
                    seen[s] = (bi, d)

            # --- end-to-end mass / volume ----------------------------------
            natoms, mass_of_type, types = read_masses_and_types(run / "filled.data")
            m_tot = sum(mass_of_type[int(t)] for t in types)
            Lx = float(blocks[-1][-1][2])           # final Lx printed in the log
            V = Lx ** 3
            rho_e2e = m_tot / V * AMU_TO_G / A3_TO_CM3

            # also: last row of the whole log (step 2e6+n_min) has same box
            Lx_lastrow = float(blocks[-1][-1][2])

            rec = per_run_csv[(per_run_csv.system == sysname) &
                              (per_run_csv.source == src)].iloc[0]

            rows.append(dict(
                system=sysname, source=src, natoms=natoms,
                n_blocks=len(blocks),
                block_rows=",".join(str(len(b)) for b in blocks),
                block_first_steps=",".join(str(b[0][0]) for b in blocks),
                block_last_steps=",".join(str(b[-1][0]) for b in blocks),
                n_min_log=n_min, block1_first_step=first_row_of_block1,
                nvt_start=nvt_start,
                nvt_start_ok=(nvt_start == 1_500_000 + n_min),
                # mine
                my_nA=nA, my_rhoA=mA_mean, my_rhoA_std=mA_std,
                L_in_A_min=LA_min, L_in_A_max=LA_max,
                my_nB=nB, my_rhoB=mB_mean, my_rhoB_std=mB_std,
                my_nE=nE, my_rhoE=mE_mean, my_rhoE_std=mE_std,
                n_unique_nvt_vals=len(nvt_vals),
                off_rows_in_A=off_vals,
                dup_steps_in_A=len(dup_steps),
                dup_all_equal=all(d[3] for d in dup_steps),
                # csv
                csv_rhoA=rec.rho_nvt_mean, csv_rhoA_std=rec.rho_nvt_std, csv_nA=rec.rho_nvt_n,
                csv_rhoB=rec.rho_npt_mean, csv_rhoB_std=rec.rho_npt_std, csv_nB=rec.rho_npt_n,
                csv_rhoE=rec.rho_nvt_exact_mean, csv_rhoE_std=rec.rho_nvt_exact_std,
                csv_nE=rec.rho_nvt_exact_n,
                csv_n_min=rec.n_min, csv_nvt_start=rec.nvt_stage_start,
                # e2e
                m_total=m_tot, Lx_final=Lx, V_A3=V, rho_e2e=rho_e2e,
                rho_log_final=dens[-1],
            ))
            print(f"[{src}/{sysname}] done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ANA / "_verify" / "audit_density.csv", index=False)

    # ---------------- report ------------------------------------------- #
    print("\n" + "=" * 120)
    print("A) BLOCK STRUCTURE / n_min / NVT START")
    print("=" * 120)
    for _, r in df.iterrows():
        print(f"{r.source:9s} {r.system:7s} nblocks={r.n_blocks} rows/block=[{r.block_rows}] "
              f"first=[{r.block_first_steps}] last=[{r.block_last_steps}]")
        print(f"{'':9s} {'':7s} n_min(log)={r.n_min_log}  block1_first_step={r.block1_first_step}  "
              f"nvt_start={r.nvt_start} (expected {1_500_000 + r.n_min_log})  "
              f"OK={r.nvt_start_ok}   csv n_min={r.csv_n_min} nvt_start={r.csv_nvt_start}")

    print("\n" + "=" * 120)
    print("B) DENSITY COMPARISON (independent vs analysis CSV)")
    print("=" * 120)
    hdr = (f"{'run':22s} {'nA':>4s} {'mine rho_NVT':>13s} {'csv rho_NVT':>13s} "
           f"{'d(rho)':>10s} {'mine std':>10s} {'csv std':>10s} | "
           f"{'nB':>4s} {'mine rho_NPT':>13s} {'csv rho_NPT':>13s} {'d(rho)':>10s} "
           f"{'mine sd':>10s} {'csv sd':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r.source + '/' + r.system:22s} {r.my_nA:4d} {r.my_rhoA:13.7f} "
              f"{r.csv_rhoA:13.7f} {r.my_rhoA - r.csv_rhoA:10.2e} "
              f"{r.my_rhoA_std:10.3e} {r.csv_rhoA_std:10.3e} | "
              f"{r.my_nB:4d} {r.my_rhoB:13.7f} {r.csv_rhoB:13.7f} {r.my_rhoB - r.csv_rhoB:10.2e} "
              f"{r.my_rhoB_std:10.3e} {r.csv_rhoB_std:10.3e}")

    print("\n" + "=" * 120)
    print("C) NVT WINDOW: box constancy + contamination + exact-NVT variant")
    print("=" * 120)
    hdr = (f"{'run':22s} {'Lx min':>11s} {'Lx max':>11s} {'dLx':>10s} "
           f"{'uniq rho':>9s} {'exact mean':>12s} {'exact std':>10s} {'nE':>4s} "
           f"{'csv exact':>12s} {'offending rho in A':>20s} {'dups':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        off = ",".join(f"{v:.7f}" for v in r.off_rows_in_A) or "-"
        print(f"{r.source + '/' + r.system:22s} {r.L_in_A_min:11.6f} {r.L_in_A_max:11.6f} "
              f"{r.L_in_A_max - r.L_in_A_min:10.2e} {r.n_unique_nvt_vals:9d} "
              f"{r.my_rhoE:12.7f} {r.my_rhoE_std:10.2e} {r.my_nE:4d} "
              f"{r.csv_rhoE:12.7f} {off:>20s} {r.dup_steps_in_A:5d}")

    print("\n" + "=" * 120)
    print("D) END-TO-END:  rho = M_total / V  vs  LAMMPS 'Density'")
    print("=" * 120)
    hdr = (f"{'run':22s} {'M_total(amu)':>14s} {'Lx_final':>11s} {'V(A^3)':>13s} "
           f"{'rho e2e':>11s} {'rho log':>11s} {'rel diff':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        rel = (r.rho_e2e - r.rho_log_final) / r.rho_log_final
        print(f"{r.source + '/' + r.system:22s} {r.m_total:14.6f} {r.Lx_final:11.6f} "
              f"{r.V_A3:13.4f} {r.rho_e2e:11.6f} {r.rho_log_final:11.6f} {rel:11.2e}")

    df.to_csv(ANA / "_verify" / "audit_density.csv", index=False)
    print(f"\nwrote {ANA / '_verify' / 'audit_density.csv'}")


if __name__ == "__main__":
    main()
