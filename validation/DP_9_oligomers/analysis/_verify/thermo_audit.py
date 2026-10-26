"""
Independent thermo / stage-structure audit for the 8 log.lammps files.

Re-implements thermo parsing from scratch (does NOT import the analysis
script) so that a parsing bug in analyze_density_rg.py cannot hide here.

Goals
  1. Enumerate every thermo block: first step, last step, n rows, columns.
  2. Reconstruct the stage boundaries (minimize -> deform -> eq_hi -> cool ->
     eq_lo -> eq_lo_nvt) and locate the true NVT production stage.
  3. Detect DUPLICATED thermo rows (a step appearing in two blocks, which
     happens at a stage boundary) and identify which stage each belongs to.
  4. Drift / stationarity analysis over the analysis window for Temp, Press,
     PotEng, Density (linear slope per ns) plus a first-half vs second-half
     comparison.
  5. Recompute rho_nvt / rho_npt from the raw log and compare against
     density_rg_per_run.csv.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")

STEP_LO, STEP_HI = 1_500_000, 2_000_000
NPT_LO, NPT_HI = 1_300_000, 1_500_000

WANT = ("Temp", "Press", "PotEng", "Density", "Lx")


def parse_blocks(path: Path):
    """Return list of dicts: {header, cols, rows:[dict]} for every thermo table."""
    lines = path.read_text(errors="replace").splitlines()
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        if re.match(r"^\s*Step\b", lines[i]):
            cols = lines[i].split()
            j = i + 1
            rows = []
            while j < n:
                p = lines[j].split()
                if not p:
                    j += 1
                    continue
                try:
                    step = int(p[0])
                except ValueError:
                    break
                if len(p) != len(cols):
                    break
                vals = []
                bad = False
                for c in p[1:]:
                    try:
                        vals.append(float(c))
                    except ValueError:
                        bad = True
                        break
                if bad:
                    break
                rows.append((step, vals))
                j += 1
            blocks.append({"header": lines[i].strip(), "cols": cols[1:],
                           "rows": rows, "header_line_no": i})
            i = j
            continue
        i += 1
    return blocks


def lin_slope_per_ns(step, y):
    """OLS slope of y vs time, expressed per ns (1 ns = 1e6 steps of 1 fs)."""
    step = np.asarray(step, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(step) < 3:
        return float("nan")
    A = np.vstack([step, np.ones_like(step)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0] * 1e6)


def main():
    report = []
    for sysname in SYSTEMS:
        for src in SOURCES:
            run = BASE / src / sysname
            log = run / "log.lammps"
            blocks = parse_blocks(log)
            rep = {"source": src, "system": sysname}

            # ---- block structure -----------------------------------------
            binfo = []
            for b in blocks:
                st = [r[0] for r in b["rows"]]
                binfo.append({
                    "cols": b["cols"],
                    "n_rows": len(st),
                    "first": st[0] if st else None,
                    "last": st[-1] if st else None,
                    "has_density": "Density" in b["cols"],
                    "step_diffs": sorted(set(np.diff(np.array(st)).tolist())) if len(st) > 1 else [],
                })
            rep["blocks"] = binfo

            # ---- minimize steps ------------------------------------------
            txt = log.read_text(errors="replace")
            m = re.search(r"Loop time of [\d.eE+-]+ on \d+ procs for (\d+) steps", txt)
            rep["min_loop_steps"] = int(m.group(1)) if m else None
            # the minimize "Stopping criterion" line gives the real iteration count
            m2 = re.search(r"Stopping criterion = .*?\n", txt)
            rep["min_stopping"] = m2.group(0).strip() if m2 else None
            m3 = re.search(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", txt, re.M)
            # iterate count from "Minimization stats" table: line "Iterations"
            it = re.search(r"Iterations:\s*(\S+)", txt)
            rep["min_iterations"] = it.group(1) if it else None

            # ---- rows from blocks that carry Density ----------------------
            rows = []          # (step, block_idx, value dict)
            for bi, b in enumerate(blocks):
                if "Density" not in b["cols"]:
                    continue
                idx = {c: k for k, c in enumerate(b["cols"])}
                for step, vals in b["rows"]:
                    d = {c: vals[idx[c]] for c in WANT if c in idx}
                    rows.append((step, bi, d))
            rep["n_density_rows"] = len(rows)

            steps = np.array([r[0] for r in rows])
            # duplicate steps across blocks
            uniq, counts = np.unique(steps, return_counts=True)
            dups = uniq[counts > 1]
            rep["duplicate_steps"] = [(int(s), int((steps == s).sum())) for s in dups]
            dup_detail = []
            for s in dups:
                entries = [(r[1], r[2]) for r in rows if r[0] == s]
                dup_detail.append({
                    "step": int(s),
                    "entries": [{"block": b, "Density": e["Density"], "Lx": e["Lx"]}
                                for b, e in entries],
                })
            rep["duplicate_detail"] = dup_detail

            # ---- stage reconstruction ------------------------------------
            nmin = rep["min_loop_steps"] or 0
            stages = [("deform", nmin, nmin + 400_000),
                      ("eq_hi", nmin + 400_000, nmin + 500_000),
                      ("cool", nmin + 500_000, nmin + 1_000_000),
                      ("eq_lo", nmin + 1_000_000, nmin + 1_500_000),
                      ("eq_lo_nvt", nmin + 1_500_000, nmin + 2_000_000)]
            rep["n_min"] = nmin
            rep["stages"] = [{"name": nm, "lo": lo, "hi": hi} for nm, lo, hi in stages]
            rep["nvt_stage_start"] = nmin + 1_500_000
            rep["run_final_step"] = nmin + 2_000_000

            # ---- analysis window ------------------------------------------
            win = [r for r in rows if STEP_LO <= r[0] <= STEP_HI]
            wsteps = np.array([r[0] for r in win])
            rep["win_n"] = len(win)
            rep["win_first"] = int(wsteps.min()) if len(win) else None
            rep["win_last"] = int(wsteps.max()) if len(win) else None
            rep["win_step_diffs"] = sorted(set(np.diff(wsteps).tolist())) if len(win) > 1 else []
            # which rows in the window come from the NPT stage?
            nvt_start = rep["nvt_stage_start"]
            rep["win_rows_before_nvt"] = [
                {"step": int(r[0]), "block": r[1], "Density": r[2]["Density"], "Lx": r[2]["Lx"]}
                for r in win if r[0] < nvt_start]
            # rows at exactly the stage boundary (duplicated step) and their provenance
            rep["win_boundary_rows"] = [
                {"step": int(r[0]), "block": r[1], "Density": r[2]["Density"], "Lx": r[2]["Lx"]}
                for r in win if r[0] == nvt_start]
            rep["win_last_block_is_nvt"] = max((r[1] for r in win), default=None)
            rep["win_blocks_used"] = sorted(set(r[1] for r in win))

            def stats(sel, key):
                y = np.array([r[2][key] for r in sel], dtype=np.float64)
                s = np.array([r[0] for r in sel], dtype=np.float64)
                if len(y) == 0:
                    return {}
                return {
                    "n": len(y), "mean": float(y.mean()), "std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                    "min": float(y.min()), "max": float(y.max()),
                    "slope_per_ns": lin_slope_per_ns(s, y),
                    "first_half_mean": float(y[:len(y) // 2].mean()),
                    "second_half_mean": float(y[len(y) // 2:].mean()),
                    "half_delta": float(y[len(y) // 2:].mean() - y[:len(y) // 2].mean()),
                }

            rep["win_stats"] = {k: stats(win, k) for k in WANT}
            # exact NVT stage only
            winnvt = [r for r in rows if nvt_start <= r[0] <= STEP_HI]
            rep["win_nvt_only_stats"] = {k: stats(winnvt, k) for k in WANT}
            rep["win_nvt_only_n"] = len(winnvt)

            # ---- NPT window -----------------------------------------------
            npt = [r for r in rows if NPT_LO <= r[0] < NPT_HI]
            rep["npt_n"] = len(npt)
            rep["npt_stats"] = {k: stats(npt, k) for k in WANT}
            # and the tail of eq_lo closer to the boundary
            npt2 = [r for r in rows if 1_400_000 <= r[0] < 1_500_000]
            rep["npt_tail400k_stats"] = {k: stats(npt2, k) for k in WANT}

            # ---- independent recomputation of the reported densities ------
            rep["rho_nvt_mean_allrows"] = float(np.mean([r[2]["Density"] for r in win]))
            rep["rho_nvt_std_allrows"] = float(np.std([r[2]["Density"] for r in win], ddof=1))
            rep["rho_nvt_mean_nvtblock"] = float(np.mean([r[2]["Density"] for r in winnvt]))
            rep["rho_npt_mean"] = float(np.mean([r[2]["Density"] for r in npt]))
            rep["rho_npt_std"] = float(np.std([r[2]["Density"] for r in npt], ddof=1))
            report.append(rep)
            print(f"{src}/{sysname}: nmin={nmin} rows(win)={len(win)} "
                  f"dups={[int(s) for s in dups]} nvt_start={nvt_start} "
                  f"rho_nvt={rep['rho_nvt_mean_allrows']:.6f} "
                  f"rho_npt={rep['rho_npt_mean']:.6f}", flush=True)

    p = Path(__file__).with_name("thermo_audit.json")
    p.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
