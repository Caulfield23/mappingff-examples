"""
Independent frame enumeration + corruption audit for the 8 dump.lammpstrj files.

Deliberately does NOT import analyze_density_rg.py -- it re-implements the scan
from scratch so that a bug in the analysis script cannot hide here.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SOURCES = ("ligpargen", "mappingff")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")

STEP_LO, STEP_HI = 1_500_000, 2_000_000
EXPECTED_HEADER = b"ITEM: ATOMS id mol type q x y z xu yu zu"

MARK = b"ITEM: TIMESTEP\n"


def enumerate_frames(path: Path):
    """Return (steps list, list of (start,end) byte offsets)."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        mm = mmap_mmap(fh)
        starts = []
        pos = 0
        while True:
            hit = mm.find(MARK, pos)
            if hit < 0:
                break
            starts.append(hit)
            pos = hit + len(MARK)
        starts.append(size)
        steps = []
        for k in range(len(starts) - 1):
            beg = starts[k] + len(MARK)
            end = mm.find(b"\n", beg)
            steps.append(int(mm[beg:end]))
        mm.close()
    return steps, starts


def mmap_mmap(fh):
    import mmap
    return mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)


def audit_run(src: str, sysname: str, deep: bool = True):
    run = BASE / src / sysname
    traj = run / "dump.lammpstrj"
    out = {"source": src, "system": sysname, "traj": str(traj)}
    t0 = time.perf_counter()

    with open(traj, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        mm = mmap_mmap(fh)
        starts = []
        pos = 0
        while True:
            hit = mm.find(MARK, pos)
            if hit < 0:
                break
            starts.append(hit)
            pos = hit + len(MARK)
        starts.append(size)
        n_frames = len(starts) - 1

        steps = []
        for k in range(n_frames):
            beg = starts[k] + len(MARK)
            end = mm.find(b"\n", beg)
            steps.append(int(mm[beg:end]))

        out["n_frames"] = n_frames
        out["first_step"] = steps[0]
        out["last_step"] = steps[-1]
        out["steps_strictly_increasing"] = all(steps[i] < steps[i + 1] for i in range(n_frames - 1))
        out["n_duplicate_steps"] = n_frames - len(set(steps))
        diffs = np.diff(np.array(steps))
        out["unique_step_diffs"] = sorted(set(diffs.tolist()))
        expected = list(range(5000, 2_000_001, 5000))
        out["n_expected"] = len(expected)
        out["steps_match_expected_exactly"] = steps == expected
        missing = sorted(set(expected) - set(steps))
        extra = sorted(set(steps) - set(expected))
        out["missing_steps"] = missing[:20]
        out["n_missing"] = len(missing)
        out["extra_steps"] = extra[:20]
        out["n_extra"] = len(extra)

        # selection window check -- replicate the analysis filter exactly
        sel_idx = [k for k, s in enumerate(steps) if STEP_LO <= s <= STEP_HI]
        out["n_selected"] = len(sel_idx)
        out["selected_steps_first"] = steps[sel_idx[0]] if sel_idx else None
        out["selected_steps_last"] = steps[sel_idx[-1]] if sel_idx else None
        out["selected_step_diffs"] = sorted(set(np.diff(np.array([steps[k] for k in sel_idx])).tolist())) if len(sel_idx) > 1 else []
        out["selected_steps_missing"] = sorted(
            set(range(STEP_LO, STEP_HI + 1, 5000)) - {steps[k] for k in sel_idx})
        out["n_selected_out_of_window"] = sum(
            1 for k in sel_idx if not (STEP_LO <= steps[k] <= STEP_HI))
        # frames just outside the window that must NOT be included
        out["steps_just_outside"] = [s for s in steps if s in (1_495_000, 2_005_000, 1_499_000, 1_501_000)]

        # ---- deep per-frame validation of the selected frames -------------
        anomalies = []
        natoms_hdr = set()
        hdrs = set()
        boxbad = []
        nanbad = []
        linecount_bad = []
        idbad = []
        molbad = []
        ncols = set()
        for k in sel_idx:
            blk = mm[starts[k]:starts[k + 1]]
            lines = blk.split(b"\n")
            # 0 ITEM: TIMESTEP | 1 step | 2 ITEM: NUMBER OF ATOMS | 3 natoms |
            # 4 ITEM: BOX BOUNDS ... | 5,6,7 bounds | 8 ITEM: ATOMS ...
            if lines[0] != b"ITEM: TIMESTEP":
                anomalies.append((steps[k], "bad TIMESTEP marker"))
            if lines[2] != b"ITEM: NUMBER OF ATOMS":
                anomalies.append((steps[k], "bad NUMBER OF ATOMS marker"))
            if not lines[4].startswith(b"ITEM: BOX BOUNDS"):
                anomalies.append((steps[k], "bad BOX BOUNDS marker: %r" % lines[4]))
            na = int(lines[3])
            natoms_hdr.add(na)
            hdrs.add(lines[8])
            # box bounds finite?
            for bi in (5, 6, 7):
                try:
                    vals = np.array([float(v) for v in lines[bi].split()])
                except Exception as e:
                    boxbad.append((steps[k], bi, "unparsable %r" % lines[bi]))
                    continue
                if not np.all(np.isfinite(vals)):
                    boxbad.append((steps[k], bi, "non-finite %r" % lines[bi]))
            # atom lines
            atom_lines = lines[9:9 + na]
            if len(atom_lines) != na:
                linecount_bad.append((steps[k], len(atom_lines), na))
                anomalies.append((steps[k], "atom line count %d != %d" % (len(atom_lines), na)))
                continue
            if atom_lines[-1] == b"":
                anomalies.append((steps[k], "last atom line empty"))
            try:
                toks = np.array(atom_lines[0].split())
                ncols.add(toks.size)
                arr = np.array(b" ".join(atom_lines).split())
                if arr.size != na * 10:
                    linecount_bad.append((steps[k], arr.size, na * 10))
                    anomalies.append((steps[k], "token count %d != %d" % (arr.size, na * 10)))
                    continue
                arr = arr.reshape(na, 10)
                ids = arr[:, 0].astype(np.int64)
                if not np.array_equal(ids, np.arange(1, na + 1)):
                    idbad.append(steps[k])
                mol = arr[:, 1].astype(np.int64)
                num = arr[:, 3:].astype(np.float64)
                if not np.all(np.isfinite(num)):
                    nanbad.append(steps[k])
                q = num[:, 0]
                if not np.all(np.isfinite(q)):
                    nanbad.append(steps[k])
                # mol must be 1..100
                if mol.min() < 1 or mol.max() > 100:
                    molbad.append((steps[k], mol.min(), mol.max()))
            except Exception as e:
                anomalies.append((steps[k], "parse error %r" % (e,)))
        out["headers_seen"] = sorted(h.decode(errors="replace") for h in hdrs)
        out["header_exact"] = (len(hdrs) == 1 and next(iter(hdrs)) == EXPECTED_HEADER)
        out["natoms_header_values"] = sorted(natoms_hdr)
        out["anomalies"] = anomalies[:50]
        out["n_anomalies"] = len(anomalies)
        out["box_bounds_problems"] = boxbad[:20]
        out["nonfinite_coord_frames"] = nanbad
        out["atom_line_count_problems"] = linecount_bad[:20]
        out["atom_id_problem_frames"] = idbad
        out["mol_range_problem_frames"] = molbad[:20]
        out["n_columns_seen"] = sorted(ncols)

        # also verify the very first and last frames are well formed (not in window)
        for k in (0, n_frames - 1):
            blk = mm[starts[k]:starts[k + 1]]
            lines = blk.split(b"\n")
            na = int(lines[3])
            out.setdefault("edge_frames", []).append(
                {"step": steps[k], "natoms": na, "n_atom_lines": len(lines[9:9 + na])})

        mm.close()

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    results = []
    for sysname in SYSTEMS:
        for src in SOURCES:
            print(f"--- auditing {src}/{sysname} ...", flush=True)
            r = audit_run(src, sysname)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items()
                              if k not in ("anomalies", "headers_seen")}, default=str))
    p = Path(__file__).with_name("frame_audit.json")
    p.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
