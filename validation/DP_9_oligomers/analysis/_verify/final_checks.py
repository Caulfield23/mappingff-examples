"""Verifier part 9: impropers (re-check), build artifact drift, density cross-check."""
import re, os, time
from pathlib import Path
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
NA = 6.02214076e23

print("### Improper counts in the single-chain templates (grep) ###")
for sysname in SYSTEMS:
    for src, fn in (("ligpargen", f"{sysname}.lmp"), ("mappingff", f"{sysname}.data")):
        p = BASE / src / sysname / "build" / fn
        if not p.exists():
            print(f"  {src}/{sysname}: {fn} MISSING")
            continue
        txt = p.read_text(errors="replace")
        lines = txt.splitlines()
        cnt = None
        for i, l in enumerate(lines[:30]):
            m = re.match(r"^(\d+)\s+impropers\b", l.strip(), re.I)
            if m:
                cnt = int(m.group(1)); hdr = i; break
        rows = -1
        if cnt is not None:
            j = hdr + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            rows = 0
            while j < len(lines) and lines[j].strip():
                rows += 1; j += 1
        print(f"  {src}/{sysname}/{fn}: header impropers={cnt}  section rows={rows}")

print()
print("### mappingff build/ vs run/ filled.data consistency ###")
for sysname in SYSTEMS:
    p_run = BASE / "mappingff" / sysname / "filled.data"
    p_bld = BASE / "mappingff" / sysname / "build" / "filled.data"
    if not p_bld.exists():
        print(f"  {sysname}: no build/filled.data"); continue
    a = p_run.read_bytes(); b = p_bld.read_bytes()
    print(f"  {sysname}: identical bytes={a == b}  "
          f"run_size={len(a)} build_size={len(b)}  "
          f"run_mtime={time.strftime('%m-%d %H:%M', time.localtime(p_run.stat().st_mtime))}  "
          f"build_mtime={time.strftime('%m-%d %H:%M', time.localtime(p_bld.stat().st_mtime))}")

print()
print("### Independent density cross-check: rho = M/(N_A V) using mass from filled.data ###")
for sysname in SYSTEMS:
    for src in ("ligpargen", "mappingff"):
        run = BASE / src / sysname
        # mass
        txt = (run / "filled.data").read_text(errors="replace").splitlines()
        i = 0
        mt = {}
        while not txt[i].strip().startswith("Masses"):
            i += 1
        i += 1
        while not txt[i].strip():
            i += 1
        while txt[i].strip():
            p = txt[i].split(); mt[int(p[0])] = float(p[1]); i += 1
        # atoms
        while not txt[i].strip().startswith("Atoms"):
            i += 1
        i += 1
        while not txt[i].strip():
            i += 1
        M = 0.0
        nat = 0
        while txt[i].strip():
            p = txt[i].split(); M += mt[int(p[2])]; nat += 1; i += 1
        # log: last thermo row with Density and Lx Ly Lz
        log = (run / "log.lammps").read_text(errors="replace").splitlines()
        last = None
        for l in log:
            s = l.split()
            if len(s) > 10 and s[0].isdigit():
                try:
                    d = float(s[10])
                except (ValueError, IndexError):
                    continue
                last = (int(s[0]), d, float(s[11]), float(s[12]), float(s[13]))
        # recompute at the LAST thermo row and at the row closest to 2e6
        # find the header to know the Density column index
        cols = None
        for l in log:
            if re.match(r"^\s*Step\b", l) and "Density" in l:
                cols = l.split()
        di = cols.index("Density"); li = cols.index("Lx")
        rows = []
        for l in log:
            s = l.split()
            if len(s) > max(di, li):
                try:
                    rows.append((int(s[0]), float(s[di]), float(s[li]), float(s[li+1]), float(s[li+2])))
                except (ValueError, IndexError):
                    pass
        sel = [r for r in rows if 1_950_000 <= r[0] <= 2_000_000]
        if sel:
            st, rho_log, lx, ly, lz = sel[-1]
            V = lx * ly * lz
            rho_calc = M / NA / (V * 1e-24)
            print(f"  {sysname:8s} {src:9s} step={st} log_rho={rho_log:.5f}  "
                  f"rho(M,V)={rho_calc:.5f}  diff={rho_log-rho_calc:+.2e}  "
                  f"L={lx:.4f} natoms={nat}")
        else:
            print(f"  {sysname:8s} {src:9s}: no rows in [1.95e6,2.0e6]")

print()
print("### Equilibrium check: Rg drift over the NVT window (from dumps, last 101 frames) ###")
for sysname in SYSTEMS:
    for src in ("ligpargen", "mappingff"):
        run = BASE / src / sysname
        log = (run / "log.lammps").read_text(errors="replace").splitlines()
        cols = None
        for l in log:
            if re.match(r"^\s*Step\b", l) and "Density" in l:
                cols = l.split()
        di = cols.index("Density")
        rows = []
        for l in log:
            s = l.split()
            if len(s) > di:
                try:
                    rows.append((int(s[0]), float(s[di])))
                except (ValueError, IndexError):
                    pass
        for lo, hi, tag in ((1_300_000, 1_500_000, "NPT"), (1_500_000, 2_000_000, "NVT")):
            sel = [r for r in rows if lo <= r[0] < hi] if tag == "NPT" else [r for r in rows if lo <= r[0] <= hi]
            d = np.array([r[1] for r in sel])
            # split into 3 thirds to see drift
            k = len(d) // 3
            print(f"  {sysname:8s} {src:9s} {tag}: n={len(d)} mean={d.mean():.5f} "
                  f"first_third={d[:k].mean():.5f} last_third={d[-k:].mean():.5f} "
                  f"drift={d[-k:].mean()-d[:k].mean():+.2e}")
