"""
Secondary adversarial probe (reuses my own rg_independent.scan_rg):

 1. Is `rg2_mean` in density_rg_per_run.csv really <Rg^2>?  (the script computes
    it as mean_over_frames( frame_mean_Rg ** 2 ), i.e. <Rg>^2 -- check whether it
    equals rg_mean**2 and how far it is from a genuine <Rg^2>.)
 2. Is `rg_sem_block` (10-frame block means) a sound uncertainty on <Rg>?
    Compare against the naive frame-wise SEM and against an
    autocorrelation-corrected SEM (integrated autocorrelation time).
 3. What does the quoted "+-" (rg_std) actually represent?
"""
from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, r"C:\schoolwork\Projects\mappingff\validation\longer\analysis\_verify")

from rg_independent import BASE, PER_RUN_CSV, parse_topology, scan_rg, SOURCES, SYSTEMS


def autocorr_sem(x, dt_ps=5.0):
    """SEM of the mean of a correlated series, using the integrated
    autocorrelation time estimated by the initial-positive-sequence rule."""
    n = len(x)
    m = statistics.fmean(x)
    dev = [v - m for v in x]
    c0 = statistics.fmean([d * d for d in dev])
    # autocovariance via FFT-free direct sum (n is only 101)
    tau_c = 1.0
    lag = 1
    while lag < n:
        c = statistics.fmean([dev[i] * dev[i + lag] for i in range(n - lag)])
        if c <= 0.0:
            break
        tau_c += 2.0 * c / c0
        lag += 1
    tau_c = max(tau_c, 1.0)
    return math.sqrt(c0 * tau_c / n), tau_c, c0


def main():
    ref = {}
    with open(PER_RUN_CSV, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ref[(row["system"], row["source"])] = row

    hdr = (f"{'system':>8} {'source':>9} | {'rg_mean':>12} {'rg_mean^2':>12} "
           f"{'csv rg2_mean':>13} {'true<Rg^2>':>12} {'gap%':>7} | "
           f"{'sd(frame)':>10} {'naive sem':>10} {'blk sem':>10} {'ac sem':>10} "
           f"{'tau_c(fr)':>9}")
    print(hdr)
    print("-" * len(hdr))

    for sysname in SYSTEMS:
        for src in SOURCES:
            run = BASE / src / sysname
            natoms, atom_type, atom_mol, mass_of_type = parse_topology(run / "filled.data")
            r = scan_rg(run / "dump.lammpstrj", natoms, atom_type, atom_mol,
                        mass_of_type, 1_500_000, 2_000_000, verbose=False)
            fm = r["per_frame_mean"]
            row = ref[(sysname, src)]

            csv_rg2 = float(row["rg2_mean"])
            sq = r["rg_mean"] ** 2
            # genuine ensemble <Rg^2>:  average over chains and frames of Rg^2.
            # recompute it from the per-frame Rg list via the identity
            # <Rg^2> = <Rg>_chain^2 + var_chain  (per frame), then average frames.
            # var_chain per frame = (std_ddof1)^2 * (n-1)/n ... do it exactly:
            # we only stored mean/std per frame, so use the unbiased correction:
            # E[Rg^2] = mean^2 + (n-1)/n * std_ddof1^2
            n = r["n_chain"]
            true_rg2 = statistics.fmean([m * m + (n - 1) / n * s * s
                                         for m, s in zip(r["per_frame_mean"], r["per_frame_std"])])
            gap = 100.0 * (csv_rg2 - true_rg2) / true_rg2

            sd_frame = statistics.stdev(fm)
            naive_sem = sd_frame / math.sqrt(len(fm))
            # script's block sem (10 frames per block)
            nb = len(fm) // 10
            blocks = [statistics.fmean(fm[i:i + 10]) for i in range(0, nb * 10, 10)]
            blk_sem = statistics.stdev(blocks) / math.sqrt(len(blocks))
            ac_sem, tau_c, c0 = autocorr_sem(fm)

            print(f"{sysname:>8} {src:>9} | {r['rg_mean']:12.6f} {sq:12.6f} "
                  f"{csv_rg2:13.6f} {true_rg2:12.6f} {gap:7.3f} | "
                  f"{sd_frame:10.6f} {naive_sem:10.6f} {blk_sem:10.6f} {ac_sem:10.6f} "
                  f"{tau_c:9.2f}")
            print(f"{'':>18}   csv rg_sem_block = {float(row['rg_sem_block']):.8f}"
                  f"   csv rg_std = {float(row['rg_std']):.6f}  "
                  f"(across-chain spread, NOT an error on the mean)")


if __name__ == "__main__":
    main()
