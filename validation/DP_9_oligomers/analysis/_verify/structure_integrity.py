"""
Structural-integrity audit of the 8 melts (independent of analyze_density_rg.py).

  * bond-length distribution (1-2 distances) in the production window
  * minimum INTER-chain and INTRA-chain distances using a CORRECT minimum
    image (the first pass forgot the minimum image on the raw difference)
  * identity of the closest inter-chain contact (element pair)
  * end-to-end distance averaged over ALL chains
  * per-chain Rg distribution (outliers / bimodality / broken chains)
  * check that no chain is split by the periodic boundary
"""
from __future__ import annotations

import json
import mmap
import os
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
RUNS = [(src, s) for s in ("9_peo", "9_pmma", "9_ps", "9_pva")
        for src in ("ligpargen", "mappingff")]
FRAME_MARK = b"ITEM: TIMESTEP\n"
PROBE = (1_500_000, 1_750_000, 2_000_000)


def parse_data(path: Path, np=np, re=re):
    lines = path.read_text(errors="replace").splitlines()
    sec = {}
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        m = re.match(r"^(?:(\d+)\s+)?([A-Za-z][A-Za-z ]*?)\s*(#.*)?$", line)
        if m and m.group(2).strip() in ("Masses", "Atoms", "Bonds",
                                        "Pair Coeffs"):
            s = m.group(2).strip()
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            rows, k = [], j
            while k < n and lines[k].strip():
                rows.append(lines[k].split())
                k += 1
            sec[s] = rows
            i = k
            continue
        i += 1
    mot = {int(r[0]): float(r[1]) for r in sec["Masses"]}
    at = sec["Atoms"]
    aid = np.array([int(r[0]) for r in at])
    mol = np.array([int(r[1]) for r in at])
    typ = np.array([int(r[2]) for r in at])
    o = np.argsort(aid)
    aid, mol, typ = aid[o], mol[o], typ[o]
    mass = np.array([mot[t] for t in typ])
    b1 = np.array([int(r[2]) for r in sec["Bonds"]]) - 1
    b2 = np.array([int(r[3]) for r in sec["Bonds"]]) - 1
    pair = {}
    for r in sec.get("Pair Coeffs", []):
        pair[int(r[0])] = (float(r[1]), float(r[2]))
    return aid, mol, typ, mass, b1, b2, pair


def read_frames(path: Path, os=os, mmap=mmap):
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        starts, pos = [], 0
        while True:
            hit = mm.find(FRAME_MARK, pos)
            if hit < 0:
                break
            starts.append(hit)
            pos = hit + len(FRAME_MARK)
        starts.append(size)
        for k in range(len(starts) - 1):
            beg = starts[k] + len(FRAME_MARK)
            end = mm.find(b"\n", beg)
            yield int(mm[beg:end]), mm[starts[k]:starts[k + 1]]
        mm.close()


def main(np=np, json=json, Path=Path):
    out = {}
    for src, sysname in RUNS:
        run = BASE / src / sysname
        aid, mol, typ, mass, b1, b2, pair = parse_data(run / "filled.data")
        natoms = len(aid)
        heavy = mass >= 12.0
        el = np.where(np.abs(mass - 12.011) < 1e-3, "C",
                      np.where(np.abs(mass - 15.999) < 1e-3, "O", "H"))
        sigma = np.array([pair[t][0] if t in pair else np.nan
                          for t in typ])
        eps = np.array([pair[t][1] if t in pair else np.nan for t in typ])
        uniq = np.unique(mol)
        chain = np.array([{m: k for k, m in enumerate(uniq.tolist())}[m]
                          for m in mol.tolist()])
        n_chain = len(uniq)

        bondmin = []
        inter = []
        intra = []
        closest = []
        rg_all = []
        e2e_all = []
        wrapstat = []
        for step, blob in read_frames(run / "dump.lammpstrj"):
            if step not in PROBE:
                continue
            lines = blob.split(b"\n")
            nat = int(lines[3])
            bx = []
            for d in (5, 6, 7):
                p = lines[d].split()
                bx.append((float(p[0]), float(p[1])))
            L = np.array([b[1] - b[0] for b in bx])
            cols = [c.decode() for c in lines[8].split()[2:]]
            arr = np.fromstring(b" ".join(lines[9:9 + nat]), sep=" ")
            arr = arr.reshape(nat, len(cols))
            ix = (cols.index("xu"), cols.index("yu"), cols.index("zu"))
            xu = np.stack([arr[:, ix[0]], arr[:, ix[1]], arr[:, ix[2]]], axis=1)

            # --- bond lengths (minimum image) ---
            d = xu[b1] - xu[b2]
            d -= L * np.round(d / L)
            bl = np.linalg.norm(d, axis=1)
            bondmin.append([float(bl.min()), float(bl.max()),
                            float(bl.mean())])

            # --- wrapped heavy positions ---
            pos = xu[heavy].copy()
            for k in range(3):
                pos[:, k] = (pos[:, k] - bx[k][0]) % L[k]
            ch_h = chain[heavy]
            tree = cKDTree(pos, boxsize=L)
            pairs = tree.query_pairs(r=6.0, output_type="ndarray")
            dd = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            dd -= L * np.round(dd / L)
            dm = np.linalg.norm(dd, axis=1)
            same = ch_h[pairs[:, 0]] == ch_h[pairs[:, 1]]
            inter.append(float(dm[~same].min()))
            intra.append(float(dm[same].min()))
            k = int(np.argmin(np.where(same, 1e9, dm)))
            ia, ib = pairs[k]
            heavy_idx = np.nonzero(heavy)[0]
            gi, gj = heavy_idx[ia], heavy_idx[ib]
            closest.append(dict(dist=float(dm[k]), el1=str(el[gi]),
                                el2=str(el[gj]), mol1=int(mol[gi]),
                                mol2=int(mol[gj]),
                                sigma1=float(sigma[gi]), sigma2=float(sigma[gj]),
                                eps1=float(eps[gi]), eps2=float(eps[gj])))

            # --- per-chain Rg + e2e ---
            wsum = np.bincount(chain, weights=mass, minlength=n_chain)
            com = np.empty((n_chain, 3))
            for kk in range(3):
                com[:, kk] = np.bincount(chain, weights=mass * xu[:, kk],
                                         minlength=n_chain)
            com /= wsum[:, None]
            df = xu - com[chain]
            r2 = np.einsum("ij,ij->i", df, df)
            rg = np.sqrt(np.bincount(chain, weights=mass * r2,
                                     minlength=n_chain) / wsum)
            rg_all.append(rg)
            # e2e over the first/last heavy atom of each chain in id order
            ends = []
            for c in range(n_chain):
                idxc = np.where(chain == c)[0]
                h = idxc[heavy[idxc]]
                ends.append(np.linalg.norm(xu[h[-1]] - xu[h[0]]))
            e2e_all.append(np.array(ends))

            # --- does any chain straddle the box? (unwrapped span) ---
            span = np.array([xu[chain == c].max(axis=0) - xu[chain == c].min(axis=0)
                             for c in range(n_chain)])
            wrapstat.append([float((span > L * 0.5).any(axis=1).mean()),
                             float(span.max())])

        rg_all = np.array(rg_all)
        e2e_all = np.array(e2e_all)
        res = dict(
            system=sysname, source=src,
            bond_min_max_mean=bondmin,
            min_inter_chain=inter, min_intra_chain=intra,
            closest_contact=closest,
            rg_pooled_mean=float(rg_all.mean()),
            rg_pooled_std=float(rg_all.std(ddof=1)),
            rg_per_chain_min=float(rg_all.min()),
            rg_per_chain_max=float(rg_all.max()),
            rg_median=float(np.median(rg_all)),
            rg_skew=float(((rg_all - rg_all.mean()) ** 3).mean() /
                          rg_all.std() ** 3),
            e2e_mean=float(e2e_all.mean()), e2e_std=float(e2e_all.std(ddof=1)),
            e2e_max=float(e2e_all.max()),
            frac_chains_spanning_gt_half_box=wrapstat,
        )
        out[f"{src}/{sysname}"] = res
        print(f"[{src}/{sysname}]")
        print(f"   bond len (min/max/mean per probe): "
              f"{[f'{a:.3f}/{b:.3f}/{c:.3f}' for a, b, c in bondmin]}")
        print(f"   min INTER-chain heavy dist per probe: "
              f"{[f'{v:.3f}' for v in inter]}")
        print(f"   min INTRA-chain heavy dist per probe: "
              f"{[f'{v:.3f}' for v in intra]}")
        print(f"   closest inter-chain contact: {closest[0]}")
        print(f"   Rg pooled mean={res['rg_pooled_mean']:.4f} "
              f"sd={res['rg_pooled_std']:.4f} median={res['rg_median']:.4f} "
              f"min={res['rg_per_chain_min']:.3f} max={res['rg_per_chain_max']:.3f} "
              f"skew={res['rg_skew']:+.3f}")
        print(f"   e2e (first-last heavy of chain) mean={res['e2e_mean']:.2f} "
              f"sd={res['e2e_std']:.2f} max={res['e2e_max']:.2f}")
        print(f"   frac chains spanning >L/2: "
              f"{[f'{a:.3f}' for a, b in wrapstat]}; max span="
              f"{max(b for a, b in wrapstat):.1f} A")
        print()

    (Path(__file__).parent / "structure_integrity.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
