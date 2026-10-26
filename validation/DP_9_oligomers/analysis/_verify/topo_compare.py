"""Verifier: topology comparability audit between ligpargen and mappingff runs."""
import sys, os, re
from pathlib import Path
import numpy as np

BASE = Path(r"C:\schoolwork\Projects\mappingff\validation\longer")
SYSTEMS = ("9_peo", "9_pmma", "9_ps", "9_pva")
SOURCES = ("ligpargen", "mappingff")


def read_data(path):
    """Parse a LAMMPS data file into structured dict."""
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    out = {"header": lines[:20]}
    counts = {}
    i = 0
    n = len(lines)
    # header counts
    while i < n:
        s = lines[i].strip()
        m = re.match(r"^(\d+)\s+(\w+)\s*$", s)
        if m and m.group(2) in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
            counts[m.group(2)] = int(m.group(1))
        if re.match(r"^[\d\.\-]+\s+[\d\.\-]+\s+xlo xhi", s):
            break
        i += 1
    out["counts"] = counts
    # bounds
    bounds = []
    while i < n:
        s = lines[i].strip()
        m = re.match(r"^([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([xyz])lo ([xyz])hi", s)
        if m:
            bounds.append((m.group(3), float(m.group(1)), float(m.group(2))))
        else:
            break
        i += 1
    out["bounds"] = bounds

    def grab_section(header_name):
        """Return list of rows (as lists of floats/ints) for a section."""
        nonlocal i, n
        while i < n:
            if lines[i].strip().startswith(header_name):
                break
            i += 1
        else:
            return []
        i += 1
        # skip comment/blank lines
        while i < n and (not lines[i].strip() or lines[i].strip().startswith("#")):
            i += 1
        rows = []
        while i < n and lines[i].strip():
            rows.append([float(x) for x in lines[i].split()])
            i += 1
        return rows

    masses = grab_section("Masses")
    out["mass_of_type"] = {int(r[0]): float(r[1]) for r in masses}
    atoms = grab_section("Atoms")
    out["atoms"] = atoms   # id mol type q x y z
    bonds = grab_section("Bonds")
    out["bonds"] = bonds   # id type a1 a2
    angles = grab_section("Angles")
    out["angles"] = angles
    dihedrals = grab_section("Dihedrals")
    out["dihedrals"] = dihedrals
    impropers = grab_section("Impropers")
    out["impropers"] = impropers
    return out


def summarize(path):
    d = read_data(path)
    atoms = d["atoms"]
    ids = np.array([int(a[0]) for a in atoms])
    mol = np.array([int(a[1]) for a in atoms])
    typ = np.array([int(a[2]) for a in atoms])
    q = np.array([float(a[3]) for a in atoms])
    xyz = np.array([[a[4], a[5], a[6]] for a in atoms])
    mtype = d["mass_of_type"]
    mass = np.array([mtype.get(int(t), np.nan) for t in typ])
    bonds = d["bonds"]
    btype = np.array([int(b[1]) for b in bonds]) if bonds else np.array([])
    ba = np.array([int(b[2]) for b in bonds]) if bonds else np.array([])
    bb = np.array([int(b[3]) for b in bonds]) if bonds else np.array([])
    ang = d["angles"]
    anh = np.array([int(a[2]) for a in ang]) if ang else np.array([])
    anh2 = np.array([int(a[3]) for a in ang]) if ang else np.array([])
    anh3 = np.array([int(a[4]) for a in ang]) if ang else np.array([])
    dih = d["dihedrals"]
    d1 = np.array([int(x[2]) for x in dih]) if dih else np.array([])
    d2 = np.array([int(x[3]) for x in dih]) if dih else np.array([])
    d3 = np.array([int(x[4]) for x in dih]) if dih else np.array([])
    d4 = np.array([int(x[5]) for x in dih]) if dih else np.array([])
    imp = d["impropers"]
    return dict(d=d, ids=ids, mol=mol, typ=typ, q=q, xyz=xyz, mass=mass,
                mass_of_type=mtype, btype=btype, ba=ba, bb=bb,
                anh=(anh, anh2, anh3), dih=(d1, d2, d3, d4), imp=imp)


def bond_key_set(btype, ba, bb):
    """Canonical set of (type, min(a,b), max(a,b)) bonds."""
    s = set()
    for t, a, b in zip(btype, ba, bb):
        lo, hi = (a, b) if a < b else (b, a)
        s.add((int(t), lo, hi))
    return s


def topo_key_set(btype, ba, bb):
    """Canonical set of (type, sorted 4 atom ids) topology entries."""
    s = set()
    for t, a, b in zip(btype, ba, bb):
        s.add((int(t), tuple(sorted((a, b)))))
    return s


def main():
    for sysname in SYSTEMS:
        print("=" * 100)
        print(f"SYSTEM {sysname}")
        print("=" * 100)
        S = {}
        for src in SOURCES:
            S[src] = summarize(BASE / src / sysname / "filled.data")
        a, b = S["ligpargen"], S["mappingff"]

        # 1. counts
        print("-- header counts --")
        ca, cb = a["d"]["counts"], b["d"]["counts"]
        for k in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
            va, vb = ca.get(k, 0), cb.get(k, 0)
            print(f"   {k:10s} ligpargen={va:8d}  mappingff={vb:8d}  "
                  f"{'SAME' if va == vb else '*** DIFFER ***'}")

        # atoms per molecule
        for tag, r in (("ligpargen", a), ("mappingff", b)):
            mols, counts = np.unique(r["mol"], return_counts=True)
            print(f"   [{tag}] n_mol={len(mols)} mol_range=({mols.min()},{mols.max()}) "
                  f"atoms_per_mol: unique={sorted(set(counts.tolist()))} "
                  f"total_mass={r['mass'].sum():.6f} (amu) "
                  f"mass/mol={r['mass'].sum()/len(mols):.6f}")
            mols_set = set(mols.tolist())
            contig = (mols_set == set(range(1, len(mols) + 1)))
            print(f"   [{tag}] mol ids contiguous 1..N: {contig}")

        # 2. vectors identical?
        print("-- per-atom vectors --")
        print(f"   atom ids identical: {np.array_equal(a['ids'], b['ids'])} "
              f"(len {len(a['ids'])} vs {len(b['ids'])})")
        print(f"   mol vector identical: {np.array_equal(a['mol'], b['mol'])}")
        print(f"   TYPE vector identical: {np.array_equal(a['typ'], b['typ'])}  "
              f"(n_types ligpargen={len(a['mass_of_type'])}, mappingff={len(b['mass_of_type'])})")
        if not np.array_equal(a["typ"], b["typ"]):
            nd = int((a["typ"] != b["typ"]).sum())
            print(f"      -> {nd} atoms have DIFFERENT type index between the two runs")
        print(f"   MASS vector identical: {np.array_equal(a['mass'], b['mass'])}")
        if not np.array_equal(a["mass"], b["mass"]):
            diff = ~np.isclose(a["mass"], b["mass"], rtol=0, atol=1e-9)
            print(f"      -> {int(diff.sum())} atoms have DIFFERENT mass; "
                  f"mass delta (mappingff-ligpargen) uniq: "
                  f"{sorted(set(np.round(a['mass'][diff]-b['mass'][diff],6).tolist()))}")
            ex = np.where(diff)[0][:5]
            for e in ex:
                print(f"         atom {e+1}: ligpargen typ={a['typ'][e]} m={a['mass'][e]} | "
                      f"mappingff typ={b['typ'][e]} m={b['mass'][e]}")
        print(f"   mass sum: ligpargen={a['mass'].sum():.6f}  mappingff={b['mass'].sum():.6f}  "
              f"delta={b['mass'].sum()-a['mass'].sum():.6f}")

        # 3. bonds
        print("-- bonds --")
        print(f"   n_bonds: {len(a['btype'])} vs {len(b['btype'])}")
        ka, kb = bond_key_set(a["btype"], a["ba"], a["bb"]), bond_key_set(b["btype"], b["ba"], b["bb"])
        print(f"   bond (type,a,b) multiset identical: {ka == kb}")
        if ka != kb:
            onlya = ka - kb
            onlyb = kb - ka
            print(f"      only in ligpargen: {len(onlya)} e.g. {sorted(onlya)[:5]}")
            print(f"      only in mappingff : {len(onlyb)} e.g. {sorted(onlyb)[:5]}")
            # connectivity only (ignore type)
            cna = bond_key_set(np.zeros_like(a["btype"]), a["ba"], a["bb"])
            cnb = bond_key_set(np.zeros_like(b["btype"]), b["ba"], b["bb"])
            print(f"      connectivity-only (a,b pairs) identical: {cna == cnb}")
            if cna != cnb:
                print(f"      only in ligpargen conn: {len(cna-cnb)} e.g. {sorted(cna-cnb)[:5]}")
                print(f"      only in mappingff conn : {len(cnb-cna)} e.g. {sorted(cnb-cna)[:5]}")
        else:
            cna = bond_key_set(np.zeros_like(a["btype"]), a["ba"], a["bb"])
            cnb = bond_key_set(np.zeros_like(b["btype"]), b["ba"], b["bb"])
            print(f"   connectivity-only (a,b pairs) identical: {cna == cnb}")

        # bond type -> length mapping (chemistry)
        print("-- bond type sets --")
        print(f"   ligpargen bond types: {sorted(set(a['btype'].tolist()))}")
        print(f"   mappingff bond types: {sorted(set(b['btype'].tolist()))}")

        # 4. box
        print("-- initial box (filled.data) --")
        for tag, r in (("ligpargen", a), ("mappingff", b)):
            bs = r["d"]["bounds"]
            txt = "  ".join(f"{ax}:[{lo:.6f},{hi:.6f}] L={hi-lo:.6f}" for ax, lo, hi in bs)
            vol = 1.0
            for ax, lo, hi in bs:
                vol *= (hi - lo)
            print(f"   [{tag}] {txt}   V={vol:.4f} A^3")
        ba_, bb_ = a["d"]["bounds"], b["d"]["bounds"]
        same_box = len(ba_) == len(bb_) and all(
            abs(x[1]-y[1]) < 1e-9 and abs(x[2]-y[2]) < 1e-9 for x, y in zip(ba_, bb_))
        print(f"   box identical: {same_box}")

        # density implied by mass & box
        for tag, r in (("ligpargen", a), ("mappingff", b)):
            vol = 1.0
            for ax, lo, hi in r["d"]["bounds"]:
                vol *= (hi - lo)
            m = r["mass"].sum()
            rho = m / 6.02214076e23 / (vol * 1e-24)
            print(f"   [{tag}] implied initial density from mass+box = {rho:.6f} g/cm^3")
        print()


if __name__ == "__main__":
    main()
