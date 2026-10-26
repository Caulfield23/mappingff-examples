#!/usr/bin/env python3
"""
compare_lammps_charges.py

Compare atomic partial charges between a LigParGen .lmp file and a MappingFF
.data file describing the same molecule and the same geometry, and report the
charge MAE (mean absolute error of the per-atom charge difference).

Batch use (every molecule under a validation root):
    python compare_lammps_charges.py --root .
    python compare_lammps_charges.py --root ../longer --glob '9_*'
The root is expected to contain a ligpargen/**/<mol>.lmp file and a
mappingff/**/<mol>.data file for each molecule; the search is recursive, so
generated files may sit in a build/ subdirectory. --glob restricts which
molecules take part, which matters when the directories also hold other files
(5_x.data leftovers, filled/final/minimized.data from box filling). A summary
is written to <root>/charge_mae_summary.csv and <root>/charge_mae_summary.md.

Single pair:
    python compare_lammps_charges.py ligpargen/5_peo/5_peo.lmp mappingff/5_peo/5_peo.data

Assumptions
-----------
1. Both files contain the same molecule (elemental composition and bond
   connectivity must match).
2. Atom style is compatible with:
       atom-ID molecule-ID atom-type charge x y z
3. Masses can be used to infer elements.

The geometries do NOT have to be identical. LigParGen re-optimises the input
geometry while MappingFF keeps it, so atom correspondence is established from
the molecular graph (colour refinement + connectivity-preserving search) and
only refined with coordinates when both files share a Cartesian frame. Atom
IDs and atom ordering may differ between the files. The summary reports the
superposition RMSD alongside the MAE so differing conformers stay visible.

Dependencies:
    numpy
    scipy (optional, for the optimal atom assignment)

Install with:
    pip install numpy scipy
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only on scipy-less installs
    HAVE_SCIPY = False


# Common elemental masses sufficient for typical organic-polymer systems.
ELEMENT_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
    "Si": 28.085,
}


SECTION_NAMES = {
    "Masses",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "Atoms",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Velocities",
}


def strip_comment(line):
    return line.split("#", 1)[0].strip()


def identify_section(line):
    s = line.strip()
    for name in SECTION_NAMES:
        if s == name or s.startswith(name + " "):
            return name
    return None


def nearest_element(mass, tolerance=0.25):
    best = min(ELEMENT_MASSES.items(), key=lambda kv: abs(kv[1] - mass))
    elem, ref_mass = best
    if abs(ref_mass - mass) > tolerance:
        raise ValueError(
            f"Could not confidently infer element from mass {mass:.6f}; "
            f"nearest is {elem} ({ref_mass:.6f})."
        )
    return elem


def parse_lammps_data(path):
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    masses = {}
    atoms = {}
    bonds = []

    section = None

    for raw in lines:
        sec = identify_section(raw)
        if sec is not None:
            section = sec
            continue

        line = strip_comment(raw)
        if not line:
            continue

        parts = line.split()

        if section == "Masses":
            # type mass
            if len(parts) >= 2:
                try:
                    atype = int(parts[0])
                    mass = float(parts[1])
                except ValueError:
                    continue
                masses[atype] = mass

        elif section == "Atoms":
            # expected: id mol type charge x y z [ix iy iz...]
            if len(parts) >= 7:
                try:
                    atom_id = int(parts[0])
                    mol_id = int(parts[1])
                    atom_type = int(parts[2])
                    charge = float(parts[3])
                    x, y, z = map(float, parts[4:7])
                except ValueError:
                    continue
                atoms[atom_id] = {
                    "id": atom_id,
                    "mol": mol_id,
                    "type": atom_type,
                    "charge": charge,
                    "coord": np.array([x, y, z], dtype=float),
                }

        elif section == "Bonds":
            # id bond_type atom1 atom2
            if len(parts) >= 4:
                try:
                    bond_id = int(parts[0])
                    bond_type = int(parts[1])
                    a1 = int(parts[2])
                    a2 = int(parts[3])
                except ValueError:
                    continue
                bonds.append((bond_id, bond_type, a1, a2))

    if not atoms:
        raise ValueError(f"No atoms parsed from {path}")
    if not masses:
        raise ValueError(f"No masses parsed from {path}")
    if not bonds:
        raise ValueError(f"No bonds parsed from {path}")

    for a in atoms.values():
        if a["type"] not in masses:
            raise ValueError(
                f"Atom {a['id']} uses atom type {a['type']}, "
                f"but that type is absent from Masses."
            )
        a["mass"] = masses[a["type"]]
        a["element"] = nearest_element(a["mass"])

    adjacency = {aid: set() for aid in atoms}
    for _, _, a1, a2 in bonds:
        if a1 not in atoms or a2 not in atoms:
            raise ValueError(f"Bond references missing atom: {a1}-{a2}")
        adjacency[a1].add(a2)
        adjacency[a2].add(a1)

    for aid, a in atoms.items():
        a["degree"] = len(adjacency[aid])
        a["neighbor_elements"] = tuple(
            sorted(atoms[n]["element"] for n in adjacency[aid])
        )

    return {
        "path": path,
        "masses": masses,
        "atoms": atoms,
        "bonds": bonds,
        "adjacency": adjacency,
    }


def element_counts(data):
    counts = {}
    for a in data["atoms"].values():
        counts[a["element"]] = counts.get(a["element"], 0) + 1
    return counts


def refine_colors(data):
    """1-dimensional Weisfeiler-Lehman colour refinement on the bond graph.

    Each atom is coloured by (its own colour, the sorted multiset of its
    neighbours' colours), iterated until the partition stops getting finer.
    The labels are canonical: two isomorphic graphs get identical colour
    assignments, so colours can be compared directly across two files.
    """
    colors = {aid: a["element"] for aid, a in data["atoms"].items()}

    def partition(colors):
        labels = {}
        for aid in sorted(colors):
            labels.setdefault(colors[aid], len(labels))
        return tuple(sorted(labels[colors[aid]] for aid in colors))

    prev = partition(colors)
    for _ in range(len(colors)):
        raw = {
            aid: (colors[aid],
                  tuple(sorted(colors[nb] for nb in data["adjacency"][aid])))
            for aid in colors
        }
        order = {v: i for i, v in enumerate(sorted(set(raw.values())))}
        colors = {aid: order[v] for aid, v in raw.items()}
        cur = partition(colors)
        if cur == prev:
            break
        prev = cur

    return colors


def color_classes(colors):
    classes = {}
    for aid, c in colors.items():
        classes.setdefault(c, []).append(aid)
    for members in classes.values():
        members.sort()
    return classes


def nearest_same_color_distance(ref, test, colors_r, colors_t):
    """Smallest coordinate distance from each reference atom to any atom of
    the same topological colour. Near-zero means the two files share a
    Cartesian frame; large values mean the geometries differ."""
    classes_t = color_classes(colors_t)
    tcoords = {
        aid: test["atoms"][aid]["coord"]
        for members in classes_t.values()
        for aid in members
    }

    distances = []
    for rid in sorted(ref["atoms"]):
        p = ref["atoms"][rid]["coord"]
        distances.append(min(
            float(np.linalg.norm(p - q)) for q in tcoords.values()
        ))
    return np.array(distances)


def build_cost_matrix(ref, test, colors_r, colors_t, coord_tol):
    ref_ids = sorted(ref["atoms"])
    test_ids = sorted(test["atoms"])

    if len(ref_ids) != len(test_ids):
        raise ValueError(
            f"Atom count mismatch: {len(ref_ids)} vs {len(test_ids)}"
        )

    big = 1.0e9
    cost = np.full((len(ref_ids), len(test_ids)), big, dtype=float)

    for i, rid in enumerate(ref_ids):
        ra = ref["atoms"][rid]
        for j, tid in enumerate(test_ids):
            if colors_r[rid] != colors_t[tid]:
                continue

            d = np.linalg.norm(ra["coord"] - test["atoms"][tid]["coord"])
            if d <= coord_tol:
                cost[i, j] = d

    return ref_ids, test_ids, cost


def greedy_assignment(cost):
    """Fallback assignment when scipy is unavailable: nearest unused column."""
    n_rows, n_cols = cost.shape
    order = sorted(range(n_rows), key=lambda r: float(cost[r].min()))
    used = set()
    pairs = []
    for r in order:
        row = cost[r]
        for c in sorted(range(n_cols), key=lambda c: row[c]):
            if c not in used:
                used.add(c)
                pairs.append((r, c))
                break
    return pairs


def connectivity_is_consistent(ref, test, mapping):
    adj_r = ref["adjacency"]
    adj_t = test["adjacency"]
    for rid, tid in mapping.items():
        for rnb in adj_r[rid]:
            if mapping[rnb] not in adj_t[tid]:
                return False
    return True


def match_by_geometry(ref, test, colors_r, colors_t, coord_tol):
    """Match atoms by colour + Cartesian proximity (optimal assignment)."""
    ref_ids, test_ids, cost = build_cost_matrix(
        ref, test, colors_r, colors_t, coord_tol
    )

    if HAVE_SCIPY:
        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows, cols))
    else:
        pairs = greedy_assignment(cost)

    mapping = {}
    bad = []
    for r, c in pairs:
        rid, tid = ref_ids[r], test_ids[c]
        if cost[r, c] >= 1.0e8:
            bad.append(rid)
        else:
            mapping[rid] = tid

    if bad or len(mapping) != len(ref_ids):
        raise ValueError(
            "Coordinate-based atom matching failed; "
            f"{len(bad)} reference atom(s) had no same-geometry counterpart "
            f"(first few: {bad[:10]})."
        )

    if not connectivity_is_consistent(ref, test, mapping):
        raise ValueError(
            "Coordinate-based matching produced a map that is inconsistent "
            "with the bond connectivity."
        )

    return mapping


def match_by_topology(ref, test, colors_r, colors_t, max_steps=500000):
    """Match atoms purely by molecular graph isomorphism.

    Colour refinement gives the candidate sets; a backtracking search then
    picks the bijection that preserves the bond graph. Used when the two
    files do not share a Cartesian frame (e.g. LigParGen re-optimises the
    geometry while MappingFF keeps the input coordinates).
    """
    classes_r = color_classes(colors_r)
    classes_t = color_classes(colors_t)

    for color, members in classes_r.items():
        if len(members) != len(classes_t.get(color, ())):
            raise ValueError(
                "The two files are not the same molecule: topological class "
                f"{color} holds {len(members)} reference atom(s) but "
                f"{len(classes_t.get(color, ()))} test atom(s)."
            )

    cands = {rid: classes_t[c] for rid, c in colors_r.items()}

    adj_r = ref["adjacency"]
    adj_t = test["adjacency"]

    # Most constrained atoms first: fewest candidates, then highest degree.
    order = sorted(
        colors_r, key=lambda a: (len(cands[a]), -len(adj_r[a]), a)
    )

    mapping = {}
    reverse = {}
    steps = [0]

    def feasible(rid, tid):
        if tid in reverse:
            return False
        if len(adj_t[tid]) != len(adj_r[rid]):
            return False
        for rnb in adj_r[rid]:
            tnb = mapping.get(rnb)
            if tnb is not None and tnb not in adj_t[tid]:
                return False
        for tnb in adj_t[tid]:
            rnb = reverse.get(tnb)
            if rnb is not None and rnb not in adj_r[rid]:
                return False
        return True

    def search(i):
        if i == len(order):
            return True
        rid = order[i]
        for tid in cands[rid]:
            steps[0] += 1
            if steps[0] > max_steps:
                raise ValueError(
                    "Atom mapping search exceeded the step budget; the two "
                    "molecules are probably not the same."
                )
            if not feasible(rid, tid):
                continue
            mapping[rid] = tid
            reverse[tid] = rid
            if search(i + 1):
                return True
            del mapping[rid]
            del reverse[tid]
        return False

    if not search(0):
        raise ValueError(
            "No bond-connectivity-preserving atom mapping exists between "
            "the two files; they do not appear to be the same molecule."
        )

    return mapping


def kabsch_rmsd(ref, test, mapping):
    """RMSD of the test coordinates onto the reference after optimal
    rigid-body superposition. Purely diagnostic: it reports how different
    the two geometries actually are."""
    ref_pts = np.array(
        [ref["atoms"][rid]["coord"] for rid in sorted(mapping)]
    )
    test_pts = np.array(
        [test["atoms"][mapping[rid]]["coord"] for rid in sorted(mapping)]
    )

    ref_c = ref_pts - ref_pts.mean(axis=0)
    test_c = test_pts - test_pts.mean(axis=0)

    cov = test_c.T @ ref_c
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rot = vt.T @ correction @ u.T

    fitted = test_c @ rot.T
    return float(np.sqrt(np.mean(np.sum((fitted - ref_c) ** 2, axis=1))))


def establish_mapping(ref, test, coord_tol, method="auto"):
    """Return (mapping, method_used, geometry_rmsd).

    method 'geometry'  - colour + Cartesian proximity (needs a shared frame)
    method 'topology'  - bond graph isomorphism only (frame-independent)
    method 'auto'      - geometry when the files share a frame, else topology
    """
    colors_r = refine_colors(ref)
    colors_t = refine_colors(test)

    if method in ("auto", "geometry"):
        distances = nearest_same_color_distance(
            ref, test, colors_r, colors_t
        )
        shared_frame = bool(len(distances)) and distances.max() <= coord_tol

        if method == "geometry" or shared_frame:
            try:
                mapping = match_by_geometry(
                    ref, test, colors_r, colors_t, coord_tol
                )
                return mapping, "geometry", kabsch_rmsd(ref, test, mapping)
            except ValueError:
                if method == "geometry":
                    raise
        # else: fall through to topology matching

    mapping = match_by_topology(ref, test, colors_r, colors_t)
    return mapping, "topology", kabsch_rmsd(ref, test, mapping)


def compare_charges(ref, test, mapping):
    rows = []

    for rid in sorted(mapping):
        tid = mapping[rid]
        ra = ref["atoms"][rid]
        ta = test["atoms"][tid]

        dq = ta["charge"] - ra["charge"]

        rows.append({
            "reference_atom_id": rid,
            "test_atom_id": tid,
            "element": ra["element"],
            "reference_atom_type": ra["type"],
            "test_atom_type": ta["type"],
            "reference_charge_e": ra["charge"],
            "test_charge_e": ta["charge"],
            "delta_q_e": dq,
            "abs_delta_q_e": abs(dq),
        })

    diffs = np.array([r["delta_q_e"] for r in rows], dtype=float)

    stats = {
        "n_atoms": len(rows),
        "mae": float(np.mean(np.abs(diffs))),
        "ref_total_charge": float(
            sum(a["charge"] for a in ref["atoms"].values())
        ),
        "test_total_charge": float(
            sum(a["charge"] for a in test["atoms"].values())
        ),
    }

    return rows, stats


def write_atom_csv(path, rows):
    fieldnames = [
        "reference_atom_id",
        "test_atom_id",
        "element",
        "reference_atom_type",
        "test_atom_type",
        "reference_charge_e",
        "test_charge_e",
        "delta_q_e",
        "abs_delta_q_e",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pair(name, reference, test, coord_tol, method="auto", verbose=True):
    """Compare one reference/test pair. Returns a summary record."""
    record = {
        "molecule": name,
        "reference_file": str(reference),
        "test_file": str(test),
        "n_atoms": 0,
        "charge_mae_e": "",
        "ref_total_charge_e": "",
        "test_total_charge_e": "",
        "geometry_rmsd_a": "",
        "match_method": "",
        "status": "ok",
    }

    try:
        ref = parse_lammps_data(reference)
        tst = parse_lammps_data(test)

        if element_counts(ref) != element_counts(tst):
            raise ValueError(
                "Elemental compositions differ: "
                f"{element_counts(ref)} vs {element_counts(tst)}"
            )

        mapping, used, rmsd = establish_mapping(
            ref, tst, coord_tol, method=method
        )
        rows, stats = compare_charges(ref, tst, mapping)

        record["n_atoms"] = stats["n_atoms"]
        record["charge_mae_e"] = f"{stats['mae']:.8f}"
        record["ref_total_charge_e"] = f"{stats['ref_total_charge']:.8f}"
        record["test_total_charge_e"] = f"{stats['test_total_charge']:.8f}"
        record["geometry_rmsd_a"] = f"{rmsd:.4f}"
        record["match_method"] = used

        if verbose:
            print(f"{name}")
            print(f"  reference      : {reference}")
            print(f"  test           : {test}")
            print(f"  atoms          : {stats['n_atoms']}")
            print(f"  atom matching  : {used}")
            print(f"  geometry RMSD  : {rmsd:.4f} A")
            print(f"  charge MAE     : {stats['mae']:.8f} e")
            print()

        return record, rows

    except Exception as exc:
        record["status"] = f"failed: {exc}"
        if verbose:
            print(f"{name}")
            print(f"  reference      : {reference}")
            print(f"  test           : {test}")
            print(f"  ERROR          : {exc}")
            print()
        return record, None


def discover_pairs(root, pattern="*"):
    """Pair every <root>/ligpargen/**/<pattern>.lmp with the same-named
    <root>/mappingff/**/<pattern>.data.

    The search is recursive so that suites which keep the generated files in
    a build/ subdirectory (e.g. validation/longer) work unchanged. `pattern`
    restricts which molecules take part, e.g. pattern='9_*' to compare only
    the 9_x.lmp / 9_x.data pairs.
    """
    root = Path(root)

    tests = {}
    for test in sorted((root / "mappingff").rglob(f"{pattern}.data")):
        tests.setdefault(test.stem, test)

    pairs = []
    for ref in sorted((root / "ligpargen").rglob(f"{pattern}.lmp")):
        pairs.append((ref.stem, ref, tests.get(ref.stem)))

    return pairs


SUMMARY_FIELDS = [
    "molecule",
    "n_atoms",
    "charge_mae_e",
    "ref_total_charge_e",
    "test_total_charge_e",
    "geometry_rmsd_a",
    "match_method",
    "status",
]


def write_summary_csv(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)


def write_summary_md(path, records):
    ok = [r for r in records if r["status"] == "ok"]
    maes = [float(r["charge_mae_e"]) for r in ok]

    lines = [
        "# Charge consistency: LigParGen vs MappingFF",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        "`compare_lammps_charges.py`.",
        "",
        "Charge MAE is the mean absolute per-atom charge difference "
        "(MappingFF - LigParGen) after atom matching.",
        "",
        "| Molecule | Atoms | Charge MAE (e) | LigParGen total q (e) | "
        "MappingFF total q (e) | Geometry RMSD (A) | Match |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in records:
        if r["status"] == "ok":
            lines.append(
                f"| {r['molecule']} | {r['n_atoms']} | {r['charge_mae_e']} | "
                f"{r['ref_total_charge_e']} | {r['test_total_charge_e']} | "
                f"{r['geometry_rmsd_a']} | {r['match_method']} |"
            )
        else:
            lines.append(
                f"| {r['molecule']} | - | ERROR | - | - | - | - |"
            )

    lines.append("")
    if maes:
        lines.append(f"Mean charge MAE over {len(maes)} molecule(s): "
                     f"{np.mean(maes):.8f} e")
        worst = max(ok, key=lambda r: float(r["charge_mae_e"]))
        lines.append(f"Largest charge MAE: {worst['molecule']} "
                     f"({worst['charge_mae_e']} e)")
    else:
        lines.append("No successful comparisons.")

    failures = [r for r in records if r["status"] != "ok"]
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"- **{r['molecule']}**: {r['status']}")

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report the charge MAE between LigParGen and MappingFF LAMMPS "
            "data files for one molecule or for every molecule under a root."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Single-pair mode: reference .lmp then test .data. "
            "Omit to run in batch mode over --root."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Batch mode root directory containing ligpargen/ and mappingff/ "
            "subdirectories (default: the directory of this script)."
        ),
    )
    parser.add_argument(
        "--glob",
        default="*",
        help=(
            "Filename pattern selecting which molecules to compare, matched "
            "against both sides (default: '*'). Use e.g. --glob '9_*' to "
            "compare only 9_x.lmp / 9_x.data and ignore other files left in "
            "the same directories."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("auto", "geometry", "topology"),
        default="auto",
        help=(
            "How to match atoms. 'geometry' needs both files in the same "
            "Cartesian frame; 'topology' matches on the bond graph alone and "
            "works even when the geometries differ; 'auto' (default) uses "
            "coordinates when possible and falls back to topology."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write the summary files (default: --root).",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help=(
            "Optional directory for per-atom charge CSV files, one per "
            "molecule (<molecule>.csv)."
        ),
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=5.0e-2,
        help=(
            "Maximum coordinate difference in Angstrom for atom matching "
            "(default: 0.05). LigParGen writes coordinates at low precision, "
            "so matching atoms can differ by ~0.01 A; this stays far below "
            "any bond length, keeping the nearest atom unambiguous."
        ),
    )
    args = parser.parse_args()

    if not HAVE_SCIPY:
        print("NOTE: scipy not found, using greedy atom assignment "
              "(install scipy for optimal assignment).")

    csv_dir = Path(args.csv_dir) if args.csv_dir else None

    # ---- single pair -----------------------------------------------------
    if args.paths:
        if len(args.paths) != 2:
            parser.error("single-pair mode takes exactly two paths: "
                         "<reference.lmp> <test.data>")
        reference, test = args.paths
        record, rows = run_pair(
            Path(reference).stem, reference, test, args.coord_tol,
            method=args.method,
        )
        if csv_dir:
            csv_dir.mkdir(parents=True, exist_ok=True)
            write_atom_csv(csv_dir / f"{record['molecule']}.csv", rows or [])
        if record["status"] != "ok":
            raise SystemExit(1)
        return

    # ---- batch -----------------------------------------------------------
    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    pairs = discover_pairs(root, pattern=args.glob)
    if not pairs:
        raise SystemExit(
            f"No ligpargen/**/{args.glob}.lmp files found under {root}."
        )

    print(f"Comparing charges for {len(pairs)} molecule(s) under {root}")
    print(f"Pattern: {args.glob}")
    print()

    records = []
    for name, reference, test in pairs:
        if test is None:
            record = {
                "molecule": name,
                "reference_file": str(reference),
                "test_file": "",
                "n_atoms": 0,
                "charge_mae_e": "",
                "ref_total_charge_e": "",
                "test_total_charge_e": "",
                "geometry_rmsd_a": "",
                "match_method": "",
                "status": "failed: no matching mappingff .data file found",
            }
            print(f"{name}\n  ERROR: no {name}.data found under "
                  f"{root / 'mappingff'}\n")
            records.append(record)
            continue

        record, rows = run_pair(
            name, reference, test, args.coord_tol, method=args.method
        )
        records.append(record)

        if csv_dir and rows:
            csv_dir.mkdir(parents=True, exist_ok=True)
            write_atom_csv(csv_dir / f"{name}.csv", rows)

    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in records if r["status"] == "ok"]
    maes = [float(r["charge_mae_e"]) for r in ok]

    print("Summary")
    print(f"{'molecule':<12} {'atoms':>6} {'charge MAE (e)':>16} "
          f"{'RMSD (A)':>9}  match")
    for r in records:
        if r["status"] == "ok":
            print(f"{r['molecule']:<12} {r['n_atoms']:>6} "
                  f"{r['charge_mae_e']:>16} {r['geometry_rmsd_a']:>9}  "
                  f"{r['match_method']}")
        else:
            print(f"{r['molecule']:<12} {'-':>6} {'ERROR':>16}")
    if maes:
        print()
        print(f"{'mean':<12} {'':>6} {np.mean(maes):>16.8f}")
    print()

    csv_path = out_dir / "charge_mae_summary.csv"
    md_path = out_dir / "charge_mae_summary.md"
    write_summary_csv(csv_path, records)
    write_summary_md(md_path, records)
    print(f"Summary written to: {csv_path}")
    print(f"Summary written to: {md_path}")

    if len(ok) != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()