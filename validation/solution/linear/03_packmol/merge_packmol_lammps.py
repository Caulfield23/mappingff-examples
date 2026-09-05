#!/usr/bin/env python3
"""
Merge Packmol XYZ coordinates with two LAMMPS data files.

Key assumptions
---------------
1. packed.xyz atom order is:
   [PS atoms in the same order as ps.data Atoms section]
   followed by n copies of
   [solvent atoms in the same order as solvent-data Atoms section]
2. The Atoms sections in ps.data and solvent-data do NOT need to be sorted by atom ID.
3. Bonds/Angles/Dihedrals/Impropers are transferred by old atom IDs through an
   old-ID -> new-ID map, so their original row order does not matter.
4. Original image flags in Atoms are ignored. Output Atoms # full has only:
   atom-ID mol-ID atom-type charge x y z

Example
-------
python merge_packmol_lammps_order_preserving.py \
  --packed packed.xyz \
  --ps-data ps_linear.data \
  --solvent-data cyclohexane.data \
  --n-solvent 4500 \
  --box -60 60 -60 60 -60 60 \
  --output system.data
"""

import argparse
import re
from collections import OrderedDict
from pathlib import Path

SECTION_NAMES = {
    "Masses",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "BondBond Coeffs",
    "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs",
    "EndBondTorsion Coeffs",
    "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs",
    "BondBond13 Coeffs",
    "AngleAngle Coeffs",
    "Atoms",
    "Velocities",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
}

TOPOLOGY_SECTIONS = {
    "Bonds": 2,
    "Angles": 3,
    "Dihedrals": 4,
    "Impropers": 4,
}

COEFF_TYPE_KIND = {
    "Masses": "atom",
    "Pair Coeffs": "atom",
    "Bond Coeffs": "bond",
    "Angle Coeffs": "angle",
    "Dihedral Coeffs": "dihedral",
    "Improper Coeffs": "improper",
    "BondBond Coeffs": "angle",
    "BondAngle Coeffs": "angle",
    "MiddleBondTorsion Coeffs": "dihedral",
    "EndBondTorsion Coeffs": "dihedral",
    "AngleTorsion Coeffs": "dihedral",
    "AngleAngleTorsion Coeffs": "dihedral",
    "BondBond13 Coeffs": "dihedral",
    "AngleAngle Coeffs": "improper",
}


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def split_comment(line: str):
    if "#" in line:
        a, b = line.split("#", 1)
        return a.rstrip(), " # " + b.strip()
    return line.rstrip(), ""


def clean_nonempty(lines):
    return [ln for ln in lines if strip_comment(ln)]


def read_lammps_data(path):
    path = Path(path)
    lines = path.read_text().splitlines()

    section_starts = []
    for i, line in enumerate(lines):
        name = strip_comment(line)
        if name in SECTION_NAMES:
            section_starts.append((i, name))

    if not section_starts:
        raise RuntimeError(f"No sections found in {path}")

    header = lines[: section_starts[0][0]]
    sections = OrderedDict()

    for idx, (start, name) in enumerate(section_starts):
        end = (
            section_starts[idx + 1][0] if idx + 1 < len(section_starts) else len(lines)
        )
        sections[name] = lines[start + 1 : end]

    return header, sections


def get_count_from_header(header, label):
    pattern = re.compile(rf"^\s*(\d+)\s+{re.escape(label)}\b")
    for line in header:
        m = pattern.match(line)
        if m:
            return int(m.group(1))
    return 0


def get_counts(header):
    return {
        "atoms": get_count_from_header(header, "atoms"),
        "bonds": get_count_from_header(header, "bonds"),
        "angles": get_count_from_header(header, "angles"),
        "dihedrals": get_count_from_header(header, "dihedrals"),
        "impropers": get_count_from_header(header, "impropers"),
        "atom_types": get_count_from_header(header, "atom types"),
        "bond_types": get_count_from_header(header, "bond types"),
        "angle_types": get_count_from_header(header, "angle types"),
        "dihedral_types": get_count_from_header(header, "dihedral types"),
        "improper_types": get_count_from_header(header, "improper types"),
    }


def parse_atoms_full(sections, source_name):
    """Parse Atoms # full and preserve the original row order."""
    if "Atoms" not in sections:
        raise RuntimeError(f"{source_name} has no Atoms section")

    atoms = []
    seen = set()
    for line in clean_nonempty(sections["Atoms"]):
        body, comment = split_comment(line)
        fields = body.split()

        if len(fields) < 7:
            raise RuntimeError(f"Bad Atoms line in {source_name}: {line}")

        old_id = int(fields[0])
        if old_id in seen:
            raise RuntimeError(f"Duplicate atom ID {old_id} in {source_name}")
        seen.add(old_id)

        atoms.append(
            {
                "old_id": old_id,
                "old_mol": int(fields[1]),
                "type": int(fields[2]),
                "charge": fields[3],
                "x": float(fields[4]),
                "y": float(fields[5]),
                "z": float(fields[6]),
                # Deliberately parsed but not written; image flags are ignored.
                "extra": fields[7:],
                "comment": comment,
            }
        )

    return atoms


def parse_topology_section(sections, section_name):
    """Parse topology and preserve original row order."""
    rows = []
    if section_name not in sections:
        return rows

    required_atoms = TOPOLOGY_SECTIONS[section_name]
    seen = set()
    for line in clean_nonempty(sections[section_name]):
        body, comment = split_comment(line)
        fields = body.split()
        if len(fields) < 2 + required_atoms:
            raise RuntimeError(f"Bad {section_name} line: {line}")

        old_id = int(fields[0])
        if old_id in seen:
            raise RuntimeError(f"Duplicate {section_name} ID {old_id}")
        seen.add(old_id)

        atoms = [int(x) for x in fields[2 : 2 + required_atoms]]
        rows.append(
            {
                "old_id": old_id,
                "type": int(fields[1]),
                "atoms": atoms,
                "comment": comment,
            }
        )

    return rows


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise RuntimeError("XYZ file is too short")

    natoms = int(lines[0].strip())
    xyz_lines = lines[2 : 2 + natoms]

    if len(xyz_lines) != natoms:
        raise RuntimeError("XYZ atom count does not match file length")

    coords = []
    for line in xyz_lines:
        f = line.split()
        if len(f) < 4:
            raise RuntimeError(f"Bad XYZ line: {line}")
        coords.append((float(f[-3]), float(f[-2]), float(f[-1])))

    return coords


def offset_coeff_line(line, offset):
    if not strip_comment(line):
        return line

    body, comment = split_comment(line)
    fields = body.split()
    if not fields:
        return line

    try:
        old_type = int(fields[0])
    except ValueError:
        return line

    fields[0] = str(old_type + offset)
    return " ".join(fields) + comment


def write_coeff_section(out, name, ps_sections, cy_sections, offsets):
    lines_out = []

    if name in ps_sections:
        lines_out.extend(clean_nonempty(ps_sections[name]))

    if name in cy_sections:
        kind = COEFF_TYPE_KIND.get(name)
        offset = offsets.get(kind, 0)
        for line in clean_nonempty(cy_sections[name]):
            lines_out.append(offset_coeff_line(line, offset))

    if not lines_out:
        return

    out.write(f"\n{name}\n\n")
    for line in lines_out:
        out.write(line.rstrip() + "\n")


def write_atoms(out, ps_atoms, cy_atoms, coords, n_cy, offsets):
    """Write Atoms # full without image flags and return old-ID -> new-ID maps."""
    nat_ps = len(ps_atoms)
    nat_cy = len(cy_atoms)

    out.write("\nAtoms # full\n\n")

    atom_id = 1
    ps_old_to_new = {}
    cy_old_to_template_new = {}

    # PS molecule-ID = 1. Coordinates follow ps.data Atoms row order.
    for i, atom in enumerate(ps_atoms):
        x, y, z = coords[i]
        ps_old_to_new[atom["old_id"]] = atom_id

        line = [
            str(atom_id),
            "1",
            str(atom["type"]),
            atom["charge"],
            f"{x:.8f}",
            f"{y:.8f}",
            f"{z:.8f}",
        ]
        out.write(" ".join(line) + "\n")
        atom_id += 1

    # Template map for one solvent molecule, preserving solvent-data Atoms row order.
    for j, atom in enumerate(cy_atoms):
        cy_old_to_template_new[atom["old_id"]] = j + 1

    # Solvent molecule-IDs = 2, 3, ... Coordinates follow repeated template order.
    coord_start = nat_ps
    for m in range(n_cy):
        mol_id = m + 2
        for j, atom in enumerate(cy_atoms):
            idx = coord_start + m * nat_cy + j
            x, y, z = coords[idx]

            line = [
                str(atom_id),
                str(mol_id),
                str(atom["type"] + offsets["atom"]),
                atom["charge"],
                f"{x:.8f}",
                f"{y:.8f}",
                f"{z:.8f}",
            ]
            out.write(" ".join(line) + "\n")
            atom_id += 1

    return ps_old_to_new, cy_old_to_template_new


def remap_atoms(atom_ids, old_to_new, section_name, source_name):
    new_ids = []
    for old_id in atom_ids:
        try:
            new_ids.append(old_to_new[old_id])
        except KeyError as exc:
            raise RuntimeError(
                f"{section_name} in {source_name} references atom ID {old_id}, "
                f"but it is absent from that file's Atoms section"
            ) from exc
    return new_ids


def write_topology(
    out,
    section_name,
    ps_rows,
    cy_rows,
    n_cy,
    nat_ps,
    nat_cy,
    type_offsets,
    ps_old_to_new,
    cy_old_to_template_new,
):
    if not (ps_rows or cy_rows):
        return

    out.write(f"\n{section_name}\n\n")

    item_id = 1

    # PS topology: preserve original row order, remap old atom IDs to new IDs.
    for row in ps_rows:
        new_atoms = remap_atoms(row["atoms"], ps_old_to_new, section_name, "ps-data")
        fields = [str(item_id), str(row["type"])] + [str(a) for a in new_atoms]
        out.write(" ".join(fields) + "\n")
        item_id += 1

    type_kind = {
        "Bonds": "bond",
        "Angles": "angle",
        "Dihedrals": "dihedral",
        "Impropers": "improper",
    }[section_name]
    type_offset = type_offsets[type_kind]

    # Solvent topology: preserve template row order for each molecule.
    for m in range(n_cy):
        molecule_atom_offset = nat_ps + m * nat_cy
        for row in cy_rows:
            template_ids = remap_atoms(
                row["atoms"], cy_old_to_template_new, section_name, "solvent-data"
            )
            new_atoms = [molecule_atom_offset + a for a in template_ids]
            new_type = row["type"] + type_offset
            fields = [str(item_id), str(new_type)] + [str(a) for a in new_atoms]
            out.write(" ".join(fields) + "\n")
            item_id += 1


def main():
    ap = argparse.ArgumentParser(
        description="Merge Packmol XYZ coordinates with single-molecule LAMMPS data files."
    )
    ap.add_argument("--packed", default="packed.xyz")
    ap.add_argument("--ps-data", default="ps_linear.data")
    ap.add_argument("--solvent-data", default="cyclohexane.data")
    ap.add_argument("--n-solvent", type=int, default=4500)
    ap.add_argument("--output", default="system.data")
    ap.add_argument(
        "--box",
        nargs=6,
        type=float,
        default=[-60.0, 60.0, -60.0, 60.0, -60.0, 60.0],
        metavar=("XLO", "XHI", "YLO", "YHI", "ZLO", "ZHI"),
    )
    args = ap.parse_args()

    ps_header, ps_sections = read_lammps_data(args.ps_data)
    cy_header, cy_sections = read_lammps_data(args.solvent_data)

    ps_counts = get_counts(ps_header)
    cy_counts = get_counts(cy_header)

    ps_atoms = parse_atoms_full(ps_sections, args.ps_data)
    cy_atoms = parse_atoms_full(cy_sections, args.solvent_data)

    nat_ps = len(ps_atoms)
    nat_cy = len(cy_atoms)
    n_cy = args.n_solvent

    coords = read_xyz(args.packed)
    expected_natoms = nat_ps + n_cy * nat_cy

    if len(coords) != expected_natoms:
        raise RuntimeError(
            f"Atom count mismatch:\n"
            f"  packed.xyz atoms     = {len(coords)}\n"
            f"  expected atoms       = {expected_natoms}\n"
            f"  PS atoms             = {nat_ps}\n"
            f"  solvent atoms/mol    = {nat_cy}\n"
            f"  n solvent            = {n_cy}\n"
        )

    offsets = {
        "atom": ps_counts["atom_types"],
        "bond": ps_counts["bond_types"],
        "angle": ps_counts["angle_types"],
        "dihedral": ps_counts["dihedral_types"],
        "improper": ps_counts["improper_types"],
    }

    ps_top = {
        name: parse_topology_section(ps_sections, name) for name in TOPOLOGY_SECTIONS
    }
    cy_top = {
        name: parse_topology_section(cy_sections, name) for name in TOPOLOGY_SECTIONS
    }

    total_counts = {
        "atoms": nat_ps + n_cy * nat_cy,
        "bonds": len(ps_top["Bonds"]) + n_cy * len(cy_top["Bonds"]),
        "angles": len(ps_top["Angles"]) + n_cy * len(cy_top["Angles"]),
        "dihedrals": len(ps_top["Dihedrals"]) + n_cy * len(cy_top["Dihedrals"]),
        "impropers": len(ps_top["Impropers"]) + n_cy * len(cy_top["Impropers"]),
        "atom_types": ps_counts["atom_types"] + cy_counts["atom_types"],
        "bond_types": ps_counts["bond_types"] + cy_counts["bond_types"],
        "angle_types": ps_counts["angle_types"] + cy_counts["angle_types"],
        "dihedral_types": ps_counts["dihedral_types"] + cy_counts["dihedral_types"],
        "improper_types": ps_counts["improper_types"] + cy_counts["improper_types"],
    }

    xlo, xhi, ylo, yhi, zlo, zhi = args.box

    with open(args.output, "w") as out:
        out.write("LAMMPS data generated from Packmol XYZ + ps.data + solvent.data\n\n")

        out.write(f"{total_counts['atoms']} atoms\n")
        out.write(f"{total_counts['bonds']} bonds\n")
        out.write(f"{total_counts['angles']} angles\n")
        out.write(f"{total_counts['dihedrals']} dihedrals\n")
        out.write(f"{total_counts['impropers']} impropers\n\n")

        out.write(f"{total_counts['atom_types']} atom types\n")
        if total_counts["bond_types"] > 0:
            out.write(f"{total_counts['bond_types']} bond types\n")
        if total_counts["angle_types"] > 0:
            out.write(f"{total_counts['angle_types']} angle types\n")
        if total_counts["dihedral_types"] > 0:
            out.write(f"{total_counts['dihedral_types']} dihedral types\n")
        if total_counts["improper_types"] > 0:
            out.write(f"{total_counts['improper_types']} improper types\n")

        out.write("\n")
        out.write(f"{xlo:.8f} {xhi:.8f} xlo xhi\n")
        out.write(f"{ylo:.8f} {yhi:.8f} ylo yhi\n")
        out.write(f"{zlo:.8f} {zhi:.8f} zlo zhi\n")

        for secname in [
            "Masses",
            "Pair Coeffs",
            "Bond Coeffs",
            "Angle Coeffs",
            "Dihedral Coeffs",
            "Improper Coeffs",
            "BondBond Coeffs",
            "BondAngle Coeffs",
            "MiddleBondTorsion Coeffs",
            "EndBondTorsion Coeffs",
            "AngleTorsion Coeffs",
            "AngleAngleTorsion Coeffs",
            "BondBond13 Coeffs",
            "AngleAngle Coeffs",
        ]:
            write_coeff_section(out, secname, ps_sections, cy_sections, offsets)

        ps_old_to_new, cy_old_to_template_new = write_atoms(
            out, ps_atoms, cy_atoms, coords, n_cy, offsets
        )

        for secname in ["Bonds", "Angles", "Dihedrals", "Impropers"]:
            write_topology(
                out,
                secname,
                ps_top[secname],
                cy_top[secname],
                n_cy,
                nat_ps,
                nat_cy,
                offsets,
                ps_old_to_new,
                cy_old_to_template_new,
            )

    print(f"Generated: {args.output}")
    print(f"PS atoms: {nat_ps}")
    print(f"Solvent atoms per molecule: {nat_cy}")
    print(f"Solvent molecules: {n_cy}")
    print(f"Total atoms: {total_counts['atoms']}")
    print("Type offsets applied to solvent:")
    print(f"  atom type offset     = {offsets['atom']}")
    print(f"  bond type offset     = {offsets['bond']}")
    print(f"  angle type offset    = {offsets['angle']}")
    print(f"  dihedral type offset = {offsets['dihedral']}")
    print(f"  improper type offset = {offsets['improper']}")
    print("Image flags from input Atoms sections were ignored and not written.")


if __name__ == "__main__":
    main()
