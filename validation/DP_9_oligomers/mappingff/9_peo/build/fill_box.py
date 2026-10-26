"""fill_box.py — 把一条聚合物链复制 N 份后随机填入盒子。

工作流
------
1. 读取单链 LAMMPS data 文件，抽出原子坐标、力场参数、拓扑。
2. 计算链的回转半径 Rg 与总质量 M_chain。
3. 按目标密度 ρ 与分子数 N 反算盒子尺寸
       V = N · M_chain / (NA · ρ)        （real 单位：g/mol, g/cm³, Å）
       L = V^(1/3)
4. 在盒子中以最小中心间距 = factor · R_sphere 随机放置 N 个球。
5. 每条模板链用随机 SO(3) 旋转 + 平移到对应球心。
6. 写出新 LAMMPS data 文件，原子/键/角/二面角编号按链分块：
       链 1:  原子 1..143,   键 1..142,   角 143..403,   二面角 404..703
       链 2:  原子 144..286, 键 143..284, 角 405..665,   二面角 704..1003
       …

所有可调参数集中在 FillParams 里。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Avogadro 常数；real 单位下 1 g/mol = 1/NA g，1 cm³ = 1e24 Å³
NA = 6.02214076e23
ANGSTROM3_PER_CM3 = 1e24


# ---------------------------------------------------------------------------
# 用户参数：所有可调项都在这里，不用命令行
# ---------------------------------------------------------------------------

@dataclass
class FillParams:
    """填盒参数。改这里就行。"""

    # ---- I/O ----
    template_data: str = "9_peo.data"   # 单链模板
    output_data: str = "filled.data"      # 输出文件名

    # ---- 填充量 ----
    n_molecules: int = 100               # 盒子里的链数
    density: float = 0.1                 # 目标质量密度 (g/cm³)

    # ---- 几何约束 ----
    sphere_padding: float = 2.0          # 占据球半径 = padding · Rg
    min_dist_factor: float = 0.5         # 球心最小间距 = factor · 球半径

    # ---- 装箱细节 ----
    max_attempts: int = 100_000          # 每个球最大随机尝试次数
    rng_seed: int = 42                   # 随机种子；设 None 不可复现


# ---------------------------------------------------------------------------
# LAMMPS data 文件 I/O
# ---------------------------------------------------------------------------

def parse_data_file(path: str | Path) -> Dict:
    """解析 LAMMPS data file，返回结构化字典。

    解析范围：header、Masses、Pair/Bond/Angle/Dihedral Coeffs、Atoms(full)、
    Bonds、Angles、Dihedrals。Velocities 仅识别、不提取（输出里不带）。
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    # ---- header ----
    n_atoms = n_bonds = n_angles = n_dihedrals = 0
    atom_types = bond_types = angle_types = dihedral_types = 0
    box = [0.0] * 6

    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) < 2:
            continue
        key = parts[1]
        if key == "atoms" and len(parts) == 2:
            n_atoms = int(parts[0])
        elif key == "bonds" and len(parts) == 2:
            n_bonds = int(parts[0])
        elif key == "angles" and len(parts) == 2:
            n_angles = int(parts[0])
        elif key == "dihedrals" and len(parts) == 2:
            n_dihedrals = int(parts[0])
        elif key == "atom" and len(parts) >= 3 and parts[2] == "types":
            atom_types = int(parts[0])
        elif key == "bond" and len(parts) >= 3 and parts[2] == "types":
            bond_types = int(parts[0])
        elif key == "angle" and len(parts) >= 3 and parts[2] == "types":
            angle_types = int(parts[0])
        elif key == "dihedral" and len(parts) >= 3 and parts[2] == "types":
            dihedral_types = int(parts[0])
        elif len(parts) >= 4 and parts[2] == "xlo" and parts[3] == "xhi":
            box[0] = float(parts[0]); box[1] = float(parts[1])
            box[2] = float(lines[i + 1].split()[0])
            box[3] = float(lines[i + 1].split()[1])
            box[4] = float(lines[i + 2].split()[0])
            box[5] = float(lines[i + 2].split()[1])
            break

    # ---- 定位各 section 起始行 ----
    def find_section(label: str) -> int:
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s == label or s.startswith(label + " "):
                return i
        return -1

    sec = {
        "masses": find_section("Masses"),
        "pair": find_section("Pair Coeffs"),
        "bond": find_section("Bond Coeffs"),
        "angle": find_section("Angle Coeffs"),
        "dihedral": find_section("Dihedral Coeffs"),
        "atoms": find_section("Atoms"),
        "bonds": find_section("Bonds"),
        "angles": find_section("Angles"),
        "dihedrals": find_section("Dihedrals"),
    }

    def rows_after(start: int, ncols: int) -> List[List[str]]:
        out: List[List[str]] = []
        if start < 0:
            return out
        for ln in lines[start + 2:]:
            s = ln.strip()
            if not s:
                break
            p = s.split()
            if len(p) >= ncols:
                out.append(p[:ncols])
        return out

    # ---- Masses ----
    masses: Dict[int, float] = {}
    for p in rows_after(sec["masses"], 2):
        masses[int(p[0])] = float(p[1])

    # ---- 力场参数 ----
    pair_coeffs: Dict[int, Tuple[float, float]] = {}
    for p in rows_after(sec["pair"], 3):
        pair_coeffs[int(p[0])] = (float(p[1]), float(p[2]))

    bond_coeffs: Dict[int, Tuple[float, float]] = {}
    for p in rows_after(sec["bond"], 3):
        bond_coeffs[int(p[0])] = (float(p[1]), float(p[2]))

    angle_coeffs: Dict[int, Tuple[float, float]] = {}
    for p in rows_after(sec["angle"], 3):
        angle_coeffs[int(p[0])] = (float(p[1]), float(p[2]))

    dihedral_coeffs: Dict[int, Tuple[float, ...]] = {}
    for p in rows_after(sec["dihedral"], 5):
        dihedral_coeffs[int(p[0])] = tuple(float(x) for x in p[1:5])

    # ---- Atoms (full style: id mol type charge x y z ix iy iz) ----
    atoms: List[Tuple] = []
    for p in rows_after(sec["atoms"], 7):
        # 兼容没有 image flags 的写法
        ix = int(p[7]) if len(p) >= 8 else 0
        iy = int(p[8]) if len(p) >= 9 else 0
        iz = int(p[9]) if len(p) >= 10 else 0
        atoms.append((
            int(p[0]), int(p[1]), int(p[2]), float(p[3]),
            float(p[4]), float(p[5]), float(p[6]), ix, iy, iz,
        ))

    # ---- 拓扑 ----
    bonds = [tuple(int(x) for x in p) for p in rows_after(sec["bonds"], 4)]
    angles = [tuple(int(x) for x in p) for p in rows_after(sec["angles"], 5)]
    # dihedrals 是 6 列: id type a b c d；之前误用 ncols=5 把第 4 个原子截掉，
    # 导致 OVITO 报 "Invalid dihedral specification"。
    dihedrals = [tuple(int(x) for x in p) for p in rows_after(sec["dihedrals"], 6)]

    return {
        "title": lines[0] if lines else "",
        "n_atoms": n_atoms, "n_bonds": n_bonds,
        "n_angles": n_angles, "n_dihedrals": n_dihedrals,
        "atom_types": atom_types, "bond_types": bond_types,
        "angle_types": angle_types, "dihedral_types": dihedral_types,
        "box": tuple(box),
        "masses": masses, "pair_coeffs": pair_coeffs,
        "bond_coeffs": bond_coeffs, "angle_coeffs": angle_coeffs,
        "dihedral_coeffs": dihedral_coeffs,
        "atoms": atoms, "bonds": bonds,
        "angles": angles, "dihedrals": dihedrals,
    }


def write_data_file(path: str | Path, template: Dict,
                    atoms_out: List[Tuple], bonds_out: List[Tuple],
                    angles_out: List[Tuple], dihedrals_out: List[Tuple],
                    box: Tuple[float, float, float, float, float, float]) -> None:
    """写出新 LAMMPS data file。"""
    L: List[str] = []
    L.append("LAMMPS data file via fill_box.py")
    L.append("")
    L.append(f"{len(atoms_out)} atoms")
    L.append(f"{len(bonds_out)} bonds")
    L.append(f"{len(angles_out)} angles")
    L.append(f"{len(dihedrals_out)} dihedrals")
    L.append("")
    L.append(f"{template['atom_types']} atom types")
    L.append(f"{template['bond_types']} bond types")
    L.append(f"{template['angle_types']} angle types")
    L.append(f"{template['dihedral_types']} dihedral types")
    L.append("")
    L.append(f"{box[0]:.6f} {box[1]:.6f} xlo xhi")
    L.append(f"{box[2]:.6f} {box[3]:.6f} ylo yhi")
    L.append(f"{box[4]:.6f} {box[5]:.6f} zlo zhi")
    L.append("")

    L.append("Masses")
    L.append("")
    for t in sorted(template["masses"]):
        L.append(f"{t} {template['masses'][t]}")
    L.append("")

    if template["pair_coeffs"]:
        L.append("Pair Coeffs # lj/cut/coul/long")
        L.append("")
        for t in sorted(template["pair_coeffs"]):
            eps, sig = template["pair_coeffs"][t]
            L.append(f"{t} {eps} {sig}")
        L.append("")

    if template["bond_coeffs"]:
        L.append("Bond Coeffs # harmonic")
        L.append("")
        for t in sorted(template["bond_coeffs"]):
            k, r0 = template["bond_coeffs"][t]
            L.append(f"{t} {k} {r0}")
        L.append("")

    if template["angle_coeffs"]:
        L.append("Angle Coeffs # harmonic")
        L.append("")
        for t in sorted(template["angle_coeffs"]):
            k, t0 = template["angle_coeffs"][t]
            L.append(f"{t} {k} {t0}")
        L.append("")

    if template["dihedral_coeffs"]:
        L.append("Dihedral Coeffs # opls")
        L.append("")
        for t in sorted(template["dihedral_coeffs"]):
            coeffs = template["dihedral_coeffs"][t]
            L.append(f"{t} " + " ".join(str(c) for c in coeffs))
        L.append("")

    L.append("Atoms # full")
    L.append("")
    # a = (id, mol, type, charge, x, y, z, ix, iy, iz)
    for a in atoms_out:
        L.append(
            f"{a[0]} {a[1]} {a[2]} {a[3]:.5f} "
            f"{a[4]:.6f} {a[5]:.6f} {a[6]:.6f} {a[7]} {a[8]} {a[9]}"
        )
    L.append("")

    if bonds_out:
        L.append("Bonds")
        L.append("")
        for b in bonds_out:
            L.append(f"{b[0]} {b[1]} {b[2]} {b[3]}")
        L.append("")

    if angles_out:
        L.append("Angles")
        L.append("")
        for a in angles_out:
            L.append(f"{a[0]} {a[1]} {a[2]} {a[3]} {a[4]}")
        L.append("")

    if dihedrals_out:
        L.append("Dihedrals")
        L.append("")
        for d in dihedrals_out:
            L.append(f"{d[0]} {d[1]} {d[2]} {d[3]} {d[4]} {d[5]}")
        L.append("")

    Path(path).write_text("\n".join(L))


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def chain_center_and_rg(atoms: List[Tuple], masses: Dict[int, float]
                        ) -> Tuple[np.ndarray, float, float]:
    """返回 (质心, 回转半径, 最大原子距质心距离)。"""
    coords = np.array([[a[4], a[5], a[6]] for a in atoms])
    types = np.array([a[2] for a in atoms])
    m = np.array([masses[t] for t in types])
    total_m = m.sum()
    com = (m[:, None] * coords).sum(axis=0) / total_m
    disp2 = ((coords - com) ** 2).sum(axis=1)
    rg = math.sqrt((m * disp2).sum() / total_m)
    rmax = float(np.sqrt(disp2).max())
    return com, rg, rmax


def chain_total_mass(atoms: List[Tuple], masses: Dict[int, float]) -> float:
    """链总质量 (g/mol)。"""
    return sum(masses[a[2]] for a in atoms)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """均匀随机 SO(3) 旋转矩阵（Mezzadri 2007）。"""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def pack_spheres(n: int, box_lo: float, box_hi: float,
                 margin: float, min_dist: float,
                 rng: np.random.Generator,
                 max_attempts: int) -> np.ndarray:
    """在 [box_lo, box_hi]^3 中随机放置 n 个不重叠的球。

    margin: 球心距盒子边界的最小距离（用 max(sphere_radius, 链最大延伸)）。
    min_dist: 任意两球心的最小距离。
    """
    centers = np.empty((n, 3))
    if box_hi - box_lo < 2 * margin:
        raise ValueError(
            f"盒子边长 {box_hi - box_lo:.2f} 小于 2×margin {2 * margin:.2f}"
        )

    min_dist2 = min_dist * min_dist
    placed = 0
    total_tries = 0
    while placed < n:
        total_tries += 1
        if total_tries > n * max_attempts:
            raise RuntimeError(
                f"装了 {placed}/{n} 个球就装不下了（重试 {total_tries} 次）。"
                " 请增大盒子、减小 n_molecules 或减小 min_dist_factor。"
            )
        trial = rng.uniform(box_lo + margin, box_hi - margin, size=3)
        if placed == 0:
            centers[0] = trial
            placed = 1
            continue
        diff = centers[:placed] - trial
        d2 = (diff * diff).sum(axis=1)
        if (d2 >= min_dist2).all():
            centers[placed] = trial
            placed += 1
    return centers


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(params: FillParams) -> None:
    rng = np.random.default_rng(params.rng_seed)

    print(f"[fill_box] 读取模板 {params.template_data}")
    tpl = parse_data_file(params.template_data)

    # 模板里原子顺序可能不是 1..N，按原 id 排序以建立稳定映射
    atoms_sorted = sorted(tpl["atoms"], key=lambda a: a[0])
    old_id_to_local = {a[0]: i for i, a in enumerate(atoms_sorted)}
    n_per_chain = len(atoms_sorted)

    # 链的质心、Rg、最大延伸
    com, rg, rmax = chain_center_and_rg(atoms_sorted, tpl["masses"])
    sphere_radius = params.sphere_padding * rg
    min_dist = params.min_dist_factor * sphere_radius
    # 用最大延伸保证原子不出盒
    margin = max(sphere_radius, rmax)
    print(f"[fill_box] Rg = {rg:.3f} Å, 最大延伸 = {rmax:.3f} Å, "
          f"占据球半径 = {sphere_radius:.3f} Å, "
          f"最小间距 = {min_dist:.3f} Å, 边界留白 = {margin:.3f} Å")

    # 盒子尺寸：质量来自 data 文件
    m_chain = chain_total_mass(atoms_sorted, tpl["masses"])
    print(f"[fill_box] 单链质量 = {m_chain:.3f} g/mol")
    # 1 mol 占据的体积 = M/(NA·ρ) cm³；换算 Å³
    vol_per_chain_A3 = m_chain * ANGSTROM3_PER_CM3 / (NA * params.density)
    box_vol_A3 = params.n_molecules * vol_per_chain_A3
    L = box_vol_A3 ** (1.0 / 3.0)
    box = (0.0, L, 0.0, L, 0.0, L)
    print(f"[fill_box] 目标密度 {params.density} g/cm³ → "
          f"V = {box_vol_A3:.1f} Å³, L = {L:.2f} Å")

    # 装球
    print(f"[fill_box] 装入 {params.n_molecules} 个球 ...")
    centers = pack_spheres(
        n=params.n_molecules,
        box_lo=0.0, box_hi=L,
        margin=margin, min_dist=min_dist,
        rng=rng, max_attempts=params.max_attempts,
    )

    # 模板坐标（相对质心），用于旋转 + 平移
    coords_tmpl = np.array([[a[4], a[5], a[6]] for a in atoms_sorted]) - com

    # 输出累积
    atoms_out: List[Tuple] = []
    bonds_out: List[Tuple] = []
    angles_out: List[Tuple] = []
    dihedrals_out: List[Tuple] = []

    n_b = len(tpl["bonds"])
    n_a = len(tpl["angles"])
    n_d = len(tpl["dihedrals"])

    for mol_id in range(1, params.n_molecules + 1):
        # 随机旋转
        R = random_rotation_matrix(rng)
        rotated = coords_tmpl @ R.T + centers[mol_id - 1]

        base = (mol_id - 1) * n_per_chain  # 该链起始原子的全局 id（0 索引）

        # 原子：链内 i=0..n_per_chain-1 → 全局 id = base+i+1
        for i, a in enumerate(atoms_sorted):
            atoms_out.append((
                base + i + 1, mol_id, a[2], a[3],
                rotated[i, 0], rotated[i, 1], rotated[i, 2],
                0, 0, 0,
            ))

        # 键/角/二面角：保留类型，把旧 atom id 映射到新位置
        for j, b in enumerate(tpl["bonds"], start=1):
            bonds_out.append((
                (mol_id - 1) * n_b + j,
                b[1],
                base + old_id_to_local[b[2]] + 1,
                base + old_id_to_local[b[3]] + 1,
            ))
        for j, a in enumerate(tpl["angles"], start=1):
            angles_out.append((
                (mol_id - 1) * n_a + j,
                a[1],
                base + old_id_to_local[a[2]] + 1,
                base + old_id_to_local[a[3]] + 1,
                base + old_id_to_local[a[4]] + 1,
            ))
        for j, d in enumerate(tpl["dihedrals"], start=1):
            dihedrals_out.append((
                (mol_id - 1) * n_d + j,
                d[1],
                base + old_id_to_local[d[2]] + 1,
                base + old_id_to_local[d[3]] + 1,
                base + old_id_to_local[d[4]] + 1,
                base + old_id_to_local[d[5]] + 1,
            ))

    print(f"[fill_box] 写出 {params.output_data}: "
          f"{len(atoms_out)} atoms, {len(bonds_out)} bonds, "
          f"{len(angles_out)} angles, {len(dihedrals_out)} dihedrals")
    write_data_file(
        params.output_data, tpl,
        atoms_out, bonds_out, angles_out, dihedrals_out, box,
    )


if __name__ == "__main__":
    run(FillParams())