# ligpargen vs mappingff — 密度 / Rg 对比分析

工作目录：`C:\schoolwork\Projects\mappingff\validation\longer`

## 体系

8 个算例 = 2 个力场（`ligpargen` / `mappingff`）× 4 种低聚物（`9_peo` / `9_pmma` / `9_ps` / `9_pva`）。
每个体系是**纯低聚物熔体，无溶剂**，含 100 条完全相同的链：

| 算例 | 单链原子数 | 总原子数 | 类型数 |
|---|---|---|---|
| 9_peo  | 66  | 6600  | 66  |
| 9_pmma | 137 | 13700 | 137 |
| 9_ps   | 146 | 14600 | 146 |
| 9_pva  | 65  | 6500  | 65  |

每个原子使用独立的 LAMMPS atom type（per-atom typing），分子 ID 1–100。

## 方法

**轨迹**：`dump.lammpstrj`，列 `id mol type q x y z xu yu zu`，每 5000 步一帧，共 400 帧（step 5000–2000000）。

**Rg**：对**每条链单独**计算质量加权回转半径

```
Rg² = Σ_i m_i |r_i − r_com|² / Σ_i m_i
```

坐标取 **未折回的 `xu yu zu`**（不是 `x y z`）——链会跨周期性边界，用折回坐标算出的 Rg 会完全错误。
每帧算 100 条链的 Rg，在 101 帧上取平均（10100 个链-帧样本）。
报告的 `rg_std` 是**帧内跨链**的标准差（不是时间序列标准差）。

> **PBC 处理有多重要（实测）**：在 step 2,000,000 的帧上，9_pva 有 62/100 条链、9_ps 有 64/100 条链
> 跨越周期边界。若改用折回坐标且不做 minimum-image 修正，9_pva 的 Rg 会从 6.17 Å 虚高到 11.19 Å，
> 9_ps 从 6.36 Å 虚高到 14.31 Å（最大单链误差 17.8 / 28.2 Å）。用 `xu` 是必须的，不是风格问题。

**密度**：直接读 `log.lammps` 的 thermo 列 `Density`（g/cm³）。
- `rho_nvt` = step 1,500,000–2,000,000 → 最后的 NVT 生产段，盒子冻结，密度基本恒定。
- `rho_npt` = step 1,300,000–1,500,000 → eq_lo NPT 段尾部（已在 300 K 稳定），密度有真实涨落。

## 步数结构与 n_min 偏移

每个算例的流程是：`minimize` → deform(NVT 400k) → eq_hi(NPT 100k) → cool(NPT 500k) →
eq_lo(NPT 500k) → eq_lo_nvt(NVT 500k)，合计 2,000,000 步。

**`minimize` 会推进 timestep**，各算例的迭代数 n_min 不同，因此最后的 NVT 段真实起点是
`1,500,000 + n_min`：

| 算例 | n_min | NVT 段真实起点 |
|---|---|---|
| ligpargen/9_ps  | 567 | 1,500,567 |
| mappingff/9_ps  | 593 | 1,500,593 |
| mappingff/9_peo | 644 | 1,500,644 |
| ligpargen/9_peo, 9_pmma, 9_pva, mappingff/9_pmma, 9_pva | 1000 | 1,501,000 |

按要求，全部 8 个算例统一使用**名义窗口 [1,500,000, 2,000,000]**。影响：

- Rg：混入 1 帧 NPT 数据（共 101 帧，真 NVT 段为 100 帧），偏差 < 0.1%；
- 密度：step=1,500,000 那一行仍属 NPT 段（盒子在动），所以 `rho_nvt_std` 是 ~1e-4 而非严格 0。
  真实 NVT 段的密度严格恒定，见 `density_rg_per_run.csv` 的 `rho_nvt_exact_*` 列。

## 输出文件

### `density_rg_comparison.csv`（主表，每行一个低聚物）

| 列 | 含义 |
|---|---|
| `system` | 低聚物 |
| `atoms_per_chain`, `n_chain` | 单链原子数、链数 |
| `rho_nvt_ligpargen/mappingff` | NVT 生产段密度 (g/cm³) |
| `rho_nvt_delta`, `rho_nvt_dev_pct` | mappingff − ligpargen 及其百分比 |
| `rho_nvt_dev_pct_abs` | 偏差绝对值（便于排序） |
| `rho_npt_ligpargen/mappingff` | NPT 段密度均值 (g/cm³) |
| `rho_npt_std_ligpargen/mappingff` | NPT 段密度标准差 |
| `rho_npt_delta`, `rho_npt_dev_pct`, `rho_npt_dev_pct_abs` | 同上 |
| `rg_ligpargen/mappingff` | 质量加权 Rg (Å) |
| `rg_std_ligpargen/mappingff` | 帧内跨链 Rg 标准差 (Å) |
| `rg_delta`, `rg_dev_pct`, `rg_dev_pct_abs` | mappingff − ligpargen 及其百分比 |
| `rg_zscore` | 双样本 z 检验（分块 SEM，考虑了自相关）；反映**统计**可分辨性 |
| `verdict_rho`, `verdict_rg` | 按 `|偏差| ≤ 3%` 判定的 `consistent` / `INCONSISTENT` |

### `density_rg_per_run.csv`（每个算例一行，8 行）

含上述量的原始值，以及 `n_min` / `nvt_stage_start` / `rho_nvt_exact_*`（真实 NVT 段密度，
std 严格为 0）、`frames_analyzed` / `chain_samples` 等溯源信息。

## 结果

| 体系 | ρ NVT (lig/mff) | ρ 偏差 | Rg (lig/mff) | Rg 偏差 | 判定 |
|---|---|---|---|---|---|
| 9_peo  | 1.1334 / 1.1244 | −0.80 % | 5.919 / 6.033 | +1.93 % | 一致 |
| 9_pmma | 0.8288 / 0.9225 | **+11.30 %** | 6.425 / 6.476 | +0.79 % | 密度不一致 |
| 9_ps   | 0.9122 / 0.9573 | **+4.94 %** | 6.361 / 6.366 | +0.07 % | 密度不一致 |
| 9_pva  | 1.0109 / 1.0946 | **+8.29 %** | 6.162 / 6.619 | **+7.41 %** | 密度、Rg 均不一致 |

结论：**Rg 三个体系一致（PEO/PMMA/PS 偏差 ≤ 2%），PVA 不一致（7.4%）；
密度三个体系不一致（PMMA 11.3%、PVA 8.3%、PS 4.9%），只有 PEO 一致（0.8%）。**

注意 `rg_zscore` 数值很大（即使 PEO 的 1.9% 偏差 z≈25）：这说明两个力场给出的 Rg 在**统计上**
可分辨，但**实际**偏差大小才是评价依据。密度在 NVT 段没有涨落，`rho_nvt_std` 不能用来判断
误差，NPT 段的 `rho_npt_std` 才能。

## 力场差异审计（ligpargen vs mappingff 到底哪里不同）

逐原子 / 逐拓扑元素比对两个力场的参数文件
（`ligpargen/9_*/build/9_*.lmp` vs `mappingff/9_*/build/9_*.data`）：

| 参数 | 9_peo | 9_pmma | 9_ps | 9_pva |
|---|---|---|---|---|
| 原子质量 | 相同 | 相同 | 相同 | 相同 |
| LJ (ε, σ) | 相同 | 相同 | 相同 | 相同 |
| 键力常数 (K, r₀) | 相同 | 相同 | 相同 | 相同 |
| 角力常数 (K, θ₀) | 相同 | 相同 | 相同 | 相同 |
| **OPLS 二面角 (k0–k3)** | 相同 | **不同** | 相同 | **不同** |
| **部分电荷** | **不同** | **不同** | **不同** | **不同** |
| 初始随机堆积坐标 | 不同 | 不同 | 相同 | 不同 |
| 初始盒子 | 相同 | 相同 | 相同 | 相同 |
| 分子 ID / 原子数 / 链数 | 相同 | 相同 | 相同 | 相同 |

要点：

- 两个力场的**非键与键合参数实质相同**，唯一系统性差别是**部分电荷**；
  9_pva / 9_pmma 另外还有 OPLS 二面角系数的差别（扭转类型分配不同，
  如 ligpargen `(0.850, -0.200, 0.200, 0.000)` vs mappingff `(1.300, -0.200, 0.200, 0.000)`）。
- 初始堆积不同（9_ps 相同），但平衡协议长达 400k 步压缩 + 1.1M 步 NPT/NVT，
  初始差异会被洗掉；初始盒子完全一致说明目标数密度相同。
- 每条链净电荷 |q| < 4×10⁻⁴，体系净电荷 |Σq| ≤ 0.04，体系整体中性，pppm 长程静电处理正常。
- **9_ps 是最干净的对照**：除了电荷（max|Δq| = 0.0073）以外一切都相同，
  密度仍差 4.94% —— 这个差异完全由电荷模型引起。

## 重跑

```bash
cd C:/schoolwork/Projects/mappingff/validation/longer/analysis
C:/Users/85897/.conda/envs/mappingff/python.exe analyze_density_rg.py
```

约 20 秒（需读 2.2 GB 轨迹）。依赖：numpy。
