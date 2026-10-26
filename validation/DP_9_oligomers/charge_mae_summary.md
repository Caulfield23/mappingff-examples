# Charge consistency: LigParGen vs MappingFF

Generated 2026-10-23 20:30 by `compare_lammps_charges.py`.

Charge MAE is the mean absolute per-atom charge difference (MappingFF - LigParGen) after atom matching.

| Molecule | Atoms | Charge MAE (e) | LigParGen total q (e) | MappingFF total q (e) | Geometry RMSD (A) | Match |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 9_peo | 66 | 0.01819367 | -0.00000000 | 0.00000200 | 1.3784 | topology |
| 9_pmma | 137 | 0.03075496 | -0.00000000 | 0.00002500 | 2.5907 | topology |
| 9_ps | 146 | 0.00144279 | -0.00040000 | -0.00002300 | 0.0004 | geometry |
| 9_pva | 65 | 0.02126974 | 0.00000000 | -0.00002100 | 0.9407 | topology |

Mean charge MAE over 4 molecule(s): 0.01791529 e
Largest charge MAE: 9_pmma (0.03075496 e)
