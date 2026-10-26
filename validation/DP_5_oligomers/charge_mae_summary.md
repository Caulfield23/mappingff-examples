# Charge consistency: LigParGen vs MappingFF

Generated 2026-10-23 20:30 by `compare_lammps_charges.py`.

Charge MAE is the mean absolute per-atom charge difference (MappingFF - LigParGen) after atom matching.

| Molecule | Atoms | Charge MAE (e) | LigParGen total q (e) | MappingFF total q (e) | Geometry RMSD (A) | Match |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 5_peo | 38 | 0.01164211 | -0.00000000 | -0.00000200 | 0.0000 | geometry |
| 5_pmma | 77 | 0.00251288 | -0.00030000 | 0.00001200 | 0.0000 | geometry |
| 5_ps | 82 | 0.00076305 | 0.00010000 | 0.00001800 | 0.0000 | geometry |
| 5_pva | 37 | 0.02738905 | 0.00000000 | -0.00000100 | 0.0032 | geometry |

Mean charge MAE over 4 molecule(s): 0.01057677 e
Largest charge MAE: 5_pva (0.02738905 e)
