#!/bin/bash
# Per-block step ranges, computed purely with awk/grep.
ROOT="C:/schoolwork/Projects/mappingff/validation/longer"
for s in ligpargen mappingff; do
  for y in 9_peo 9_pmma 9_ps 9_pva; do
    echo "=== $s/$y ==="
    awk -f "$ROOT/analysis/_verify/extract.awk" "$ROOT/$s/$y/log.lammps" 2>/dev/null |
    awk -F'\t' '
      { b[$1]++; if (!($1 in first)) first[$1]=$2; last[$1]=$2;
        if (NF>=4) { d[$1]=$3; l[$1]=$4; if (!($1 in dmin) || $3<dmin[$1]) dmin[$1]=$3;
                     if (!($1 in dmax) || $3>dmax[$1]) dmax[$1]=$3;
                     if (!($1 in lmin) || $4<lmin[$1]) lmin[$1]=$4;
                     if (!($1 in lmax) || $4>lmax[$1]) lmax[$1]=$4 } }
      END { for (i=1;i<=5;i++) printf "  block %d: n=%d  step %d..%d   rho[%.6f..%.6f]  Lx[%.6f..%.6f]\n",
            i, b[i], first[i], last[i], dmin[i], dmax[i], lmin[i], lmax[i] }'
  done
done
