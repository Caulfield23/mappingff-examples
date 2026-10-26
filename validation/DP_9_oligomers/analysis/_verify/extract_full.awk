# Extract full thermo rows for all blocks: block \t step \t temp \t press \t density \t lx \t ly \t lz
BEGIN { FS=" " ; inblock=0 ; bidx=0 }
{
  if (inblock == 0) {
    if ($1 == "Step" && $0 ~ /Density/) {
      ti=0; pi=0; di=0; xi=0; yi=0; zi=0
      for (i=1;i<=NF;i++) {
        if ($i=="Temp") ti=i; if ($i=="Press") pi=i; if ($i=="Density") di=i
        if ($i=="Lx") xi=i; if ($i=="Ly") yi=i; if ($i=="Lz") zi=i
      }
      inblock=1; bidx++
    }
    next
  }
  if ($1 ~ /^[0-9]+$/ && NF>=di && NF>=di) {
    printf "%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n", bidx, $1, $ti, $pi, $di, $xi, $yi, $zi
  } else { inblock=0 }
}
