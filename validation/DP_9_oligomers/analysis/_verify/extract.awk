# Independent thermo extractor: awk only, no reuse of analysis code.
# Emits: block_index, step, Density, Lx  for every thermo row of every
# thermo block that has a "Density" column.
#
# Blocks are delimited by the thermo table HEADER (a line starting with
# "Step" and containing "Density"), never by step gaps.

BEGIN { FS=" " ; inblock=0 ; bidx=0 ; nrows=0 }

/^[ \t]*Step[ \t]/ && /Density/ {
    # header line of a density-bearing thermo table
    di=0 ; li=0
    for (i=1; i<=NF; i++) {
        if ($i == "Density") di=i
        if ($i == "Lx")     li=i
    }
    if (di==0 || li==0) { print "ERROR: bad header" > "/dev/stderr"; exit 2 }
    inblock=1 ; bidx++ ; nrows=0 ; hdrline=NR
    next
}

inblock {
    if ($1 ~ /^[0-9]+$/ && NF >= di && NF >= li) {
        printf "%d\t%d\t%s\t%s\n", bidx, $1, $di, $li
        nrows++
    } else {
        # non-numeric, non-empty line ends the block
        print "BLOCK\t" bidx "\tfirst_row_line\t" hdrline "\tnrows\t" nrows > "/dev/stderr"
        inblock=0
    }
}

END {
    if (inblock) print "BLOCK\t" bidx "\tfirst_row_line\t" hdrline "\tnrows\t" nrows > "/dev/stderr"
}
