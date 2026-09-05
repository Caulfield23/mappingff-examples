#!/bin/bash

for dir in /home/zhangzhao/zhangzhao/Projects/mappingff_run/validation/e_consistency/ligpargen/*/; do
    cd "$dir" && lmp -in in_e_consistency.lammps -var datafile *.lmp
done

for dir in /home/zhangzhao/zhangzhao/Projects/mappingff_run/validation/e_consistency/mappingff/*/; do
    cd "$dir" && lmp -in in_e_consistency.lammps -var datafile *.data
done
