#!/bin/bash
# PS-b-PMMA Block Copolymer Example
# Demonstrates parameterization of:
#   1. 50-b-50 diblock copolymer (50_ps_50_pmma.mol)
#   2. 10-block alternating copolymer (5_alt_20_ps_20_pmma.mol)

set -e

# Step 1: Build parameter database from styrene and MMA samples
mappingff build-db samples/ -d ps_pmma.db

# Step 2: Parameterize 50-b-50 PS-PMMA diblock copolymer
mappingff parameterize 50_ps_50_pmma.mol -d ps_pmma.db -c 0.0 -v

# Step 3: Parameterize 5-block alternating copolymer
mappingff parameterize 5_alt_20_ps_20_pmma.mol -d ps_pmma.db -c 0.0 -v