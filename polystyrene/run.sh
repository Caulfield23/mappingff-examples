#!/bin/bash
# Polystyrene example: Build database and parameterize target molecule

set -e

# Step 1: Build parameter database
mappingff build samples/ -d polystyrene.db

# Step 2: Parameterize target molecule
mappingff par 200_polystyrene.mol -d polystyrene.db -c 0.0 -v