#!/bin/bash

#SBATCH -e logs/job4-411473fa92cfa42b0d7ea0674f51e2c0109d73af-2025-12-09.%J.err
#SBATCH -o logs/job4-411473fa92cfa42b0d7ea0674f51e2c0109d73af-2025-12-09.%J.out
#SBATCH -J job4-411473fa92cfa42b0d7ea0674f51e2c0109d73af-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 4 run_2025_12_09_11_57