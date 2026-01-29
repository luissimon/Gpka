#!/bin/bash

#SBATCH -e logs/job7-e61ee5642f26feb0ad9a065f3281f8ccb9b47cab-2025-03-15.%J.err
#SBATCH -o logs/job7-e61ee5642f26feb0ad9a065f3281f8ccb9b47cab-2025-03-15.%J.out
#SBATCH -J job7-e61ee5642f26feb0ad9a065f3281f8ccb9b47cab-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_03_15_18_07