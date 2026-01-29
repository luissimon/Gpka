#!/bin/bash

#SBATCH -e logs/job5-5f07052b8248e4f4199f186e9d50c94146afff40-2025-03-17.%J.err
#SBATCH -o logs/job5-5f07052b8248e4f4199f186e9d50c94146afff40-2025-03-17.%J.out
#SBATCH -J job5-5f07052b8248e4f4199f186e9d50c94146afff40-2025-03-17

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_03_17_10_28