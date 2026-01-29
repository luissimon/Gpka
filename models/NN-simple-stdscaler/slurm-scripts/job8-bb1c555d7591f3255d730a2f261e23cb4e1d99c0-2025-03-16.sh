#!/bin/bash

#SBATCH -e logs/job8-bb1c555d7591f3255d730a2f261e23cb4e1d99c0-2025-03-16.%J.err
#SBATCH -o logs/job8-bb1c555d7591f3255d730a2f261e23cb4e1d99c0-2025-03-16.%J.out
#SBATCH -J job8-bb1c555d7591f3255d730a2f261e23cb4e1d99c0-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_16_23_00