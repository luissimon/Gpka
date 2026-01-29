#!/bin/bash

#SBATCH -e logs/job0-ffc212a8bbeebcc6ea9acd2a1ecbedb96fbd2fad-2025-03-16.%J.err
#SBATCH -o logs/job0-ffc212a8bbeebcc6ea9acd2a1ecbedb96fbd2fad-2025-03-16.%J.out
#SBATCH -J job0-ffc212a8bbeebcc6ea9acd2a1ecbedb96fbd2fad-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_03_16_21_36