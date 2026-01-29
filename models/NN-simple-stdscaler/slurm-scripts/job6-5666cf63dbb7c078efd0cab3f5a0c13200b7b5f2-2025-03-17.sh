#!/bin/bash

#SBATCH -e logs/job6-5666cf63dbb7c078efd0cab3f5a0c13200b7b5f2-2025-03-17.%J.err
#SBATCH -o logs/job6-5666cf63dbb7c078efd0cab3f5a0c13200b7b5f2-2025-03-17.%J.out
#SBATCH -J job6-5666cf63dbb7c078efd0cab3f5a0c13200b7b5f2-2025-03-17

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 6 run_2025_03_17_10_28