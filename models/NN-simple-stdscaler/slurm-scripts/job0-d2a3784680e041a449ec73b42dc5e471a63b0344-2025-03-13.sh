#!/bin/bash

#SBATCH -e logs/job0-d2a3784680e041a449ec73b42dc5e471a63b0344-2025-03-13.%J.err
#SBATCH -o logs/job0-d2a3784680e041a449ec73b42dc5e471a63b0344-2025-03-13.%J.out
#SBATCH -J job0-d2a3784680e041a449ec73b42dc5e471a63b0344-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_03_13_10_44