#!/bin/bash

#SBATCH -e logs/job1-7a0f3f2f656d30fb12a7e0e20ef49d5dd8fdbe16-2025-03-15.%J.err
#SBATCH -o logs/job1-7a0f3f2f656d30fb12a7e0e20ef49d5dd8fdbe16-2025-03-15.%J.out
#SBATCH -J job1-7a0f3f2f656d30fb12a7e0e20ef49d5dd8fdbe16-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_03_15_18_02