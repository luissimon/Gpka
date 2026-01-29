#!/bin/bash

#SBATCH -e logs/job6-82ae8d4264caeac2e897ea15eb8a2590e11ab0b7-2025-12-09.%J.err
#SBATCH -o logs/job6-82ae8d4264caeac2e897ea15eb8a2590e11ab0b7-2025-12-09.%J.out
#SBATCH -J job6-82ae8d4264caeac2e897ea15eb8a2590e11ab0b7-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 6 run_2025_12_09_11_57