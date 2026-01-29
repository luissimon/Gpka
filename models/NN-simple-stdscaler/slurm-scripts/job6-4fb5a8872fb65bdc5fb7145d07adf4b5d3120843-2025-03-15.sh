#!/bin/bash

#SBATCH -e logs/job6-4fb5a8872fb65bdc5fb7145d07adf4b5d3120843-2025-03-15.%J.err
#SBATCH -o logs/job6-4fb5a8872fb65bdc5fb7145d07adf4b5d3120843-2025-03-15.%J.out
#SBATCH -J job6-4fb5a8872fb65bdc5fb7145d07adf4b5d3120843-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 6 run_2025_03_15_18_07