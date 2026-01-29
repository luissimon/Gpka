#!/bin/bash

#SBATCH -e logs/job7-d6053c57f5e08ebb60b75e5dfbe968a21c7d3274-2025-03-13.%J.err
#SBATCH -o logs/job7-d6053c57f5e08ebb60b75e5dfbe968a21c7d3274-2025-03-13.%J.out
#SBATCH -J job7-d6053c57f5e08ebb60b75e5dfbe968a21c7d3274-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_03_13_11_14