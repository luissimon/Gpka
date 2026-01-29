#!/bin/bash

#SBATCH -e logs/job7-f98fb247e22b9a5bd216a9741724e470e5a2debb-2025-03-16.%J.err
#SBATCH -o logs/job7-f98fb247e22b9a5bd216a9741724e470e5a2debb-2025-03-16.%J.out
#SBATCH -J job7-f98fb247e22b9a5bd216a9741724e470e5a2debb-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_03_16_23_00