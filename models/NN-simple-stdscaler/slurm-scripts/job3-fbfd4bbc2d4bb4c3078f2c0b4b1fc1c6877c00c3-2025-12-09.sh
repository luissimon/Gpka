#!/bin/bash

#SBATCH -e logs/job3-fbfd4bbc2d4bb4c3078f2c0b4b1fc1c6877c00c3-2025-12-09.%J.err
#SBATCH -o logs/job3-fbfd4bbc2d4bb4c3078f2c0b4b1fc1c6877c00c3-2025-12-09.%J.out
#SBATCH -J job3-fbfd4bbc2d4bb4c3078f2c0b4b1fc1c6877c00c3-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 3 run_2025_12_09_11_31