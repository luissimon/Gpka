#!/bin/bash

#SBATCH -e logs/job8-d1c8a17ecc5b50a2932400e090d0c43d56d7a35f-2025-03-13.%J.err
#SBATCH -o logs/job8-d1c8a17ecc5b50a2932400e090d0c43d56d7a35f-2025-03-13.%J.out
#SBATCH -J job8-d1c8a17ecc5b50a2932400e090d0c43d56d7a35f-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_13_11_14