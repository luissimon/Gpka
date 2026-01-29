#!/bin/bash

#SBATCH -e logs/job3-0081d8341d01f892c0c4c4bd99df61ff3e08a1af-2025-03-13.%J.err
#SBATCH -o logs/job3-0081d8341d01f892c0c4c4bd99df61ff3e08a1af-2025-03-13.%J.out
#SBATCH -J job3-0081d8341d01f892c0c4c4bd99df61ff3e08a1af-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 3 run_2025_03_13_11_34