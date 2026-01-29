#!/bin/bash

#SBATCH -e logs/job3-38b9306bd4cc53af9e3250460c7e91f1961f2d8c-2025-03-15.%J.err
#SBATCH -o logs/job3-38b9306bd4cc53af9e3250460c7e91f1961f2d8c-2025-03-15.%J.out
#SBATCH -J job3-38b9306bd4cc53af9e3250460c7e91f1961f2d8c-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 3 run_2025_03_15_18_07