#!/bin/bash

#SBATCH -e logs/job1-d63a8b1f5a9a703b62a0e447fb4e8262f8249d7b-2025-03-15.%J.err
#SBATCH -o logs/job1-d63a8b1f5a9a703b62a0e447fb4e8262f8249d7b-2025-03-15.%J.out
#SBATCH -J job1-d63a8b1f5a9a703b62a0e447fb4e8262f8249d7b-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_03_15_18_15