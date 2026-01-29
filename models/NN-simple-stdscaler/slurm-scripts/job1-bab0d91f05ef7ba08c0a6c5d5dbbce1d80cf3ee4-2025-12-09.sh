#!/bin/bash

#SBATCH -e logs/job1-bab0d91f05ef7ba08c0a6c5d5dbbce1d80cf3ee4-2025-12-09.%J.err
#SBATCH -o logs/job1-bab0d91f05ef7ba08c0a6c5d5dbbce1d80cf3ee4-2025-12-09.%J.out
#SBATCH -J job1-bab0d91f05ef7ba08c0a6c5d5dbbce1d80cf3ee4-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_12_09_22_49