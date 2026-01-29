#!/bin/bash

#SBATCH -e logs/job1-04dd8eba46eaab8cdbc6f6635daa18ca8b1ef3fc-2025-03-15.%J.err
#SBATCH -o logs/job1-04dd8eba46eaab8cdbc6f6635daa18ca8b1ef3fc-2025-03-15.%J.out
#SBATCH -J job1-04dd8eba46eaab8cdbc6f6635daa18ca8b1ef3fc-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_03_15_18_07