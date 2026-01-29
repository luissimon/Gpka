#!/bin/bash

#SBATCH -e logs/job6-b48b3b1a571292ce777c6b9c38c4aeeefe740eb5-2025-03-13.%J.err
#SBATCH -o logs/job6-b48b3b1a571292ce777c6b9c38c4aeeefe740eb5-2025-03-13.%J.out
#SBATCH -J job6-b48b3b1a571292ce777c6b9c38c4aeeefe740eb5-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 6 run_2025_03_13_11_05