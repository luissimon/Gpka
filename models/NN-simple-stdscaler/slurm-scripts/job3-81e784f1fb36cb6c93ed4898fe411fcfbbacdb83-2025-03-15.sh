#!/bin/bash

#SBATCH -e logs/job3-81e784f1fb36cb6c93ed4898fe411fcfbbacdb83-2025-03-15.%J.err
#SBATCH -o logs/job3-81e784f1fb36cb6c93ed4898fe411fcfbbacdb83-2025-03-15.%J.out
#SBATCH -J job3-81e784f1fb36cb6c93ed4898fe411fcfbbacdb83-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 3 run_2025_03_15_18_15