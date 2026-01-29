#!/bin/bash

#SBATCH -e logs/job3-ce9d47bd45eeb40b91d775ca91a11bbd7022428a-2025-03-16.%J.err
#SBATCH -o logs/job3-ce9d47bd45eeb40b91d775ca91a11bbd7022428a-2025-03-16.%J.out
#SBATCH -J job3-ce9d47bd45eeb40b91d775ca91a11bbd7022428a-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 3 run_2025_03_16_21_36