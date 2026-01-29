#!/bin/bash

#SBATCH -e logs/job8-ad86ee6cf78e802020ff4b18e03add7ec1b228b1-2025-03-16.%J.err
#SBATCH -o logs/job8-ad86ee6cf78e802020ff4b18e03add7ec1b228b1-2025-03-16.%J.out
#SBATCH -J job8-ad86ee6cf78e802020ff4b18e03add7ec1b228b1-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_16_21_36