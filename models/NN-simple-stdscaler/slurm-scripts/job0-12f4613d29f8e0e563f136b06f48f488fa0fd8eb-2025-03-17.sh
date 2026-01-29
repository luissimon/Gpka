#!/bin/bash

#SBATCH -e logs/job0-12f4613d29f8e0e563f136b06f48f488fa0fd8eb-2025-03-17.%J.err
#SBATCH -o logs/job0-12f4613d29f8e0e563f136b06f48f488fa0fd8eb-2025-03-17.%J.out
#SBATCH -J job0-12f4613d29f8e0e563f136b06f48f488fa0fd8eb-2025-03-17

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_03_17_10_28