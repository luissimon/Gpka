#!/bin/bash

#SBATCH -e logs/job7-ab7c83c6872f9caba633f3ffa44c4e15203e362d-2025-03-13.%J.err
#SBATCH -o logs/job7-ab7c83c6872f9caba633f3ffa44c4e15203e362d-2025-03-13.%J.out
#SBATCH -J job7-ab7c83c6872f9caba633f3ffa44c4e15203e362d-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_03_13_10_44