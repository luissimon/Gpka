#!/bin/bash

#SBATCH -e logs/job8-d3c3f3b6283ac1c0fcc0973344ec9f6883cc7ce4-2025-03-15.%J.err
#SBATCH -o logs/job8-d3c3f3b6283ac1c0fcc0973344ec9f6883cc7ce4-2025-03-15.%J.out
#SBATCH -J job8-d3c3f3b6283ac1c0fcc0973344ec9f6883cc7ce4-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_15_18_15