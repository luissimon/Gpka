#!/bin/bash

#SBATCH -e logs/job0-14345de7d046e3b337dbfa36da9f249339bc649f-2025-03-16.%J.err
#SBATCH -o logs/job0-14345de7d046e3b337dbfa36da9f249339bc649f-2025-03-16.%J.out
#SBATCH -J job0-14345de7d046e3b337dbfa36da9f249339bc649f-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_03_16_23_00