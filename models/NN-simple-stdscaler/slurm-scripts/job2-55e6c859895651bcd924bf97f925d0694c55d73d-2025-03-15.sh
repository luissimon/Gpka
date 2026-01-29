#!/bin/bash

#SBATCH -e logs/job2-55e6c859895651bcd924bf97f925d0694c55d73d-2025-03-15.%J.err
#SBATCH -o logs/job2-55e6c859895651bcd924bf97f925d0694c55d73d-2025-03-15.%J.out
#SBATCH -J job2-55e6c859895651bcd924bf97f925d0694c55d73d-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_15_18_02