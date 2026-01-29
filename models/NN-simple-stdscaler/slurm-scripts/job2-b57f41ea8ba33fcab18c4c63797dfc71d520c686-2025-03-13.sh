#!/bin/bash

#SBATCH -e logs/job2-b57f41ea8ba33fcab18c4c63797dfc71d520c686-2025-03-13.%J.err
#SBATCH -o logs/job2-b57f41ea8ba33fcab18c4c63797dfc71d520c686-2025-03-13.%J.out
#SBATCH -J job2-b57f41ea8ba33fcab18c4c63797dfc71d520c686-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_13_11_34