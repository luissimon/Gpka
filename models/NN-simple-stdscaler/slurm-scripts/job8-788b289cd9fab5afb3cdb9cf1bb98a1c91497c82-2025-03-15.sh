#!/bin/bash

#SBATCH -e logs/job8-788b289cd9fab5afb3cdb9cf1bb98a1c91497c82-2025-03-15.%J.err
#SBATCH -o logs/job8-788b289cd9fab5afb3cdb9cf1bb98a1c91497c82-2025-03-15.%J.out
#SBATCH -J job8-788b289cd9fab5afb3cdb9cf1bb98a1c91497c82-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_15_18_02