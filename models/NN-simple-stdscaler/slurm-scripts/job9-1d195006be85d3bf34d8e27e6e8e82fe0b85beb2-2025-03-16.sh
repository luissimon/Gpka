#!/bin/bash

#SBATCH -e logs/job9-1d195006be85d3bf34d8e27e6e8e82fe0b85beb2-2025-03-16.%J.err
#SBATCH -o logs/job9-1d195006be85d3bf34d8e27e6e8e82fe0b85beb2-2025-03-16.%J.out
#SBATCH -J job9-1d195006be85d3bf34d8e27e6e8e82fe0b85beb2-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 9 run_2025_03_16_21_36