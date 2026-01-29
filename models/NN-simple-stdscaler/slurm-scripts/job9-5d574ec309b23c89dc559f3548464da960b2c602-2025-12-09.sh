#!/bin/bash

#SBATCH -e logs/job9-5d574ec309b23c89dc559f3548464da960b2c602-2025-12-09.%J.err
#SBATCH -o logs/job9-5d574ec309b23c89dc559f3548464da960b2c602-2025-12-09.%J.out
#SBATCH -J job9-5d574ec309b23c89dc559f3548464da960b2c602-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 9 run_2025_12_09_22_49