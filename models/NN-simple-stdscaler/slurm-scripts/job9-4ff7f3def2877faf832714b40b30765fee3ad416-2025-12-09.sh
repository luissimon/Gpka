#!/bin/bash

#SBATCH -e logs/job9-4ff7f3def2877faf832714b40b30765fee3ad416-2025-12-09.%J.err
#SBATCH -o logs/job9-4ff7f3def2877faf832714b40b30765fee3ad416-2025-12-09.%J.out
#SBATCH -J job9-4ff7f3def2877faf832714b40b30765fee3ad416-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 9 run_2025_12_09_11_31