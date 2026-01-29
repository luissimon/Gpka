#!/bin/bash

#SBATCH -e logs/job8-b9e170d0aa566b664028dcc68611ffef9c3e610c-2025-12-09.%J.err
#SBATCH -o logs/job8-b9e170d0aa566b664028dcc68611ffef9c3e610c-2025-12-09.%J.out
#SBATCH -J job8-b9e170d0aa566b664028dcc68611ffef9c3e610c-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_12_09_11_31