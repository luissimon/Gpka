#!/bin/bash

#SBATCH -e logs/job1-fce10c3407c66632444e0b33fe59d7d3330399bf-2025-03-17.%J.err
#SBATCH -o logs/job1-fce10c3407c66632444e0b33fe59d7d3330399bf-2025-03-17.%J.out
#SBATCH -J job1-fce10c3407c66632444e0b33fe59d7d3330399bf-2025-03-17

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_03_17_10_28