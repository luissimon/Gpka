#!/bin/bash

#SBATCH -e logs/job2-1480a1274ff3a1f28b3aef63820e15554b2dfc12-2025-03-13.%J.err
#SBATCH -o logs/job2-1480a1274ff3a1f28b3aef63820e15554b2dfc12-2025-03-13.%J.out
#SBATCH -J job2-1480a1274ff3a1f28b3aef63820e15554b2dfc12-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_13_10_44