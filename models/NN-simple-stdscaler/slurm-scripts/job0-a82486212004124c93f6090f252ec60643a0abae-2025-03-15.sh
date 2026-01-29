#!/bin/bash

#SBATCH -e logs/job0-a82486212004124c93f6090f252ec60643a0abae-2025-03-15.%J.err
#SBATCH -o logs/job0-a82486212004124c93f6090f252ec60643a0abae-2025-03-15.%J.out
#SBATCH -J job0-a82486212004124c93f6090f252ec60643a0abae-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_03_15_18_07