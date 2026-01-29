#!/bin/bash

#SBATCH -e logs/job2-96a5ff5fcc8a61371907ff63e0cef64cd17845e6-2025-03-13.%J.err
#SBATCH -o logs/job2-96a5ff5fcc8a61371907ff63e0cef64cd17845e6-2025-03-13.%J.out
#SBATCH -J job2-96a5ff5fcc8a61371907ff63e0cef64cd17845e6-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_13_11_14