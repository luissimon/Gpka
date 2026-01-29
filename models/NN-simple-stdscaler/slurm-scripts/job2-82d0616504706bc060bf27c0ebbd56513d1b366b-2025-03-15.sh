#!/bin/bash

#SBATCH -e logs/job2-82d0616504706bc060bf27c0ebbd56513d1b366b-2025-03-15.%J.err
#SBATCH -o logs/job2-82d0616504706bc060bf27c0ebbd56513d1b366b-2025-03-15.%J.out
#SBATCH -J job2-82d0616504706bc060bf27c0ebbd56513d1b366b-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_15_18_03