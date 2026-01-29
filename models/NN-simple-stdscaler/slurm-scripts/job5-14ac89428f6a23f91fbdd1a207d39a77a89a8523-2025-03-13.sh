#!/bin/bash

#SBATCH -e logs/job5-14ac89428f6a23f91fbdd1a207d39a77a89a8523-2025-03-13.%J.err
#SBATCH -o logs/job5-14ac89428f6a23f91fbdd1a207d39a77a89a8523-2025-03-13.%J.out
#SBATCH -J job5-14ac89428f6a23f91fbdd1a207d39a77a89a8523-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_03_13_11_05