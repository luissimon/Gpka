#!/bin/bash

#SBATCH -e logs/job9-f962d5ec97781c438cdab3ffe1c83a9143c2e075-2025-03-15.%J.err
#SBATCH -o logs/job9-f962d5ec97781c438cdab3ffe1c83a9143c2e075-2025-03-15.%J.out
#SBATCH -J job9-f962d5ec97781c438cdab3ffe1c83a9143c2e075-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 9 run_2025_03_15_18_02