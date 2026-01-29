#!/bin/bash

#SBATCH -e logs/job0-d420f62b740395a4865318704799de7c0c807957-2025-12-09.%J.err
#SBATCH -o logs/job0-d420f62b740395a4865318704799de7c0c807957-2025-12-09.%J.out
#SBATCH -J job0-d420f62b740395a4865318704799de7c0c807957-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_12_09_11_57