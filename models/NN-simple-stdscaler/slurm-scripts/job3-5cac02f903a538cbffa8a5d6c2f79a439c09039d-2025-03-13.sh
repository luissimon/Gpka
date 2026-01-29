#!/bin/bash

#SBATCH -e logs/job3-5cac02f903a538cbffa8a5d6c2f79a439c09039d-2025-03-13.%J.err
#SBATCH -o logs/job3-5cac02f903a538cbffa8a5d6c2f79a439c09039d-2025-03-13.%J.out
#SBATCH -J job3-5cac02f903a538cbffa8a5d6c2f79a439c09039d-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 3 run_2025_03_13_11_14