#!/bin/bash

#SBATCH -e logs/job6-27136403f32b8aedd16c09613b2d484528176d60-2025-12-09.%J.err
#SBATCH -o logs/job6-27136403f32b8aedd16c09613b2d484528176d60-2025-12-09.%J.out
#SBATCH -J job6-27136403f32b8aedd16c09613b2d484528176d60-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 6 run_2025_12_09_22_49