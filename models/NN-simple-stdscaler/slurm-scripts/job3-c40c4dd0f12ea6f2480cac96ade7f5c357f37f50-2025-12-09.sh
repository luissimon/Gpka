#!/bin/bash

#SBATCH -e logs/job3-c40c4dd0f12ea6f2480cac96ade7f5c357f37f50-2025-12-09.%J.err
#SBATCH -o logs/job3-c40c4dd0f12ea6f2480cac96ade7f5c357f37f50-2025-12-09.%J.out
#SBATCH -J job3-c40c4dd0f12ea6f2480cac96ade7f5c357f37f50-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 3 run_2025_12_09_11_57