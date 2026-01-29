#!/bin/bash

#SBATCH -e logs/job6-dd2613ca82bb8073a457f4db51eb2d0f878074c4-2025-03-16.%J.err
#SBATCH -o logs/job6-dd2613ca82bb8073a457f4db51eb2d0f878074c4-2025-03-16.%J.out
#SBATCH -J job6-dd2613ca82bb8073a457f4db51eb2d0f878074c4-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 6 run_2025_03_16_21_36