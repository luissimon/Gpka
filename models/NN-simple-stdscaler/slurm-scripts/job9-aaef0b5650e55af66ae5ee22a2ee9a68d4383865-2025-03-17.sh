#!/bin/bash

#SBATCH -e logs/job9-aaef0b5650e55af66ae5ee22a2ee9a68d4383865-2025-03-17.%J.err
#SBATCH -o logs/job9-aaef0b5650e55af66ae5ee22a2ee9a68d4383865-2025-03-17.%J.out
#SBATCH -J job9-aaef0b5650e55af66ae5ee22a2ee9a68d4383865-2025-03-17

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 9 run_2025_03_17_10_28