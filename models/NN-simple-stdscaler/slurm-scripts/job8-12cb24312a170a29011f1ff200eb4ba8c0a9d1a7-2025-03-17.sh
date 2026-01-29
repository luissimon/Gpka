#!/bin/bash

#SBATCH -e logs/job8-12cb24312a170a29011f1ff200eb4ba8c0a9d1a7-2025-03-17.%J.err
#SBATCH -o logs/job8-12cb24312a170a29011f1ff200eb4ba8c0a9d1a7-2025-03-17.%J.out
#SBATCH -J job8-12cb24312a170a29011f1ff200eb4ba8c0a9d1a7-2025-03-17

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_17_10_28