#!/bin/bash

#SBATCH -e logs/job8-c948d65ed7adcc6d0a57603dd8662f0552b3efb9-2025-03-15.%J.err
#SBATCH -o logs/job8-c948d65ed7adcc6d0a57603dd8662f0552b3efb9-2025-03-15.%J.out
#SBATCH -J job8-c948d65ed7adcc6d0a57603dd8662f0552b3efb9-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_15_18_07