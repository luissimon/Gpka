#!/bin/bash

#SBATCH -e logs/job8-eb1aa4b9f347a6faf1aabbd61fed36bbe5b9b9de-2025-03-13.%J.err
#SBATCH -o logs/job8-eb1aa4b9f347a6faf1aabbd61fed36bbe5b9b9de-2025-03-13.%J.out
#SBATCH -J job8-eb1aa4b9f347a6faf1aabbd61fed36bbe5b9b9de-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 8 run_2025_03_13_11_34