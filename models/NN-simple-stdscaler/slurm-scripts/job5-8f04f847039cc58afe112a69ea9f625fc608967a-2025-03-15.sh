#!/bin/bash

#SBATCH -e logs/job5-8f04f847039cc58afe112a69ea9f625fc608967a-2025-03-15.%J.err
#SBATCH -o logs/job5-8f04f847039cc58afe112a69ea9f625fc608967a-2025-03-15.%J.out
#SBATCH -J job5-8f04f847039cc58afe112a69ea9f625fc608967a-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_03_15_18_03