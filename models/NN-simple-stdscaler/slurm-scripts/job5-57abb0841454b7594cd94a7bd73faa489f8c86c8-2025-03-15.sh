#!/bin/bash

#SBATCH -e logs/job5-57abb0841454b7594cd94a7bd73faa489f8c86c8-2025-03-15.%J.err
#SBATCH -o logs/job5-57abb0841454b7594cd94a7bd73faa489f8c86c8-2025-03-15.%J.out
#SBATCH -J job5-57abb0841454b7594cd94a7bd73faa489f8c86c8-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_03_15_18_07