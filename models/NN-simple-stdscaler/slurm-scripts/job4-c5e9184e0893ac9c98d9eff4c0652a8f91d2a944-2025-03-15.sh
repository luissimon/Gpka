#!/bin/bash

#SBATCH -e logs/job4-c5e9184e0893ac9c98d9eff4c0652a8f91d2a944-2025-03-15.%J.err
#SBATCH -o logs/job4-c5e9184e0893ac9c98d9eff4c0652a8f91d2a944-2025-03-15.%J.out
#SBATCH -J job4-c5e9184e0893ac9c98d9eff4c0652a8f91d2a944-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 4 run_2025_03_15_18_07