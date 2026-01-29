#!/bin/bash

#SBATCH -e logs/job4-f1ff798161507e167a68e88b74ebaa363229c609-2025-03-16.%J.err
#SBATCH -o logs/job4-f1ff798161507e167a68e88b74ebaa363229c609-2025-03-16.%J.out
#SBATCH -J job4-f1ff798161507e167a68e88b74ebaa363229c609-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 4 run_2025_03_16_23_00