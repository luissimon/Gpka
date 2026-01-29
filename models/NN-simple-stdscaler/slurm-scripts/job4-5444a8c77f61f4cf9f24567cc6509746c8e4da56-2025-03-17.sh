#!/bin/bash

#SBATCH -e logs/job4-5444a8c77f61f4cf9f24567cc6509746c8e4da56-2025-03-17.%J.err
#SBATCH -o logs/job4-5444a8c77f61f4cf9f24567cc6509746c8e4da56-2025-03-17.%J.out
#SBATCH -J job4-5444a8c77f61f4cf9f24567cc6509746c8e4da56-2025-03-17

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 4 run_2025_03_17_10_28