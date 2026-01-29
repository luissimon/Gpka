#!/bin/bash

#SBATCH -e logs/job5-ee28e060ddcaef869e6da04999b32fc11169424f-2025-03-13.%J.err
#SBATCH -o logs/job5-ee28e060ddcaef869e6da04999b32fc11169424f-2025-03-13.%J.out
#SBATCH -J job5-ee28e060ddcaef869e6da04999b32fc11169424f-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_03_13_11_14