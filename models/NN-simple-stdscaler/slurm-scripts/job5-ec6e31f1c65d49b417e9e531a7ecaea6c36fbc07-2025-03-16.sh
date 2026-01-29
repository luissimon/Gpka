#!/bin/bash

#SBATCH -e logs/job5-ec6e31f1c65d49b417e9e531a7ecaea6c36fbc07-2025-03-16.%J.err
#SBATCH -o logs/job5-ec6e31f1c65d49b417e9e531a7ecaea6c36fbc07-2025-03-16.%J.out
#SBATCH -J job5-ec6e31f1c65d49b417e9e531a7ecaea6c36fbc07-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_03_16_21_36