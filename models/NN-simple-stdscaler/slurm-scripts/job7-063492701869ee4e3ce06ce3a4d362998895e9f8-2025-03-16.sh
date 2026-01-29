#!/bin/bash

#SBATCH -e logs/job7-063492701869ee4e3ce06ce3a4d362998895e9f8-2025-03-16.%J.err
#SBATCH -o logs/job7-063492701869ee4e3ce06ce3a4d362998895e9f8-2025-03-16.%J.out
#SBATCH -J job7-063492701869ee4e3ce06ce3a4d362998895e9f8-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_03_16_21_36