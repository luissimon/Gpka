#!/bin/bash

#SBATCH -e logs/job1-7675defb165910540fa6e619e1ce2ff0767f78f5-2025-03-13.%J.err
#SBATCH -o logs/job1-7675defb165910540fa6e619e1ce2ff0767f78f5-2025-03-13.%J.out
#SBATCH -J job1-7675defb165910540fa6e619e1ce2ff0767f78f5-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_03_13_11_05