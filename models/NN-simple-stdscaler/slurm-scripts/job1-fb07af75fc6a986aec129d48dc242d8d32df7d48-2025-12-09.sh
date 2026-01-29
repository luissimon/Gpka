#!/bin/bash

#SBATCH -e logs/job1-fb07af75fc6a986aec129d48dc242d8d32df7d48-2025-12-09.%J.err
#SBATCH -o logs/job1-fb07af75fc6a986aec129d48dc242d8d32df7d48-2025-12-09.%J.out
#SBATCH -J job1-fb07af75fc6a986aec129d48dc242d8d32df7d48-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_12_09_11_57