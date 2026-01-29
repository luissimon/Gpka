#!/bin/bash

#SBATCH -e logs/job1-8f5d554d7b75dd3045382d3c77b5a3814b0f785f-2025-03-16.%J.err
#SBATCH -o logs/job1-8f5d554d7b75dd3045382d3c77b5a3814b0f785f-2025-03-16.%J.out
#SBATCH -J job1-8f5d554d7b75dd3045382d3c77b5a3814b0f785f-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_03_16_21_36