#!/bin/bash

#SBATCH -e logs/job0-9bf60c364bc6ff886b426df7b404ff16bb2da438-2025-03-15.%J.err
#SBATCH -o logs/job0-9bf60c364bc6ff886b426df7b404ff16bb2da438-2025-03-15.%J.out
#SBATCH -J job0-9bf60c364bc6ff886b426df7b404ff16bb2da438-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_03_15_18_02