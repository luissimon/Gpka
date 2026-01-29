#!/bin/bash

#SBATCH -e logs/job2-8823e8946ebe973f9a76bcf1de87aed04161aa45-2025-03-16.%J.err
#SBATCH -o logs/job2-8823e8946ebe973f9a76bcf1de87aed04161aa45-2025-03-16.%J.out
#SBATCH -J job2-8823e8946ebe973f9a76bcf1de87aed04161aa45-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_16_23_00