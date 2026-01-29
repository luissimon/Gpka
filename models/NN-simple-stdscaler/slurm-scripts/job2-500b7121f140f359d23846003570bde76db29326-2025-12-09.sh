#!/bin/bash

#SBATCH -e logs/job2-500b7121f140f359d23846003570bde76db29326-2025-12-09.%J.err
#SBATCH -o logs/job2-500b7121f140f359d23846003570bde76db29326-2025-12-09.%J.out
#SBATCH -J job2-500b7121f140f359d23846003570bde76db29326-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_12_09_11_57