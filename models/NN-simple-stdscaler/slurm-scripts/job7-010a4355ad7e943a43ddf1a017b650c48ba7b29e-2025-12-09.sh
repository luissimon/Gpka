#!/bin/bash

#SBATCH -e logs/job7-010a4355ad7e943a43ddf1a017b650c48ba7b29e-2025-12-09.%J.err
#SBATCH -o logs/job7-010a4355ad7e943a43ddf1a017b650c48ba7b29e-2025-12-09.%J.out
#SBATCH -J job7-010a4355ad7e943a43ddf1a017b650c48ba7b29e-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_12_09_11_31