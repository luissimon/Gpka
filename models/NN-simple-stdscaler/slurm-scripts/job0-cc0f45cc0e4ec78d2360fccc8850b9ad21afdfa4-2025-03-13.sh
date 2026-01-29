#!/bin/bash

#SBATCH -e logs/job0-cc0f45cc0e4ec78d2360fccc8850b9ad21afdfa4-2025-03-13.%J.err
#SBATCH -o logs/job0-cc0f45cc0e4ec78d2360fccc8850b9ad21afdfa4-2025-03-13.%J.out
#SBATCH -J job0-cc0f45cc0e4ec78d2360fccc8850b9ad21afdfa4-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 0 run_2025_03_13_11_05