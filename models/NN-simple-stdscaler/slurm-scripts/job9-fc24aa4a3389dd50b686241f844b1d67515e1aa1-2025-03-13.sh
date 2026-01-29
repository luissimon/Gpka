#!/bin/bash

#SBATCH -e logs/job9-fc24aa4a3389dd50b686241f844b1d67515e1aa1-2025-03-13.%J.err
#SBATCH -o logs/job9-fc24aa4a3389dd50b686241f844b1d67515e1aa1-2025-03-13.%J.out
#SBATCH -J job9-fc24aa4a3389dd50b686241f844b1d67515e1aa1-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 9 run_2025_03_13_11_05