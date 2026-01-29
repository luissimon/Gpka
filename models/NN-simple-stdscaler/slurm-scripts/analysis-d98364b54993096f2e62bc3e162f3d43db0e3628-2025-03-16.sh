#!/bin/bash

#SBATCH -e logs/analysis-d98364b54993096f2e62bc3e162f3d43db0e3628-2025-03-16.%J.err
#SBATCH -o logs/analysis-d98364b54993096f2e62bc3e162f3d43db0e3628-2025-03-16.%J.out
#SBATCH -J analysis-d98364b54993096f2e62bc3e162f3d43db0e3628-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py analyze 