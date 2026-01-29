#!/bin/bash

#SBATCH -e logs/job4-037aaed2e072a01d577e4fb5cc63771d221a0813-2025-03-13.%J.err
#SBATCH -o logs/job4-037aaed2e072a01d577e4fb5cc63771d221a0813-2025-03-13.%J.out
#SBATCH -J job4-037aaed2e072a01d577e4fb5cc63771d221a0813-2025-03-13

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 4 run_2025_03_13_11_05