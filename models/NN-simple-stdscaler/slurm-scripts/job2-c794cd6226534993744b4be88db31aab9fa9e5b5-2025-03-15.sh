#!/bin/bash

#SBATCH -e logs/job2-c794cd6226534993744b4be88db31aab9fa9e5b5-2025-03-15.%J.err
#SBATCH -o logs/job2-c794cd6226534993744b4be88db31aab9fa9e5b5-2025-03-15.%J.out
#SBATCH -J job2-c794cd6226534993744b4be88db31aab9fa9e5b5-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_15_18_07