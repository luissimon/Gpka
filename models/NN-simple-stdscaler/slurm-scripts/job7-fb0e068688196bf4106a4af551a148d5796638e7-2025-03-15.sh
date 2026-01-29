#!/bin/bash

#SBATCH -e logs/job7-fb0e068688196bf4106a4af551a148d5796638e7-2025-03-15.%J.err
#SBATCH -o logs/job7-fb0e068688196bf4106a4af551a148d5796638e7-2025-03-15.%J.out
#SBATCH -J job7-fb0e068688196bf4106a4af551a148d5796638e7-2025-03-15

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_03_15_18_02