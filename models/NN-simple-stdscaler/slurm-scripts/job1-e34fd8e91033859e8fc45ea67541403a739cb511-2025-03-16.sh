#!/bin/bash

#SBATCH -e logs/job1-e34fd8e91033859e8fc45ea67541403a739cb511-2025-03-16.%J.err
#SBATCH -o logs/job1-e34fd8e91033859e8fc45ea67541403a739cb511-2025-03-16.%J.out
#SBATCH -J job1-e34fd8e91033859e8fc45ea67541403a739cb511-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 1 run_2025_03_16_23_00