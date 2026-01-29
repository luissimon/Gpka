#!/bin/bash

#SBATCH -e logs/job2-27e651faa387dc0224c4fa1a3f4b36221fc1b2ff-2025-03-16.%J.err
#SBATCH -o logs/job2-27e651faa387dc0224c4fa1a3f4b36221fc1b2ff-2025-03-16.%J.out
#SBATCH -J job2-27e651faa387dc0224c4fa1a3f4b36221fc1b2ff-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 2 run_2025_03_16_21_36