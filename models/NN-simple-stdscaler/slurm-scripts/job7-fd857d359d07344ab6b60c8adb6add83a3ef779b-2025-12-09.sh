#!/bin/bash

#SBATCH -e logs/job7-fd857d359d07344ab6b60c8adb6add83a3ef779b-2025-12-09.%J.err
#SBATCH -o logs/job7-fd857d359d07344ab6b60c8adb6add83a3ef779b-2025-12-09.%J.out
#SBATCH -J job7-fd857d359d07344ab6b60c8adb6add83a3ef779b-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 7 run_2025_12_09_11_57