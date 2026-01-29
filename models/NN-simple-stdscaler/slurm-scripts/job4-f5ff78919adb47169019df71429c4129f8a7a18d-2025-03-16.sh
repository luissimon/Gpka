#!/bin/bash

#SBATCH -e logs/job4-f5ff78919adb47169019df71429c4129f8a7a18d-2025-03-16.%J.err
#SBATCH -o logs/job4-f5ff78919adb47169019df71429c4129f8a7a18d-2025-03-16.%J.out
#SBATCH -J job4-f5ff78919adb47169019df71429c4129f8a7a18d-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 4 run_2025_03_16_21_36