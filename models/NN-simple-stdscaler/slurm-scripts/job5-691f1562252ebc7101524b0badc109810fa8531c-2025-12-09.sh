#!/bin/bash

#SBATCH -e logs/job5-691f1562252ebc7101524b0badc109810fa8531c-2025-12-09.%J.err
#SBATCH -o logs/job5-691f1562252ebc7101524b0badc109810fa8531c-2025-12-09.%J.out
#SBATCH -J job5-691f1562252ebc7101524b0badc109810fa8531c-2025-12-09

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_12_09_11_31