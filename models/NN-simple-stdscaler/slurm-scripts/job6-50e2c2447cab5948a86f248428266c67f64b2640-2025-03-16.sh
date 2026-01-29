#!/bin/bash

#SBATCH -e logs/job6-50e2c2447cab5948a86f248428266c67f64b2640-2025-03-16.%J.err
#SBATCH -o logs/job6-50e2c2447cab5948a86f248428266c67f64b2640-2025-03-16.%J.out
#SBATCH -J job6-50e2c2447cab5948a86f248428266c67f64b2640-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 6 run_2025_03_16_23_00