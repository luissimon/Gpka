#!/bin/bash

#SBATCH -e logs/job5-d784e54fb942a2b1783cf157c3aead04dfae98ea-2025-03-16.%J.err
#SBATCH -o logs/job5-d784e54fb942a2b1783cf157c3aead04dfae98ea-2025-03-16.%J.out
#SBATCH -J job5-d784e54fb942a2b1783cf157c3aead04dfae98ea-2025-03-16

#SBATCH --ntasks-per-node=14
#SBATCH --nodes=1
#SBATCH --exclude=atwood
#SBATCH --time=84:00:00

set -eo pipefail -o nounset


###
 ./simple-NN.py run_node 5 run_2025_03_16_23_00