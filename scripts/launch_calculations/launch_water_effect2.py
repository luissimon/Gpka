#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
import copy
import numpy as np
import time 


if __name__ == "__main__":

    filename=sys.argv[1]
    #n_waters=sys.argv[2]
    t="#!/bin/bash\n#SBATCH -t 99:00:00\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=14\n#SBATCH --job-name="+sys.argv[1].split(".xyz")[0]+"\n"
    t+="#SBATCH --output="+sys.argv[1].split(".xyz")[0]+".qlog\n"
    t+="#SBATCH --error="+sys.argv[1].split(".xyz")[0]+".err\n"
    t+="./ab_sequential_wat.py "+sys.argv[1]+" 14"
    #t+="./water_effect.py 14 "+sys.argv[1]+" "+n_waters
    with open("./"+sys.argv[1].split(".xyz")[0]+".qsub","w") as f: f.write(t)
    os.system("sbatch "+sys.argv[1].split(".xyz")[0]+".qsub")
