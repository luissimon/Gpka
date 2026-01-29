#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
#script para comprobar si falta algún archivo 

import string
import os
import os.path
import sys
sys.path.append('../import')

if __name__ == "__main__":
    if len (sys.argv)>1 and sys.argv[1]!="": file_names=[sys.argv[1]]
    else: 
        #file_names=[f.split("_fake.out")[0] for f in os.listdir() if f.endswith("fake.out")]
        file_names=[f.split(".hess")[0] for f in os.listdir() if f.endswith(".hess")]
        water_file_names=[f.split("_1wat.conformers.gfn2_gfnff_gbsa.xyz") for f in os.listdir() if f.endswith("_1wat.conformers.gfn2_gfnff_gbsa.xyz")]
	
        #file_names=[f.split("_sm06_chrg.cpcm")[0] for f in os.listdir() if f.endswith("_sm06_chrg.cpcm")]
    for f in file_names:
        if f not in water_file_names:
             s="./launch_water_effect2.py  "+f+".xyz"
             print (s)
             os.system(s)
