#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
#script para colocar los archivos cada vez que hago un nuevo cálculo en sus carpetas correspondientes

import string
import os
import sys
sys.path.append('../import')
import Molecular_structure
import numpy as np
text="kcal/mol higher than the most stable"
out_files=[f for f in os.listdir() if f.endswith(".out")]
img_files=[f for f in os.listdir() if f.endswith(".img_freqs")]
for f in out_files:
    if f.split(".out")[0]+".img_freqs" in img_files:
        #print (f)
        m=Molecular_structure.Molecular_structure(f,"last")
        min_freq=np.min(m.QM_output.frequencies[0])
        #print (min_freq)
        if min_freq<-20.0 and min_freq>-45.0:
            print (f+ "   freq:"+str(min_freq))
        if f.split(".out")[0]+".old.out" in out_files:
            mm=Molecular_structure.Molecular_structure(f.split(".out")[0]+".old.out","last")
            min_freq=np.min(mm.QM_output.frequencies[0])
        if min_freq<-20.0 and min_freq>-45.0:
            print (f.split(".out")[0]+".old.out"+ "   freq:"+str(min_freq))
