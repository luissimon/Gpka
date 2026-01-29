#! /usr/bin/env python
# -*- coding: utf-8 -*-
#script to launch calculations using perturbations of the last n atoms taking m. call it:  ./combs.py name_of_compound-neut  n:m 
#it can be used, for example, to generate tautomers: drawing n H atoms on each possible heteroatom, of which only m will be keeped.
#each time that it is called the last coordinates are shown (they can be modified), so it is possible to change m so geometries with different
#number of H atoms are generated.   

import sys
import os
import string
import getpass
import subprocess
import itertools

if len(sys.argv) > 2:
    filename= sys.argv[1]

    var_atoms,n_h=sys.argv[2].split(":")
    var_atoms=int(var_atoms)
    n_h=int(n_h)

    os.system("vi borrame.xyz")
    with open("borrame.xyz","r") as f: lines=[l for l in f.readlines() if l.strip()!=""]
    fixed_lines=lines[0:-var_atoms]
    var_lines=lines[-var_atoms:]
    #for f in fixed_lines: print (f)
    #print()
    #for v in var_lines: print (v)
    #sys.exit()
    combs=[list(com)  for com in itertools.combinations(var_lines,n_h)]
    i=0
    for v in combs:
        """
        print (v)
        print (len(v))
        print ()
        """
        i+=1
        all_lines=fixed_lines+v #[var_lines[int(c)] for c in combs]
        t=""
        for l in all_lines: 
            t+=l
        #print("   ")
        #print(t)
        with open(filename+str(i)+".xyz","w") as f: f.write(t)
        os.system("./genxyz.py "+filename+str(i))+ " -auto"
        

    


