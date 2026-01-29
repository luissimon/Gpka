#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
# script that, given two compound names, searches for their structure energies and in the labels csv file 
# useful to decide which files/entries to remove in case of duplicates. 

import string
import os
import os.path
import sys
sys.path.append('../import')
import pandas as pd
import numpy as np
import annotated_atoms

from routes import output_files_route
from routes import extracted_data_route
from routes import labels_name

route=output_files_route+"PBEh3c_optimized/SP-swb97xd/" #change this to use another energy
labels_route= extracted_data_route

#the csv file containing the experimental pka values
labels=pd.read_csv(labels_route+labels_name,encoding='unicode_escape')

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]!="": 
        file_name_one=sys.argv[1]
        if len(sys.argv)>2 and sys.argv[2]!="": 
            file_name_two=sys.argv[2]
        else: print("gimme names!"); sys.exit()
    
    else: print("gimme names!"); sys.exit()

    print ("Structures and energies for: " +file_name_one)
    os.system( "grep -i 'final single' "+route+file_name_one+"*.out")
    print ("Structures and energies for: " +file_name_two)
    os.system( "grep -i 'final single' "+route+file_name_two+"*.out")

    comps=labels["compn"].to_list()
    pkas=labels["pKa"].to_list()
    refs=labels["reference"].to_list()
    repeated_molecules=annotated_atoms.repeated_molecules.keys()
    indexes_one,indexes_two=[],[]
    for i,c in enumerate(comps):
        if c.startswith(file_name_one+"_"): indexes_one.append(i)
        if c.startswith(file_name_two+"_"): indexes_two.append(i)
    print ("entries in excel file:")
    print ("for "+file_name_one+":")
    for i in indexes_one: print (str(i+2)+ "..."+str(pkas[i])+"..."+str(refs[i])+"\n")
    print ("for "+file_name_two+":")
    for i in indexes_two: print (str(i+2)+ "..."+str(pkas[i])+"..."+str(refs[i])+"\n")

    for ra in repeated_molecules:
        if ra.startswith(file_name_one+"_"): print (file_name_one+" in 'repeated_molecules")
        if ra.startswith(file_name_two+"_"): print (file_name_two+" in 'repeated_molecules")




