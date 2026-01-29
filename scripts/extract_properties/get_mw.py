#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# script to get molecular weights i

import string
import os
import os.path
import sys
sys.path.append('../import')
import copy
import Molecular_structure
import numpy as np
import pandas as pd
from routes import output_files_route
from routes import extracted_data_route

#the place where all the files live
route_of_files=output_files_route+"PBEh3c_optimized/SP-swb97xd/"
files=[f for f in os.listdir(route_of_files) if f[-8:]=="chrg.out"]
labels=pd.read_csv(extracted_data_route+"values_extracted-gibbs-swb97xd.new.csv",encoding='latin-1')
labels.set_index("compn",inplace=True)
weights_file="mw.csv"
name,correct_name,pka_value,mw=pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="float")
                                                                                                              
old_names=labels.index
for compn in old_names:


    ####################################       D E T E R M I N E    F I L E     S U F F I X E S    F R O M    E N T R Y    I N    D A T A B A S E       ########################################
    if str(compn.split("_")[1]).startswith("cation"):   protonated_str="-cation"
    if str(compn.split("_")[1]).startswith("2cation"):  protonated_str="-2cation"
    if str(compn.split("_")[1]).startswith("3cation"):  protonated_str="-3cation"
    if str(compn.split("_")[1]).startswith("4cation"):  protonated_str="-4cation"
    if str(compn.split("_")[1]).startswith("5cation"):  protonated_str="-5cation"
    if str(compn.split("_")[1]).startswith("6cation"):  protonated_str="-6cation"
    if str(compn.split("_")[1]).startswith("neut"):     protonated_str="-neut"
    if str(compn.split("_")[1]).startswith("an"):       protonated_str="-an"
    if str(compn.split("_")[1]).startswith("2an"):      protonated_str="-2an"
    if str(compn.split("_")[1]).startswith("3an"):      protonated_str="-3an"
    if str(compn.split("_")[1]).startswith("4an"):      protonated_str="-4an"
    if str(compn.split("_")[1]).startswith("5an"):      protonated_str="-5an"

    ############################################################              L I S T S       O F        F I L E S               ################################################################
    protonated_molecules_file_names=[f for f in files if f.startswith( str(compn).split("_")[0]+protonated_str)]
    pm=Molecular_structure.Molecular_structure(route_of_files+"/"+protonated_molecules_file_names[0])
    m=pm.molecular_weight #np.sum([ pm.atomic_weights[a.symbol.lower()] for a in pm.atom_list   ])
    print ("compn: "+str(compn)+" mw:"+str(m),end="\r")
    mw[compn]=m
    



labels["MW"]=mw
labels.to_csv(weights_file)
