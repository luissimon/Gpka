#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-



import spektral
import json
import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
import gpka_spektral_dataset
from drop_compounds import drop_compounds


dataset=gpka_spektral_dataset.gpka_spektral_dataset("molecular_graphs-gibbs-M06.25.json",label_key="pKa",equilibrium_keys="")


counter=0
all_counter=0
for graph in dataset.graphs:
    print (graph.y)
    if graph.y not in drop_compounds:
        all_counter+=1
        non_zeros_in_mask=[i!=0 for i in graph.mask].count(True)
        if non_zeros_in_mask>1:
           print (str(graph.y)+ "  " +str(graph.mask)+"   "+str(non_zeros_in_mask) )
           counter+=1

print (all_counter)
print (counter)
