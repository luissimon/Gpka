#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#script to find duplicates in the labels file
#it compares inchi of entries with different compound names

import string
import os
import os.path
import numpy as np
import pandas as pd
import ast


log=""
#the place where all the files live
labels_route= "/Users/luissimon/Documents/proyecto2017/pka/"

#the csv file containing the experimental pka values
labels=pd.read_csv(labels_route+"labels-all-inchy9.csv",encoding='unicode_escape')
labels.set_index("compn",inplace=True)
#labels.drop(['alternative pka','alternative ref.','is Lange?'],axis=1,inplace=True)
labels.dropna(how='all', axis=1, inplace=True)

names=labels.index
print (names)
print (names[0])
print (names[2])
print (len(names))
count=0
for i in range(0,len(names)): 
    name=names[i]
    print ("checking: "+str(name))
    repeated_names=[]
    repeated_ids=[]
    for j in range(i,len(names)):
        name2=names[j]
        inchi_match=False
        inchi_key_match=False
        smarts_match=False
        smiles_match=False

        if name.split("_")[0]!=name2.split("_")[0]:
            #print (labels["inchi"][name])
            #print (name2+"-"+labels["inchi"][name2])
            for inchi in ast.literal_eval(labels["inchi"][name]):
                if inchi in ast.literal_eval(labels["inchi"][name2]): inchi_match=True
            for inchi_key in ast.literal_eval(labels["inchiKey"][name]):
                if inchi_key in ast.literal_eval(labels["inchiKey"][name2]): inchi_key_match=True
            #for smiles in ast.literal_eval(labels["smiles"][name]):
            #    if smiles in ast.literal_eval(labels["smiles"][name2]): smiles_match=True
            #for smarts in ast.literal_eval(labels["smarts"][name]):
            #    if smarts in ast.literal_eval(labels["smarts"][name2]): smarts_match=True
        
        if inchi_match or inchi_key_match or smiles_match or smarts_match:
            if name2 not in repeated_names:
                repeated_names.append(name2)
                repeated_ids.append([name2,inchi_match,inchi_key_match,smiles_match, smarts_match])

            
    if len(repeated_names)!=0:
        count+=1
        s= "\n"+name+" is possibly repeated in:\n"
        for n,i in zip(repeated_names,repeated_ids):
            s+="     "+n+"              matches:   inchi: "+str(i[0])+"  inchiKey: "+str(i[1])+"   smiles: "+str(i[2])+"   smarts: "+str(i[3])+"\n"
        print (s)
        print ("found: "+str(count)+" structures repeated so far...")
        log+=s

with open ("repeated.txt","w") as f: f.write(log)
