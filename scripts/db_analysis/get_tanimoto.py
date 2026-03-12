#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
sys.path.append('../import')
import pandas as pd
import numpy as np



def tanemoto_similarity_functional_groups(funct_groups1,funct_groups2):   

    #if funct_groups1.startswith("0/r"): funct_groups1=funct_groups1.replace("0/r","")
    funct_groups1=eval(funct_groups1)

    #if funct_groups2.startswith("0/r"): funct_groups2=funct_groups2.replace("0/r","")
    funct_groups2=eval(funct_groups2)

    keys=list(funct_groups1.keys()&funct_groups2.keys())
    v1=[funct_groups1[k] for k in keys]
    v2=[funct_groups2[k] for k in keys]
    c=[ np.min([vv1,vv2]) for vv1,vv2 in zip(v1,v2)]
    a,b=np.sum(v1),np.sum(v2)
    return c/(a+b-c)


files=["Gpka_database_with_descriptors.csv","CHEMBL_with_pka_database_with_descriptors.csv","PubChem_with_pka_database_with_descriptors.csv","SAMPL_database_with_descriptors.csv"]
reference_files=["Gpka_database_with_descriptors.csv","CHEMBL_with_pka_database_with_descriptors.csv","PubChem_with_pka_database_with_descriptors.csv","Gpka_database_with_descriptors.csv"]

files=["SAMPL_database_with_descriptors.csv"]
reference_files=["Gpka_database_with_descriptors.csv"]

for file_name in files:
    new_file_name=file_name.split(".csv")[0]+"_tanimoto_similarity.csv"
    descriptors_db=pd.read_csv(file_name,encoding='unicode_escape')
    descriptors_db.set_index("name",inplace=True)
    descriptors_db.dropna(how='all', axis=1, inplace=True)
    reference_descriptors_db=pd.read_csv(reference_files[files.index(file_name)],encoding='unicode_escape')
    reference_descriptors_db.set_index("name",inplace=True)
    reference_descriptors_db.dropna(how='all', axis=1, inplace=True)

    #new series:
    tanimoto_similarity_average,tanimoto_similarity_std,tanimoto_similarity_min,tanimoto_similarity_max=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

    counter=0
    for compn in descriptors_db.index:
        counter+=1
        #print ("evaluating tanimoto similarity, "+str(counter) + " of " + str(len(descriptors_db.index)),end='\r')
        if isinstance(descriptors_db["functional_groups"][compn],str) and descriptors_db["functional_groups"][compn]!="":
            tanimoto_similarities=[ tanemoto_similarity_functional_groups(descriptors_db["functional_groups"][compn],reference_descriptors_db["functional_groups"][other_compn]) 
                                for other_compn in reference_descriptors_db.index if other_compn!=compn and isinstance(reference_descriptors_db["functional_groups"][other_compn],str) ]

            tanimoto_similarity_average[compn]=np.mean(tanimoto_similarities)
            tanimoto_similarity_std[compn]=np.std(tanimoto_similarities)
            tanimoto_similarity_min[compn]=np.min(tanimoto_similarities)
            tanimoto_similarity_max[compn]=np.max(tanimoto_similarities)

        if counter%2==0 or counter==len(descriptors_db.index): 
            descriptors_db["tanimoto_similarity_average"]=tanimoto_similarity_average
            descriptors_db["tanimoto_similarity_std"]=tanimoto_similarity_std
            descriptors_db["tanimoto_similarity_min"]=tanimoto_similarity_min
            descriptors_db["tanimoto_similarity_max"]=tanimoto_similarity_max
            print(descriptors_db.info())
            print(descriptors_db)
            print ("writing to file: "+new_file_name)                                   
            descriptors_db.to_csv(new_file_name)

        


    
