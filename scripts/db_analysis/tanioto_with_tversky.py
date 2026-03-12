#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
import numpy as np
sys.path.append('../import')
import pandas as pd
from rdkit import Chem
import rdkit
print (rdkit.__version__)
from rdkit.Chem import Fragments
from rdkit.Chem import inchi
from rdkit.Chem import Descriptors 
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs


from routes import extracted_data_route
from routes import sampl_extracted_data_route
from routes import labels_csv_file_name
from routes import sampl_labels_csv_file_name

from db_analysis_functions import find_functional_groups


files=["Gpka_database_with_descriptors.csv","CHEMBL_with_pka_database_with_descriptors.csv","PubChem_with_pka_database_with_descriptors.csv","SAMPL_database_with_descriptors.csv"]
#reference_files=["Gpka_database_with_descriptors.csv","CHEMBL_with_pka_database_with_descriptors.csv","PubChem_with_pka_database_with_descriptors.csv","Gpka_database_with_descriptors.csv"]

Gpka_fingerprints={}
CHEMBL_fingerprints={}
PubChem_fingerprints={}
SAMPL_fingerprints={}
custom_fingerprints={}

fingerprints=[Gpka_fingerprints,CHEMBL_fingerprints,PubChem_fingerprints,SAMPL_fingerprints]
#reference_fingerprints=[Gpka_fingerprints,CHEMBL_fingerprints,PubChem_fingerprints,SAMPL_fingerprints] #to reference each db internally
reference_fingerprints=[Gpka_fingerprints,Gpka_fingerprints,Gpka_fingerprints,Gpka_fingerprints] #to reference all to Gpka

tanimoto_results_Gpka,tanimoto_results_CHEMBL,tanimoto_results_PubChem,tanimoto_results_SAMPL={},{},{},{}
tanimoto_lists=[tanimoto_results_Gpka,tanimoto_results_CHEMBL,tanimoto_results_PubChem,tanimoto_results_SAMPL]

#files=["Gpka_database_with_descriptors.csv","SAMPL_database_with_descriptors.csv"]
#reference_files=["SAMPL_database_with_descriptors.csv"]
#fingerprints=[SAMPL_fingerprints,Gpka_fingerprints]
#reference_fingerprints=[Gpka_fingerprints]
#tanimoto_lists=[tanimoto_results_SAMPL]


mfpgen2 = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mfpgen3 = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
mfpgen4 = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=2048)
print ("generating fingerprints")
for file_name,fingerprint in zip(files,fingerprints):
    db=pd.read_csv(file_name,encoding='unicode_escape')
    db.set_index("name",inplace=True)
    db.dropna(how='all', axis=1, inplace=True)
    for compn in db.index:
        try: mol=Chem.MolFromInchi(db["inchi"][compn])
        except: continue
        funct_groups=find_functional_groups(mol)
        funct_groups_fgp=[funct_groups[k] for k in funct_groups.keys()]
        custom_uintvect = rdkit.DataStructs.cDataStructs.UIntSparseIntVect(len(funct_groups_fgp))
        for i in range(0,len(funct_groups_fgp)): custom_uintvect[i]=funct_groups_fgp[i]
        fingerprint[compn]=[mfpgen2.GetCountFingerprint(mol),mfpgen3.GetCountFingerprint(mol),mfpgen4.GetCountFingerprint(mol),custom_uintvect]


#reference_fingerprints=[Gpka_fingerprints] #borrar
#fingerprints=[SAMPL_fingerprints]
#tanimoto_lists=[tanimoto_results_SAMPL]  #borrarm
#files=["SAMPL_database_with_descriptors.csv"]

for fingerprint,reference_fingerprint,tanimoto_list,df_file in zip(fingerprints,reference_fingerprints,tanimoto_lists,files):
    counter=0
    name=pd.Series(dtype="str")
    t_s_average_custom,t_s_average_2,t_s_average_3,t_s_average_4=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
    t_s_std_custom,t_s_std_2,t_s_std_3,t_s_std_4=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
    t_s_max_custom,t_s_max_2,t_s_max_3,t_s_max_4=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
    t_s_median_custom,t_s_median_2,t_s_median_3,t_s_median_4=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
    t_s_perc08_custom,t_s_perc08_2,t_s_perc08_3,t_s_perc08_4=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
    t_s_perc09_custom,t_s_perc09_2,t_s_perc09_3,t_s_perc09_4=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
    t_s_perc095_custom,t_s_perc095_2,t_s_perc095_3,t_s_perc095_4=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

    #new_file_name=df_file.split(".csv")[0]+"_tanimoto_with.csv"
    new_file_name=df_file.split(".csv")[0]+"_tanimoto_with_Gpka.csv"

    for compn in fingerprint.keys():
        rest_fp2=[reference_fingerprint[k][0] for k in reference_fingerprint.keys() if k!=compn]
        rest_fp3=[reference_fingerprint[k][1] for k in reference_fingerprint.keys() if k!=compn]
        rest_fp4=[reference_fingerprint[k][2] for k in reference_fingerprint.keys() if k!=compn]
        rest_custom=[reference_fingerprint[k][3] for k in reference_fingerprint.keys() if k!=compn]
        tanimoto_2=DataStructs.BulkTanimotoSimilarity(fingerprint[compn][0],rest_fp2)
        tanimoto_3=DataStructs.BulkTanimotoSimilarity(fingerprint[compn][1],rest_fp3)
        tanimoto_4=DataStructs.BulkTanimotoSimilarity(fingerprint[compn][2],rest_fp4)
        tanimoto_custom=DataStructs.BulkTanimotoSimilarity(fingerprint[compn][3],rest_custom)
        #tanimoto_list[compn]=[tanimoto_2,tanimoto_3,tanimoto_4,tanimoto_custom]
        name[compn]=compn
        t_s_average_custom[compn],t_s_std_custom[compn],t_s_max_custom[compn],t_s_median_custom[compn]=np.average(tanimoto_custom),np.std(tanimoto_custom),np.max(tanimoto_custom),np.median(tanimoto_custom)
        t_s_perc08_custom[compn],t_s_perc09_custom[compn],t_s_perc095_custom[compn]=np.percentile(tanimoto_custom,80),np.percentile(tanimoto_custom,90),np.percentile(tanimoto_custom,95)
        t_s_average_2[compn],t_s_std_2[compn],t_s_max_2[compn],t_s_median_2[compn]=np.average(tanimoto_2),np.std(tanimoto_2),np.max(tanimoto_2),np.median(tanimoto_2)
        t_s_perc08_2[compn],t_s_perc09_2[compn],t_s_perc095_2[compn]=np.percentile(tanimoto_2,80),np.percentile(tanimoto_2,90),np.percentile(tanimoto_2,95)
        t_s_average_3[compn],t_s_std_3[compn],t_s_max_3[compn],t_s_median_3[compn]=np.average(tanimoto_3),np.std(tanimoto_3),np.max(tanimoto_3),np.median(tanimoto_3)
        t_s_perc08_3[compn],t_s_perc09_3[compn],t_s_perc095_3[compn]=np.percentile(tanimoto_3,80),np.percentile(tanimoto_3,90),np.percentile(tanimoto_3,95)
        t_s_average_4[compn],t_s_std_4[compn],t_s_max_4[compn],t_s_median_4[compn]=np.average(tanimoto_4),np.std(tanimoto_4),np.max(tanimoto_4),np.median(tanimoto_4)
        t_s_perc08_4[compn],t_s_perc09_4[compn],t_s_perc095_4[compn]=np.percentile(tanimoto_4,80),np.percentile(tanimoto_4,90),np.percentile(tanimoto_4,95)

        counter+=1
        if counter%10==0:
            Gpka_database_with_descriptors=pd.DataFrame()
            print (str(counter)+ "entries out of " +str(len(fingerprint.keys()))+ " in this database",end="\r")
            Gpka_database_with_descriptors["name"]=name

            Gpka_database_with_descriptors["Morgan radius 2 tanimoto similarity average"]=t_s_average_2
            Gpka_database_with_descriptors["Morgan radius 2 tanimoto similarity std"]=t_s_std_2
            Gpka_database_with_descriptors["Morgan radius 2 tanimoto similarity max"]=t_s_max_2
            Gpka_database_with_descriptors["Morgan radius 2 tanimoto similarity median"]=t_s_median_2
            Gpka_database_with_descriptors["Morgan radius 2 tanimoto similarity percentile 80"]=t_s_perc08_2
            Gpka_database_with_descriptors["Morgan radius 2 tanimoto similarity percentile 90"]=t_s_perc09_2
            Gpka_database_with_descriptors["Morgan radius 2 tanimoto similarity percentile 95"]=t_s_perc095_2

            Gpka_database_with_descriptors["Morgan radius 3 tanimoto similarity average"]=t_s_average_3
            Gpka_database_with_descriptors["Morgan radius 3 tanimoto similarity std"]=t_s_std_3
            Gpka_database_with_descriptors["Morgan radius 3 tanimoto similarity max"]=t_s_max_3
            Gpka_database_with_descriptors["Morgan radius 3 tanimoto similarity median"]=t_s_median_3
            Gpka_database_with_descriptors["Morgan radius 3 tanimoto similarity percentile 80"]=t_s_perc08_3
            Gpka_database_with_descriptors["Morgan radius 3 tanimoto similarity percentile 90"]=t_s_perc09_3
            Gpka_database_with_descriptors["Morgan radius 3 tanimoto similarity percentile 95"]=t_s_perc095_3

            Gpka_database_with_descriptors["Morgan radius 4 tanimoto similarity average"]=t_s_average_4
            Gpka_database_with_descriptors["Morgan radius 4 tanimoto similarity std"]=t_s_std_4
            Gpka_database_with_descriptors["Morgan radius 4 tanimoto similarity max"]=t_s_max_4
            Gpka_database_with_descriptors["Morgan radius 4 tanimoto similarity median"]=t_s_median_4
            Gpka_database_with_descriptors["Morgan radius 4 tanimoto similarity percentile 80"]=t_s_perc08_4
            Gpka_database_with_descriptors["Morgan radius 4 tanimoto similarity percentile 90"]=t_s_perc09_4
            Gpka_database_with_descriptors["Morgan radius 4 tanimoto similarity percentile 95"]=t_s_perc095_4

            Gpka_database_with_descriptors["custom fingerprint tanimoto similarity average"]=t_s_average_custom
            Gpka_database_with_descriptors["custom fingerprint tanimoto similarity std"]=t_s_std_custom
            Gpka_database_with_descriptors["custom fingerprint tanimoto similarity max"]=t_s_max_custom
            Gpka_database_with_descriptors["custom fingerprint tanimoto similarity median"]=t_s_median_custom
            Gpka_database_with_descriptors["custom fingerprint tanimoto similarity percentile 80"]=t_s_perc08_4
            Gpka_database_with_descriptors["custom fingerprint tanimoto similarity percentile 90"]=t_s_perc09_4
            Gpka_database_with_descriptors["custom fingerprint tanimoto similarity percentile 95"]=t_s_perc095_4

            Gpka_database_with_descriptors.to_csv(new_file_name)




