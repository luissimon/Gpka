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


#definition of panda series
name=pd.Series(dtype="str")
inchi_key,smiles,inchi= pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str")
mw,alogp,hba,hbd,psa=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="int"),pd.Series(dtype="int"),pd.Series(dtype="float")
n_rot_bonds=pd.Series(dtype="int")
Murcko_Scaffold=pd.Series(dtype="int")
functional_groups=pd.Series(dtype="str")




#the csv file containing the experimental pka values
#labels_route= extracted_data_route
labels_route= sampl_extracted_data_route
#labels=pd.read_csv(labels_route+labels_csv_file_name,encoding='unicode_escape')
labels=pd.read_csv(labels_route+sampl_labels_csv_file_name,encoding='unicode_escape')
labels.set_index("compn",inplace=True)
labels.dropna(how='all', axis=1, inplace=True)
#output_file="Gpka_database_with_descriptors.csv"
output_file="SAMPL_database_with_descriptors.csv"


def process_entry(s):

    if s.startswith("["):
        ss=s.replace("\\","\\\\")
        return eval(ss)
    else: 
        return [s.strip()]

counter=0
for compn in labels.index:
    counter+=1
    n=compn.split("_")[0]
    if n not in name:
        possible_inchis=process_entry(str(labels["inchi"][compn]))
        print(possible_inchis)
        for m in possible_inchis:
            try:
                mol = Chem.MolFromInchi(m)
                inchi[n]=process_entry(str(labels["inchi"][compn]))
                break
            except: continue
        if mol != None:
            name[n]=n
            smiles[n]=Chem.MolToSmiles(mol,canonical=True).replace("\\","\\\\")
            inchi_key[n]=Chem.MolToInchiKey(mol)
            inchi[n]=Chem.MolToInchi(mol)
            mw[n]=Descriptors.ExactMolWt(mol)
            funct_groups=find_functional_groups(mol)
            alogp[n]=Descriptors.TPSA(mol)
            hba[n],hbd[n]=funct_groups["num_hba"],funct_groups["num_hbd"]
            psa[n]=Descriptors.MolLogP(mol)
            n_rot_bonds[n]=rdMolDescriptors.CalcNumRotatableBonds(mol)
            Chem.RemoveStereochemistry(mol)
            Murcko_similes=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),canonical=True)
            Murcko_Scaffold[n]=Murcko_similes
            functional_groups[n]=repr(funct_groups)





    if counter%100==0 or counter==len(labels.index): 
            Gpka_database_with_descriptors=pd.DataFrame()
            
            Gpka_database_with_descriptors["name"] = name
            Gpka_database_with_descriptors.set_index("name",inplace=True)
            Gpka_database_with_descriptors["inchi_key"] = inchi_key
            Gpka_database_with_descriptors["smiles"] = smiles
            Gpka_database_with_descriptors["inchi"] = inchi
            Gpka_database_with_descriptors["MW"] = mw
            Gpka_database_with_descriptors["ALOGP"] = alogp
            Gpka_database_with_descriptors["PSA"] = psa
            Gpka_database_with_descriptors["HBA"] = hba
            Gpka_database_with_descriptors["HBD"] = hbd
            Gpka_database_with_descriptors["n_rot_bonds"] = n_rot_bonds
            Gpka_database_with_descriptors["Murcko_Scaffold"] = Murcko_Scaffold
            Gpka_database_with_descriptors["functional_groups"] = functional_groups
            #print(Gpka_database_with_descriptors)
            #print(Gpka_database_with_descriptors.info())
            #print ("writing to file: "+output_file)                                   
            #Gpka_database_with_descriptors.to_csv(output_file)
            


