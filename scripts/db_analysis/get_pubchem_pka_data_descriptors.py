#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
import pandas as pd
from rdkit import Chem
import rdkit
print (rdkit.__version__)
from rdkit.Chem import Fragments
from rdkit.Chem import inchi
from rdkit.Chem import Descriptors 
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from db_analysis_functions import find_functional_groups

#definition of panda series
name=pd.Series(dtype="str")
inchi_key,smiles,inchi= pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str")
mw,alogp,hba,hbd,psa=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="int"),pd.Series(dtype="int"),pd.Series(dtype="float")
n_rot_bonds=pd.Series(dtype="int")
Murcko_Scaffold=pd.Series(dtype="int")
functional_groups=pd.Series(dtype="str")

pubchem_with_pka_database=pd.read_csv("PubChem_with_pka.csv")
names=pubchem_with_pka_database["Name"]
inchis=pubchem_with_pka_database["InChI"]

extracted_pubchem_pka_database=pd.DataFrame()
#chembl_pka_database=pd.read_csv("chembl_pka_database.csv",encoding='unicode_escape')
#chembl_pka_database.set_index("chembl_id",inplace=True)
#chembl_pka_database=[]

counter=0
for n,inc in zip(names,inchis):
    counter+=1
    print ("evaluating compoun d: "+str(counter) + " of " + str(len(inchis)),end='\r')
    name[n]=n
    inchi[n]=inc
    mol = Chem.MolFromInchi(inc)
    mw[n]=Descriptors.ExactMolWt(mol)
    smiles[n]=Chem.MolToSmiles(mol,canonical=True).replace("\\","\\\\")
    inchi_key[n]=Chem.MolToInchiKey(mol)
    funct_groups=find_functional_groups(mol)
    alogp[n]=Descriptors.TPSA(mol)
    hba[n],hbd[n]=funct_groups["num_hba"],funct_groups["num_hbd"]
    psa[n]=Descriptors.MolLogP(mol)
    n_rot_bonds[n]=rdMolDescriptors.CalcNumRotatableBonds(mol)
    Chem.RemoveStereochemistry(mol)
    Murcko_similes=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),canonical=True)
    Murcko_Scaffold[n]=Murcko_similes
    functional_groups[n]=repr(funct_groups)



    if counter%100==0 or counter==len(names): 
            pubchem_with_pka_database_modified=pd.DataFrame()
            print (len(inchi_key))
            
            pubchem_with_pka_database_modified["name"] = name
            pubchem_with_pka_database_modified.set_index("name",inplace=True)
            pubchem_with_pka_database_modified["inchi_key"] = inchi_key
            pubchem_with_pka_database_modified["smiles"] = smiles
            pubchem_with_pka_database_modified["inchi"] = inchi
            pubchem_with_pka_database_modified["MW"] = mw
            pubchem_with_pka_database_modified["ALOGP"] = alogp
            pubchem_with_pka_database_modified["PSA"] = psa
            pubchem_with_pka_database_modified["HBA"] = hba
            pubchem_with_pka_database_modified["HBD"] = hbd
            pubchem_with_pka_database_modified["n_rot_bonds"] = n_rot_bonds
            pubchem_with_pka_database_modified["Murcko_Scaffold"] = Murcko_Scaffold
            pubchem_with_pka_database_modified["functional_groups"] = functional_groups
            print(pubchem_with_pka_database_modified)
            print(pubchem_with_pka_database_modified.info())
            print ("writing to file: pubchem_with_pka_database_modified.csv")                                   
            pubchem_with_pka_database_modified.to_csv("PubChem_with_pka_database_with_descriptors.csv")
            


