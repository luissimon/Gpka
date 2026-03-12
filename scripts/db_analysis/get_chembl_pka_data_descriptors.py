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


from chembl_webresource_client.new_client import new_client


assay=new_client.assay
res = assay.filter(description__icontains='pka', assay_type='P')


#definition of panda series
chembl_id_name=pd.Series(dtype="str")
inchi_key,smiles,inchi= pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str")
mw,alogp,hba,hbd,psa=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="int"),pd.Series(dtype="int"),pd.Series(dtype="float")
n_rot_bonds=pd.Series(dtype="int")
Murcko_Scaffold=pd.Series(dtype="int")
functional_groups=pd.Series(dtype="str")



counter=0
for r in res[1:]:
    counter+=1
    print ("evaluating assay: "+str(counter) + " of " + str(len(res)),end='\r')
    if 'assay_chembl_id' in r.keys():
        #print (r['assay_chembl_id'])
        activity = new_client.activity
        activities = activity.filter(assay_chembl_id=r['assay_chembl_id']).only('molecule_chembl_id','canonical_smiles','standard_value')

        for a in activities: 
            chembl_id=a['molecule_chembl_id']

            if chembl_id not in inchi_key:        
                chembl_id_name[chembl_id]=chembl_id
                m=new_client.molecule.filter(chembl_id=chembl_id)[0]
                if m['molecule_structures'] is not None and m['molecule_properties']['hba'] is not None:
                    inchi_key[chembl_id]=m['molecule_structures']['standard_inchi_key']
                    smiles[chembl_id]=m['molecule_structures']['canonical_smiles'].replace("\\","\\\\")
                    inchi[chembl_id]=m['molecule_structures']['standard_inchi']
                    mw[chembl_id]=m['molecule_properties']['full_mwt']
                    #alogp[chembl_id]=m['molecule_properties']['alogp']
                    #psa[chembl_id]=m['molecule_properties']['psa']
                    hba[chembl_id]=m['molecule_properties']['hba']
                    hbd[chembl_id]=m['molecule_properties']['hbd']
                    mol = Chem.MolFromInchi(m['molecule_structures']['standard_inchi'])
                    
                    n_rot_bonds[chembl_id]=rdMolDescriptors.CalcNumRotatableBonds(mol)
                    psa[chembl_id]=Descriptors.MolLogP(mol)
                    alogp[chembl_id]=Descriptors.TPSA(mol)
                    Chem.RemoveStereochemistry(mol)
                    #print ("...")
                    #Murcko_Scaffold[chembl_id]=rdkit.Chem.rdinchi.MolToInchiKey(MurckoScaffold.GetScaffoldForMol(mol))# Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),canonical=True)
                    Murcko_similes=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),canonical=True)
                    #print (chembl_id)
                    #print(Murcko_similes)
                    #print ("...")
                    Murcko_Scaffold[chembl_id]=Murcko_similes
                    functional_groups[chembl_id]=repr(find_functional_groups(mol))


    if counter%100==0 or counter==len(res): 
            chembl_pka_database=pd.DataFrame()
            print (len(inchi_key))
            
            chembl_pka_database["name"] = chembl_id_name
            chembl_pka_database.set_index("name",inplace=True)
            chembl_pka_database["inchi_key"] = inchi_key
            chembl_pka_database["smiles"] = smiles
            chembl_pka_database["inchi"] = inchi
            chembl_pka_database["MW"] = mw
            chembl_pka_database["ALOGP"] = alogp
            chembl_pka_database["PSA"] = psa
            chembl_pka_database["HBA"] = hba
            chembl_pka_database["HBD"] = hbd
            chembl_pka_database["n_rot_bonds"] = n_rot_bonds
            chembl_pka_database["Murcko_Scaffold"] = Murcko_Scaffold
            chembl_pka_database["functional_groups"] = functional_groups
            print(chembl_pka_database)
            print(chembl_pka_database.info())
            print ("writing to file: chembl_pka_database.csv")                                   
            chembl_pka_database.to_csv("CHEMBL_with_pka_database_with_descriptors.csv")
            


