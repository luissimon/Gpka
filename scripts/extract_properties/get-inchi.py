#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import string
import os
import os.path
import sys
sys.path.append('../import')
import copy
import Molecular_structure
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.spatial.transform import Rotation 
import pandas as pd
import rdkit
import xyz2mol
from rdkit import Chem

import pubchempy
#import Levenshtein
from routes import extracted_data_route
from routes import output_files_route
from routes import labels_csv_file_name


#the place where all the files live
labels_route= extrated_data_route 
routes=[output_files_route+"PBEh3c_optimized/SP-m06/"]

#the csv file containing the experimental pka values
labels=pd.read_csv(extracted_data_route+labels_csv_file_name,encoding='unicode_escape')
labels.set_index("compn",inplace=True)
labels.dropna(how='all', axis=1, inplace=True)
names_file=labels_csv_file_name.split(".")[0]+"inchi.csv" 




import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


#borrowed from xyz2mol.py:
def unpack_xyz_string(xyz_string):
    
    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""
    __ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']   

    for line_number, line in enumerate(xyz_string.split("\n")):
        if line.strip()!="":
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                title = line.split("/")[-1].split("_m06_chrg.out")[0]
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [(__ATOM_LIST__.index(atom.lower()) + 1) for atom in atomic_symbols]

    return atoms, xyz_coordinates,title


def get_pubchem_parent(cid, orphans_as_self=True):
    """
    From a pubchem_cid, retreive the parent compound's cid.
    If function is unsuccesful in retrieving a single parent,
    `orphans_as_self = True` returns `cid` rather than None.
    
    According to pubmed:
    
    > A parent is conceptually the "important" part of the molecule
    > when the molecule has more than one covalent component.
    > Specifically, a parent component must have at least one carbon
    > and contain at least 70% of the heavy (non-hydrogen) atoms of
    > all the unique covalent units (ignoring stoichiometry).
    > Note that this is a very empirical definition and is subject to change.

    A parallel query can be executed using the REST PUG API:
    http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/11477084/cids/XML?cids_type=parent
    """
    assert cid
    
    try:
        parent_cids = pubchempy.get_cids(identifier=cid, namespace='cid', domain='compound', cids_type='parent')
    except pubchempy.BadRequestError as e:
        print ('Error getting parent of {}. {}'.format(cid, e))
        return cid if orphans_as_self else None
    try:
        parent_cid, = parent_cids
        return parent_cid
    except ValueError:
        print ('Error getting parent of {}. Parents retreived: {}'.format(cid, parent_cids))
    return cid if orphans_as_self else None

already_named={}

name,correct_name,pka_value,reference=pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str")
smiles_for_each_comp,smarts_for_each_comp,inchikey_for_each_comp,inchi_for_each_comp=    pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str"),pd.Series(dtype="str")

counter=0

#old_names=labels.index[6395:6396]
#old_names=labels.index[1241:1243]
old_names=labels.index
for compn in old_names:
    counter+=1
    print ("compn: "+str(compn))
    list_of_smiles,list_of_inchi,list_of_smarts,list_of_inchi_keys=[],[],[],[]
    
    if compn.split("_")[0].lower() not in already_named.keys() or already_named[compn.split("_")[0].lower()]=="not found":
        molecule_file_names=[]
        old_name=compn.split("_")[0]
        charge=0
        for route in routes:  #ff.split("-")[0]==old_name 
            for f in [ff for ff in os.listdir(route) if  "-".join(ff.split("-")[:-1])==old_name   and "neut" in ff and ff.endswith("_m06_chrg.out") ]:
                molecule_file_names.append(route+f)

        #if there is not a neutral compound...
        if len(molecule_file_names)==0:
            charge=1
            for route in routes:
                for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-cation" in ff and ff.endswith("_m06_chrg.out") )]:
                    molecule_file_names.append(route+f)
            if len(molecule_file_names)==0:
                charge=-1
                for route in routes:
                    for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-an" in ff and ff.endswith("_m06_chrg.out") )]:
                        molecule_file_names.append(route+f)
                if len(molecule_file_names)==0:
                    charge=2                          
                    for route in routes:
                        for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-2cation" in ff and ff.endswith("_m06_chrg.out") )]:
                            molecule_file_names.append(route+f)
                    if len(molecule_file_names)==0:
                        charge=-2
                        for route in routes:
                            for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-2an" in ff and ff.endswith("_m06_chrg.out") )]:
                                molecule_file_names.append(route+f)
                        if len(molecule_file_names)==0:
                            charge=-3
                            for route in routes:
                                for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-3an" in ff and ff.endswith("_m06_chrg.out") )]:
                                    molecule_file_names.append(route+f)
                            if len(molecule_file_names)==0:
                                charge=-4
                                for route in routes:
                                    for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-4an" in ff and ff.endswith("_m06_chrg.out") )]:
                                        molecule_file_names.append(route+f)
                                if len(molecule_file_names)==0:
                                    charge=-5
                                    for route in routes:
                                        for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-5an" in ff and ff.endswith("_m06_chrg.out") )]:
                                            molecule_file_names.append(route+f)
                                    if len(molecule_file_names)==0:
                                        charge=3
                                        for route in routes:
                                            for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-3cation" in ff and ff.endswith("_m06_chrg.out") )]:
                                                molecule_file_names.append(route+f)
                                        if len(molecule_file_names)==0:
                                            charge=4
                                            for route in routes:
                                                for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "-4cation" in ff and ff.endswith("_m06_chrg.out") )]:
                                                    molecule_file_names.append(route+f)
                                            if len(molecule_file_names)==0:
                                                charge=5
                                                for route in routes:
                                                    for f in [ff for ff in os.listdir(route) if (ff.split("-")[0]==old_name and "5cation" in ff and ff.endswith("_m06_chrg.out") )]:
                                                        molecule_file_names.append(route+f)

        if len(molecule_file_names)==0: 
            print ("unable to find files for "+str(compn))
            already_named[compn.split("_")[0]]="not found"
            continue
        #elif len(molecule_file_names)>8: molecule_file_names=molecule_file_names[0:4]
        #print (molecule_file_names)
        molecules=[Molecular_structure.Molecular_structure(f,"last") for f in molecule_file_names]
        energies=[m_s.gibbs_free_energy for m_s in molecules]
        molecules=[x for _,x in sorted(zip(energies,molecules))] #sort structures according to their energies.
        molecules_xyzs=[m.print_xyz() for m in molecules]
        neutral_mols=[]
        for m in molecules_xyzs:
            atoms,coords,title= unpack_xyz_string(m)
            try:
                mols= xyz2mol.xyz2mol(atoms, coords, charge,use_huckel=False,covalent_factor=1.3)
            except:
                try:
                    mols= xyz2mol.xyz2mol(atoms, coords, charge,use_huckel=False,covalent_factor=1.2)
                    print ("using rcov=1.2")
                except:
                    mols= xyz2mol.xyz2mol(atoms, coords, charge,use_huckel=False,covalent_factor=1.1)
                    print ("using rcov=1.1")
            for mm in mols: neutral_mols.append(mm)
        
        synonyms=[]

        for m in neutral_mols:
            smiles=Chem.MolToSmiles(m,isomericSmiles=True)
            if smiles not in list_of_smiles: list_of_smiles.append(smiles)
            smarts=Chem.MolToSmarts(m,isomericSmiles=True)
            if smarts not in list_of_smarts: list_of_smarts.append(smarts)
            inchi=Chem.MolToInchi(m)
            #inchi2,auxinfo=Chem.MolToInchiAndAuxInfo(m)
            #print (auxinfo)
            if inchi not in list_of_inchi: list_of_inchi.append(inchi)
            inchikey=Chem.MolToInchiKey(m)
            if inchikey not in list_of_inchi_keys: list_of_inchi_keys.append(inchikey)

            
            """
            try:
                compounds = pubchempy.get_compounds(inchi, namespace='inchi')
            except pubchempy.PubChemPyError:
                try: 
                    compounds = pubchempy.get_compounds(smiles, namespace='smiles')
                except pubchempy.PubChemPyError: continue

            
            #print (smiles)
            if len(compounds)>0:
                for mm in compounds:
                    if mm.iupac_name!=None: synonyms.append(mm.iupac_name.lower())
                    if mm.synonyms!=None and len(mm.synonyms)>0:
                        for s in mm.synonyms: synonyms.append(s.lower())
                    cid=mm.cid
                    #print (synonyms)
                    #print (cid)
                    
                    if cid!=None and len(synonyms)==0:
                        parent= get_pubchem_parent(cid) 
                        parent_comp=pubchempy.Compound.from_cid(parent)
                        synonyms.append(parent_comp.iupac_name)
                        if len(parent_comp.synonyms)>0:
                            for s in parent_comp.synonyms: synonyms.append(s.lower())
                        #print(synonyms)
            """

        #name[compn]=compn
        smiles_for_each_comp[compn]=list_of_smiles
        inchi_for_each_comp[compn]=list_of_inchi
        inchikey_for_each_comp[compn]=list_of_inchi_keys
        smarts_for_each_comp[compn]=list_of_smarts 
        #pka_value[compn]=labels["pKa"][compn]

        #correct_name[compn]=labels["correct name"][compn]
        #reference[compn]=labels["reference"][compn]

        """          
        if len(synonyms)>0 and synonyms!=[None]:
            lenvenshtein_distances=[Levenshtein.distance(old_name,s,weights=(1,2,2)) for s in synonyms if s!=None]
            sorted_synonyms=[ l for _,l in sorted(zip(lenvenshtein_distances,synonyms))]
            already_named[compn.split("_")[0]]=sorted_synonyms[0]
            print ("the distance between "+compn.split("_")[0]+" and "+sorted_synonyms[0]+" is: "+str(Levenshtein.distance(old_name,sorted_synonyms[0],weights=(1,2,2))))
        else: already_named[compn.split("_")[0].lower()]="not found"
        """




    if counter%1==0 or counter==len(old_names): 
        print (counter)
        #print ("writing to file")
        #new_names["compn"]=name
        labels["smiles"]=smiles_for_each_comp
        labels["smarts"]=smarts_for_each_comp
        labels["inchi"]=inchi_for_each_comp
        labels["inchiKey"]=inchikey_for_each_comp
        #labels["pKa"]=pka_value
        #labels["reference"]=reference
        #labels["correct name"]=new_names
        labels.to_csv(labels_route+names_file)
    




