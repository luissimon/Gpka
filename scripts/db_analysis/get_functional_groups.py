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


import inspect
from routes import extracted_data_route
from routes import labels_csv_file_name

def find_functional_groups(mol):
    #mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return {"error": "Invalid SMILES string"}

    functional_groups = {}

    functional_groups['num_aliphatic_carboxylic_acid'] = Chem.Fragments.fr_Al_COO(mol) # Aliphatic carboxylic acid
    functional_groups['num_aliphatic_alcohol'] = Chem.Fragments.fr_Al_OH(mol) # Aliphatic alcohol
    #functional_groups['num_aliphatic_alcohol_not_tertiary'] = Chem.Fragments.fr_Al_OH_noTert(mol) # Aliphatic alcohol (not tertiary)
    functional_groups['num_aromatic_nitrogen_heterocycle'] = Chem.Fragments.fr_ArN(mol) # Aromatic nitrogen atom in a ring (e.g., pyridine)
    functional_groups['num_aromatic_carboxylic_acid'] = Chem.Fragments.fr_Ar_COO(mol) # Aromatic carboxylic acid
    functional_groups['num_aromatic_nitrogen'] = Chem.Fragments.fr_Ar_N(mol) # Aromatic nitrogen (any type)
    functional_groups['num_aromatic_primary_secondary_amine'] = Chem.Fragments.fr_Ar_NH(mol) # Primary or secondary aromatic amine
    functional_groups['num_aromatic_hydroxyl'] = Chem.Fragments.fr_Ar_OH(mol) # Aromatic hydroxyl (phenol)
    functional_groups['num_carboxylic_acid_any'] = Chem.Fragments.fr_COO(mol) # Carboxylic acid (any type)
    functional_groups['num_carboxylic_acid_ortho_ring'] = Chem.Fragments.fr_COO2(mol) # Carboxylic acid with ortho-ring substitution
    #functional_groups['num_carbonyl_general'] = Chem.Fragments.fr_C_O(mol) # Carbonyl (C=O) in acids/esters/ketones/aldehydes etc.
    #functional_groups['num_carbonyl_not_carboxylic_acid'] = Chem.Fragments.fr_C_O_noCOO(mol) # Carbonyl (not part of a carboxylic acid)
    #functional_groups['num_thiocarbonyl'] = Chem.Fragments.fr_C_S(mol) # Thiocarbonyl (C=S)
    functional_groups['num_hydroxyamine'] = Chem.Fragments.fr_HOCCN(mol) # Hydroxyamine
    functional_groups['num_imine'] = Chem.Fragments.fr_Imine(mol) # Imine
    functional_groups['num_primary_amine'] = Chem.Fragments.fr_NH0(mol) # Primary amine
    functional_groups['num_secondary_amine'] = Chem.Fragments.fr_NH1(mol) # Secondary amine
    functional_groups['num_tertiary_amine'] = Chem.Fragments.fr_NH2(mol) # Tertiary amine
    functional_groups['num_N_oxide_or_oxime'] = Chem.Fragments.fr_N_O(mol) # N-oxide or Oxime
    #functional_groups['num_ndealkylation_primary'] = Chem.Fragments.fr_Ndealkylation1(mol) # N-dealkylation (primary)
    #functional_groups['num_ndealkylation_secondary'] = Chem.Fragments.fr_Ndealkylation2(mol) # N-dealkylation (secondary)
    functional_groups['num_N_substituted_pyrrole'] = Chem.Fragments.fr_Nhpyrrole(mol) # N-substituted pyrrole
    functional_groups['num_thiol'] = Chem.Fragments.fr_SH(mol) # Thiol
    #functional_groups['num_aldehyde'] = Chem.Fragments.fr_aldehyde(mol) # Aldehyde
    #functional_groups['num_alkyl_carbamate'] = Chem.Fragments.fr_alkyl_carbamate(mol) # Alkyl carbamate
    #functional_groups['num_alkyl_halide'] = Chem.Fragments.fr_alkyl_halide(mol) # Alkyl halide
    #functional_groups['num_allylic_oxidation_site'] = Chem.Fragments.fr_allylic_oxid(mol) # Allylic oxidation site
    functional_groups['num_amide'] = Chem.Fragments.fr_amide(mol) # Amide
    functional_groups['num_amidine'] = Chem.Fragments.fr_amidine(mol) # Amidine
    functional_groups['num_aniline'] = Chem.Fragments.fr_aniline(mol) # Aniline
    #functional_groups['num_aryl_methyl'] = Chem.Fragments.fr_aryl_methyl(mol) # Aryl-methyl
    #functional_groups['num_azide'] = Chem.Fragments.fr_azide(mol) # Azide
    #functional_groups['num_azo'] = Chem.Fragments.fr_azo(mol) # Azo group
    functional_groups['num_barbituric_acid_derivative'] = Chem.Fragments.fr_barbitur(mol) # Barbituric acid derivative
    #functional_groups['num_benzene_ring'] = Chem.Fragments.fr_benzene(mol) # Benzene ring
    functional_groups['num_benzodiazepine'] = Chem.Fragments.fr_benzodiazepine(mol) # Benzodiazepine
    #functional_groups['num_bicyclic_system'] = Chem.Fragments.fr_bicyclic(mol) # Bicyclic system
    #functional_groups['num_diazo'] = Chem.Fragments.fr_diazo(mol) # Diazo group
    functional_groups['num_dihydropyridine'] = Chem.Fragments.fr_dihydropyridine(mol) # Dihydropyridine
    #functional_groups['num_epoxide'] = Chem.Fragments.fr_epoxide(mol) # Epoxide
    #functional_groups['num_ester'] = Chem.Fragments.fr_ester(mol) # Ester
    #functional_groups['num_ether'] = Chem.Fragments.fr_ether(mol) # Ether
    #functional_groups['num_furan'] = Chem.Fragments.fr_furan(mol) # Furan
    functional_groups['num_guanidine'] = Chem.Fragments.fr_guanido(mol) # Guanidine
    #functional_groups['num_halogen'] = Chem.Fragments.fr_halogen(mol) # Halogen
    functional_groups['num_hydrazine'] = Chem.Fragments.fr_hdrzine(mol) # Hydrazine
    functional_groups['num_hydrazone'] = Chem.Fragments.fr_hdrzone(mol) # Hydrazone
    functional_groups['num_imidazole'] = Chem.Fragments.fr_imidazole(mol) # Imidazole
    functional_groups['num_imide'] = Chem.Fragments.fr_imide(mol) # Imide
    #functional_groups['num_isocyanate'] = Chem.Fragments.fr_isocyan(mol) # Isocyanate
    #functional_groups['num_isothiocyanate'] = Chem.Fragments.fr_isothiocyan(mol) # Isothiocyanate
    #functional_groups['num_ketone'] = Chem.Fragments.fr_ketone(mol) # Ketone
    #functional_groups['num_ketone_topliss'] = Chem.Fragments.fr_ketone_Topliss(mol) # Topliss ketone
    functional_groups['num_lactam'] = Chem.Fragments.fr_lactam(mol) # Lactam
    #functional_groups['num_lactone'] = Chem.Fragments.fr_lactone(mol) # Lactone
    #functional_groups['num_methoxy'] = Chem.Fragments.fr_methoxy(mol) # Methoxy group
    functional_groups['num_morpholine'] = Chem.Fragments.fr_morpholine(mol) # Morpholine
    #functional_groups['num_nitrile'] = Chem.Fragments.fr_nitrile(mol) # Nitrile (Cyano group)
    #functional_groups['num_nitro_general'] = Chem.Fragments.fr_nitro(mol) # Nitro group (general)
    #functional_groups['num_nitro_aromatic'] = Chem.Fragments.fr_nitro_arom(mol) # Aromatic nitro group
    #functional_groups['num_nitro_aromatic_non_ortho'] = Chem.Fragments.fr_nitro_arom_nonortho(mol) # Aromatic nitro group (not ortho)
    #functional_groups['num_nitroso'] = Chem.Fragments.fr_nitroso(mol) # Nitroso group
    functional_groups['num_oxazole'] = Chem.Fragments.fr_oxazole(mol) # Oxazole
    functional_groups['num_oxime'] = Chem.Fragments.fr_oxime(mol) # Oxime
    #functional_groups['num_para_hydroxylation'] = Chem.Fragments.fr_para_hydroxylation(mol) # Para-hydroxylation site
    functional_groups['num_phenol'] = Chem.Fragments.fr_phenol(mol) # Phenol
    functional_groups['num_phenol_no_ortho_hbond'] = Chem.Fragments.fr_phenol_noOrthoHbond(mol) # Phenol (no ortho H-bond donor)
    functional_groups['num_phosphoric_acid'] = Chem.Fragments.fr_phos_acid(mol) # Phosphoric acid
    #functional_groups['num_phosphate_ester'] = Chem.Fragments.fr_phos_ester(mol) # Phosphate ester
    functional_groups['num_piperidine'] = Chem.Fragments.fr_piperdine(mol) # Piperidine
    functional_groups['num_piperazine'] = Chem.Fragments.fr_piperzine(mol) # Piperazine
    functional_groups['num_primary_amide'] = Chem.Fragments.fr_priamide(mol) # Primary amide
    functional_groups['num_primary_sulfonamide'] = Chem.Fragments.fr_prisulfonamd(mol) # Primary sulfonamide
    functional_groups['num_pyridine'] = Chem.Fragments.fr_pyridine(mol) # Pyridine
    functional_groups['num_quaternary_nitrogen'] = Chem.Fragments.fr_quatN(mol) # Quaternary nitrogen
    #functional_groups['num_sulfide'] = Chem.Fragments.fr_sulfide(mol) # Sulfide
    functional_groups['num_sulfonamide'] = Chem.Fragments.fr_sulfonamd(mol) # Sulfonamide (any type)
    #functional_groups['num_sulfone'] = Chem.Fragments.fr_sulfone(mol) # Sulfone
    # Note: fr_sulfox was removed as it does not exist in the module
    #functional_groups['num_terminal_acetylene'] = Chem.Fragments.fr_term_acetylene(mol) # Terminal acetylene
    functional_groups['num_tetrazole'] = Chem.Fragments.fr_tetrazole(mol) # Tetrazole
    functional_groups['num_thiazole'] = Chem.Fragments.fr_thiazole(mol) # Thiazole
    #functional_groups['num_thiocyanate'] = Chem.Fragments.fr_thiocyan(mol) # Thiocyanate
    #functional_groups['num_thiophene'] = Chem.Fragments.fr_thiophene(mol) # Thiophene
    #functional_groups['num_unbranched_alkane'] = Chem.Fragments.fr_unbrch_alkane(mol) # Unbranched alkane
    functional_groups['num_urea'] = Chem.Fragments.fr_urea(mol) # Urea

    return functional_groups


def categorize_compounds(mol):
    functional_groups=find_functional_groups(mol)
    categories={}
    categories['carboxylic_acid']=       int(functional_groups['num_carboxylic_acid_any']>0)
    categories['polycarboxylic_acid']=   int(functional_groups['num_carboxylic_acid_any']>1)
    categories['amine']=                 int((functional_groups['num_primary_amine']+functional_groups['num_secondary_amine']+functional_groups['num_tertiary_amine'])>0)
    categories['aromatic_amine']=        int(functional_groups['num_aromatic_primary_secondary_amine']>0)
    categories['polyamine']=             int((functional_groups['num_primary_amine']+functional_groups['num_secondary_amine']+functional_groups['num_tertiary_amine'])+functional_groups['num_aromatic_primary_secondary_amine']>2)
    categories['pyridine']=              int(functional_groups['num_pyridine']>0)
    categories['amino_acid']=            int(functional_groups['num_carboxylic_acid_any']>1 and (categories['amine']==1 or functional_groups['num_quaternary_nitrogen']>0))
    categories['thiol']=                 int(functional_groups['num_thiol']>0)
    categories['phenol']=                int(functional_groups['num_aromatic_hydroxyl']>0 or functional_groups['num_phenol']>1)
    categories['amidine']=               int(functional_groups['num_amidine']>0)
    categories['guanidine']=             int(functional_groups['num_guanidine']>0)
    categories['phosphoric_acid']=       int(functional_groups['num_phosphoric_acid']>0)
    categories['sulfonamide']=           int(functional_groups['num_sulfonamide']>0)
    categories['Nheterocycle']=          int(functional_groups['num_aromatic_nitrogen_heterocycle'] + functional_groups['num_aromatic_nitrogen']+functional_groups['num_oxazole']+functional_groups['num_tetrazole']+functional_groups['num_thiazole']>0)
    categories['N-N_compounds']=         int(functional_groups['num_hydrazine']+functional_groups['num_hydrazone']+functional_groups['num_hydrazine'] >0)
    categories['N-O_compounds']=         int(functional_groups['num_N_oxide_or_oxime']+functional_groups['num_hydroxyamine']>0)
    categories['C=N_compounds']=         int(functional_groups['num_imine']+functional_groups['num_hydrazone']+functional_groups['num_oxime']>0)
    categories['imide']=                 int(functional_groups['num_imide']>0)

    return categories

labels_route= extracted_data_route
#the csv file containing the experimental pka values
labels=pd.read_csv(labels_route+labels_csv_file_name,encoding='unicode_escape')
print (labels)
labels.set_index("compn",inplace=True)
labels.dropna(how='all', axis=1, inplace=True)

for compn in labels.index[0:10]:

    inchi=labels.loc[compn,'inchi']

    if inchi.startswith("["): inchi=eval(inchi)[0]
    print (compn, inchi)
    mol = Chem.MolFromInchi(inchi)
    print (find_functional_groups(mol))
    mol = Chem.MolFromInchi(inchi)
    #print(rdkit.Chem.Descriptors.NumRotatableBonds(mol))

    print(rdMolDescriptors.CalcNumRotatableBonds(mol, strict=rdkit.Chem.rdMolDescriptors.NumRotatableBondsOptions.Strict))
    print (categorize_compounds(mol))
    #print(rdkit.Chem.Lipinski.NumRotatableBonds(mol))
    #print(rdMolDescriptors.CalcNumHeterocycles(mol))

    #print (rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    #print(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    Chem.RemoveStereochemistry(mol)
    scaff=MurckoScaffold.GetScaffoldForMol(mol)
    print(Chem.MolToSmiles(scaff))

    print ("....")
    #descriptors = rdkit.Chem.rdMolDescriptors.CalcCrippenDescriptors(str,includeHs=True)
    #print("descriptors are: {}".format(descriptors))
    #print(Descriptors.NumRotatableBonds(mol))

#smiles="COc1cc(NS(C)(=O)=O)ccc1Nc1c2ccccc2nc2c(C(=O)N(C)C)cccc12"
#smiles="C(C(C(=O)O)O)C(=O)O"
#smiles="C1=CC=C(C=C1)O"
#print (find_functional_groups(smiles))

with open("/home/lsimon/jobs/chembl_36_chemreps.txt", "r") as f: chemblines=f.readlines()[1:]
print (len(chemblines))
chembl_inchis=[x.split()[2] for x in chemblines]
chembl_scaffolds={}
for c in chembl_inchis[:10000]:
    try:
        #print ("---")
        #print (c)
        mol2 = Chem.MolFromInchi(c)
        #print (find_functional_groups(mol2))
        #print (rdMolDescriptors.CalcNumRotatableBonds(mol2, strict=rdkit.Chem.rdMolDescriptors.NumRotatableBondsOptions.Strict))
        #print (categorize_compounds(mol2))
        Chem.RemoveStereochemistry(mol2)
        scaff=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol2),canonical=True)
        #print (rdkit.Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol2)))
        #print (rdkit.Chem.rdinchi.MolToInchi(MurckoScaffold.GetScaffoldForMol(mol2)))
        #scaff=rdkit.Chem.rdinchi.MolToInchiKey(MurckoScaffold.GetScaffoldForMol(mol2))

        if scaff!="":
            if scaff not in chembl_scaffolds.keys(): chembl_scaffolds[scaff]=1
            else: chembl_scaffolds[scaff]+=1
    except: continue
    #print ("---")  
chembl_scaffolds_sorted = dict(sorted(chembl_scaffolds.items(), key=lambda item: item[1], reverse=True))
for k in chembl_scaffolds_sorted.keys(): print (k,chembl_scaffolds_sorted[k])