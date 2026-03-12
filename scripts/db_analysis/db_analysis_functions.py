#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
from rdkit import Chem
import rdkit
from rdkit.Chem import Fragments
from rdkit.Chem import Descriptors 
from rdkit.Chem import rdMolDescriptors
import numpy as np


import inspect
def find_functional_groups(mol):

    if mol is None: return {"error": "no molecule passed in"}
    # Get all attributes of Chem.Fragments
    all_fragments_attributes = dir(Chem.Fragments)
    #print("all_fragments_attributes:",all_fragments_attributes)
    # Filter for callable functions starting with 'fr_'
    fr_functions_to_execute = [
        attr for attr in all_fragments_attributes
        if attr.startswith('fr_') and inspect.isfunction(getattr(Chem.Fragments, attr))
    ]
    #print("funtions_to_execute",fr_functions_to_execute)
    results={}
    for func_name in sorted(fr_functions_to_execute):
        try:
            func = getattr(Chem.Fragments, func_name)
            results[func_name]=func(mol)
        except Exception as e:
            results[func_name]=0

    results["num_rings"]=rdMolDescriptors.CalcNumRings(mol)
    results["num_aromatic_rings"]=rdMolDescriptors.CalcNumAromaticRings(mol)
    results["num_aliphatic_rings"]=rdMolDescriptors.CalcNumAliphaticRings(mol)
    results["num_aromatic_carbocycles"]=rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    results["num_aliphatic_carbocycles"]=rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    results["num_saturated_carbocycles"]=rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
    results["num_heterocycles"]=rdMolDescriptors.CalcNumHeterocycles(mol)
    results["num_aliphatic_heterocycles"]=rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
    results["num_aromatic_heterocycles"]=rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    results["num_saturated_heterocycles"]=rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
    results["num_hba"]=rdMolDescriptors.CalcNumHBA(mol)
    results["num_hbd"]=rdMolDescriptors.CalcNumHBD(mol)
    results["num_aromatic_rings"]=rdMolDescriptors.CalcNumAromaticRings(mol)
    results["num_rotatable_bonds"]=rdMolDescriptors.CalcNumRotatableBonds(mol)

    return results


def shannon_entropy_scaffolds(scaffolds):
    scaffolds=list(scaffolds)
    unique_scaffolds=list(set(scaffolds))
    entropy=0
    for scaffold in unique_scaffolds:
        p=len([x for x in scaffolds if x==scaffold])/len(scaffolds)
        if p!=0: entropy+=p*np.log2(p)
    return -entropy

def normalized_shannon_entropy_scaffolds(scaffolds):
    return shannon_entropy_scaffolds(scaffolds)/np.log2(len(scaffolds))

def tanemoto_similarity_functional_groups(funct_groups1,funct_groups2):
    keys=list(funct_groups1.keys()&funct_groups2.keys())
    v1=[funct_groups1[k] for k in keys]
    v2=[funct_groups2[k] for k in keys]
    c=[min([vv1,vv2]) for vv1,vv2 in zip(v1,v2)]
    a,b=np.sum(v1),np.sum(v2)
    return c/(a+b-c)



