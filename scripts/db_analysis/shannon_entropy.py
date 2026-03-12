#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-


import string
import os
import os.path
import sys
import numpy as np
import pandas as pd

from db_analysis_functions import shannon_entropy_scaffolds

dbs=["Gpka_database_with_descriptors.csv","PubChem_with_pka_database_with_descriptors.csv","CHEMBL_with_pka_database_with_descriptors.csv"]
db_names=["Gpka database","PubChem with pKa data database:","CHEMBL with pka data database"]

for db, dbn in zip(dbs,db_names):
    
    df=pd.read_csv(db)
    print ("shannon entropy in scaffolds for ",dbn,":    ",shannon_entropy_scaffolds(df["Murcko_Scaffold"]))



