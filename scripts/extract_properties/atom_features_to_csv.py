#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-



import json
import numpy as np
import pandas as pd
import joblib
import sklearn
import os
import types
import copy
import sys
sys.path.append('../import')
import gpka_spektral_dataset
from routes import extracted_data_route


if "-weighting" in sys.argv:  weighting=sys.argv[sys.argv.index("-weighting")+1]
else: weighting="gibbs" # "gibbs", "sp", or "zero"
if "-lot" in sys.argv:  level_of_theory=sys.argv[sys.argv.index("-lot")+1]
else: level_of_theory="sM06" # "M06","sM06","swb97xd","wb97xd","pbeh3c"
if "suffix" in sys.argv: output_suffix=sys.argv[sys.argv.index("-suffix")+1]
else: suffix="25"

if weighting not in ["gibbs","sp","zero"]: print ("weighting should be gibbs, sp, or zero"); sys.exit()
if level_of_theory not in ["M06","sM06","swb97xd","wb97xd","pbeh3c"]: print ("lot must be M06, sM06, swb97xd, wb97xd, or pbeh3c"); sys.exit()


json_file=extracted_data_route+  "molecular_graphs"+weighting+"-"+level_of_theory+"."+output_suffix+".json"
csv_file= extracted_data_route+  "values_extracted"+weighting+"-"+level_of_theory+"."+output_suffix+".json"

json_file="molecular_graphs-gibbs-swb97xd.25.json"
#json_file="test-spk.json"
csv_file="values_extracted-gibbs-swb97xd.25.csv" 

"""
feature_keys=["isN_protonated","isC_protonatd","isO_protonated","isS_protonated","protonated Hirshfeld", "deprotonated Hirshfeld", "Hirshfeld", "protonated Voronoy", "deprotonated Voronoy", "Voronoy", "protonated Mulliken", "deprotonated Mulliken", "Mulliken", "protonated Lowdin", "deprotonated Lowdin", "Lowdin", "protonated Becke", "deprotonated Becke", "Becke", "protonated ADCH", "deprotonated ADCH", "ADCH", "protonated CHELPG", "deprotonated CHELPG", "CHELPG", "protonated MK", "deprotonated MK", "MK", "protonated CM5", "deprotonated CM5", "CM5", "protonated 12CM5", "deprotonated 12CM5", "12CM5", "protonated RESP", "deprotonated RESP", "RESP", "protonated PEOE", "deprotonated PEOE", "PEOE", "protonated (a)Surf", "deprotonated (a)Surf", "(a)Surf", "protonated (a)Surf+", "deprotonated (a)Surf+", "(a)Surf+", "protonated (a)Surf-", "deprotonated (a)Surf-", "(a)Surf-", "protonated (a)min-ESP", "deprotonated (a)min-ESP", "(a)min-ESP", "protonated (a)max-ESP", "deprotonated (a)max-ESP", "(a)max-ESP", "protonated (a)avg-ESP", "deprotonated (a)avg-ESP", "(a)avg-ESP", "protonated (a)avg-ESP+", "deprotonated (a)avg-ESP+", "(a)avg-ESP+", "protonated (a)avg-ESP-", "deprotonated (a)avg-ESP-", "(a)avg-ESP-", "protonated (a)var-ESP", "deprotonated (a)var-ESP", "(a)var-ESP", "protonated (a)var-ESP+", "deprotonated (a)var-ESP+", "(a)var-ESP+", "protonated (a)var-ESP-", "deprotonated (a)var-ESP-", "(a)var-ESP-", "protonated (a)*PI-ESP", "deprotonated (a)*PI-ESP", "(a)*PI-ESP", "protonated (a)*mu", "deprotonated (a)*mu", "(a)*mu", "protonated (a)*mu-ctb", "deprotonated (a)*mu-ctb", "(a)*mu-ctb", "protonated (a)tr-e*theta", "deprotonated (a)tr-e*theta", "(a)tr-e*theta", "protonated ESP-nucl", "deprotonated ESP-nucl", "ESP-nucl", "protonated (a)avg-ALIE", "deprotonated (a)avg-ALIE", "(a)avg-ALIE", "protonated (a)var-ALIE", "deprotonated (a)var-ALIE", "(a)var-ALIE", "protonated (a)max-ALIE", "deprotonated (a)max-ALIE", "(a)max-ALIE", "protonated (a)min-ALIE", "deprotonated (a)min-ALIE", "(a)min-ALIE", "protonated (a)min-LEA", "deprotonated (a)min-LEA", "(a)min-LEA", "protonated (a)max-LEA", "deprotonated (a)max-LEA", "(a)max-LEA", "protonated (a)avg-LEA", "deprotonated (a)avg-LEA", "(a)avg-LEA", "protonated (a)var-LEA", "deprotonated (a)var-LEA", "(a)var-LEA", "isC_protonated", "isN_protonated", "isO_protonated", "isS_protonated", "protonated NBO-chg", "deprotonated NBO-chg", "NBO-chg", "protonated NMR*delta", "deprotonated NMR*delta", "NMR*delta"]
"""

H_feature_keys=["protonated Mayer-BO-*H", "deprotonated Mayer-BO-*H", "Mayer-BO-*H", 
                "protonated WBO-*H", "deprotonated WBO-*H", "WBO-*H", 
                "protonated Mulliken-BO-*H", "deprotonated Mulliken-BO-*H", 
                "Mulliken-BO-*H", "protonated FBO-*H", "deprotonated FBO-*H", 
                "FBO-*H", "protonated LBO-*H", "deprotonated LBO-*H", "LBO-*H", 
                "protonated IBSI-*H", "deprotonated IBSI-*H", "IBSI-*H", 
                "protonated FUERZA-FC-*H", "deprotonated FUERZA-FC-*H", "FUERZA-FC-*H", 
                "protonated FUERZA-FC*relative*H", "deprotonated FUERZA-FC*relative*H", 
                "FUERZA-FC*relative*H", 
                "protonated 1/BD-*H", "deprotonated 1/BD-*H","1/BD-*H", 
                "protonated BD-*H", "deprotonated BD-*H", "BD-*H", 
                "protonated BD*relative*H", "deprotonated BD*relative*H", "BD*relative*H", 
                "protonated NBI-*H", "deprotonated NBI-*H", "NBI-*H", 
                "protonated WBO-NAO-*H", "deprotonated WBO-NAO-*H", "WBO-NAO-*H", 
                "protonated NLMO-BO-*H", "deprotonated NLMO-BO-*H",  "NLMO-BO-*H", 
                "protonated Mayer-BO*relative*H", "deprotonated Mayer*relativeBO-*H", "Mayer-BO*relative*H", 
                "protonated WBO*relative*H", "deprotonated WBO*relative*H", "WBO*relative*H", 
                "protonated Mulliken-BO*relative*H", "deprotonated Mulliken-BO*relative*H", "Mulliken-BO*relative*H", 
                "protonated FBO*relative*H", "deprotonated FBO*relative*H", "FBO*relative*H", 
                "protonated LBO*relative*H", "deprotonated LBO*relative*H", "LBO*relative*H", 
                "protonated IBSI*relative*H", "deprotonated IBSI*relative*H", "IBSI*relative*H", 
                "protonated NBI*relative*H", "deprotonated NBI*relative*H", "NBI*relative*H", 
                "protonated WBO-NAO*relative*H", "deprotonated WBO-NAO*relative*H", "WBO-NAO*relative*H", 
                "protonated NLMO-BO*relative*H", "deprotonated NLMO-BO*relative*H",  "NLMO-BO*relative*H",
                "protonated *mu*BP-*H", "deprotonated *mu*BP-*H", "*mu*BP-*H", 
                "protonated *ind*mu*BP-*H", "deprotonated *ind*mu*BP-*H", "*ind*mu*BP-*H", 
                "protonated diag-e*theta*BP-*H", "deprotonated diag-e*theta*BP-*H", "diag-e*theta*BP-*H", 
                "protonated diag-*theta*BP-*H", "deprotonated diag-*theta*BP-*H", "diag-*theta*BP-*H"]

#linear=["deltaZPE"]
with open(csv_file,"r") as f: eq_keys=[k.strip() for k in f.readlines()[0].split(",")]
dataset=gpka_spektral_dataset.gpka_spektral_dataset(json_file,csv_file=csv_file,label_key="pKa",equilibrium_keys="")

dataset.features_n_bonds_away(suffixes=["alpha"],exclude_ending="")
dataset.features_n_bonds_away(suffixes=["beta"],exclude_ending=["-*H","-*H+","-*H-","relative*H","relative*H+","relative*H+"])
dataset.features_n_bonds_away(suffixes=["gamma"],exclude_ending=["-*H","-*H+","-*H-","relative*H","relative*H+","relative*H+"])
dataset.aply_mask_to_atom_features(H_feature_keys,replace=True,weighting=True)

pd_dataframe=dataset.eq_features_to_pd_series()
pd_dataframe.to_csv("values+alpha-beta-gamma-gibbs-swb97xd.csv")

