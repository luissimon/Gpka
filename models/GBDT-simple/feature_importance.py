#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go
import plotly
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import time
import sys
import math
import copy
import json
import joblib 

import sys
#imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
#sys.path.insert(0,imports_path)
sys.path.insert(0,"../../scripts/import")
from correlated_groups import correlated_groups
from drop_compounds import drop_compounds
from drop_compounds import force_in_test_set_compounds
from drop_compounds import force_in_train_set_compounds
from prepare_data import prepare_eq_data
from prepare_data import prepare_graph_data
from prepare_data import prepare_graph_data_to_ML
from routes import extracted_data_route

from composed_regressor import composed_regressor


# define groups of correlated features
correlated_groups={}

correlated_groups["lin_eq_features_groups"]=[["deltaZPE","deltaE","deltaG"],
                                            # ["SMD-solv"],
                                            # ["expl1wat"]
                                            ]

correlated_groups["eq_features_groups"]=[["SMD-solv"],["expl1wat"],
                                         ["RDG%HB"],["RDG%VdW"],["RDG%st"],
                                         ["prom-RDG%HB"],["prom-RDG%VdW"],["prom-RDG%st"],
                                         ["*mu"],
                                         ["tr-e*theta"],
                                         ["tr-*alpha"],
                                         ["HLgap"],
                                         ["Vol"],
                                         ["Surf"],
                                         ["Surf+"],#"Surf-"],
                                         ["min-ESP"],
                                         ["max-ESP"],
                                         ["avg-ESP","avg-ALIE"],
                                         ["avg-ESP+"],
                                         ["avg-ESP-"],
                                         ["var-ESP"],
                                         ["var-ESP+"],
                                         ["var-ESP-"],
                                         ["*PI-ESP"],
                                         ["MPI"],
                                         ["min-LEA"],
                                         ["max-LEA"],
                                         ["avg-LEA"],
                                         ["var-LEA"],
                                         ["min-ALIE"],
                                         ["max-ALIE"],
                                         ["var-ALIE"],
                                         ["avg|EF|","0.75q|EF|"],#"0.5q|EF|"],
                                         ["avgEF*tang","0.75qEF*tang"],#"0.5qEF*tang"],
                                         ["avgEF*norm","0.75qEF*norm"],#"0.5qEF*norm"],
                                         ["0.95q|EF|","0.9q|EF|"],
                                         ["0.95qEF*tang","0.9qEF*tang"],
                                         ["0.95qEF*norm","0.9qEF*norm"],
                                         ["avgEF*angle","0.75qEF*angle"],#"0.5qEF*angle"],
                                         ["0.95qEF*angle","0.9qEF*angle"],
                                         ]    

correlated_groups["categorical_features"]=['protonated charge']

correlated_groups["atomic_features_groups"]= [["Hirshfeld","Voronoy","Lowdin","CM5","12CM5","NBO-chg"],
                                              ["Mulliken"],
                                              ["Becke"],
                                              ["ADCH"],
                                              ["CHELPG","MK","RESP"],
                                              ["PEOE"],
                                              ["(a)Surf"],
                                              ["(a)Surf-","(a)Surf+"],
                                              ["(a)avg-ESP","(a)min-ESP","(a)max-ESP"],
                                              ["(a)avg-ESP+"],
                                              ["(a)avg-ESP-"],
                                              ["(a)var-ESP"],
                                              ["(a)var-ESP+"],
                                              ["(a)var-ESP-"],
                                              ["(a)*PI-ESP"],
                                              ["ESP-nucl"],
                                              ["NMR*delta"],
                                              ["(a)*mu"],
                                              ["(a)*mu-ctb"],
                                              ["(a)tr-e*theta"],
                                              ["(a)avg-ALIE","(a)max-ALIE","(a)min-ALIE","(a)avg-LEA","(a)max-LEA","(a)min-LEA"],
                                              ["(a)var-ALIE"],
                                              ["(a)var-LEA"],
                                    ]

#properties of ionizable H atom that will be projected from vector or matrix properties 
correlated_groups["alpha_masked_eq_features_groups"]= [ ["PEOE-*H"],
                                                        ["PEOE*relative*H"],
                                                        ["Mulliken-*H"],
                                                        ["Mulliken*relative*H"],
                                                        ["Hirshfeld-*H","Voronoy-*H","Lowdin-*H","CM5-*H","12CM5-*H"],
                                                        ["Hirshfeld*relative*H","Voronoy*relative*H","Lowdin*relative*H","CM5*relative*H","12CM5*relative*H"],
                                                        ["ADCH-*H"],
                                                        ["ADCH*relative*H"],
                                                        ["CHELPG-*H","MK-*H","RESP-*H"],
                                                        ["CHELPG*relative*H","MK*relative*H","RESP*relative*H"],
                                                        ["NBO-chg-*H"],
                                                        ["NBO-chg*relative*H"],
                                                        ["NMR*delta-*H"],
                                                        ["NMR*delta*relative*H"],
                                                        ["ESP-nucl-*H"],
                                                        ["ESP-nucl*relative*H"],

                                                        ["Mayer-BO-*H"],
                                                        ["Mayer-BO*relative*H"],
                                                        ["WBO-*H","WBO-NAO-*H","NLMO-BO-*H"],
                                                        ["WBO*relative*H","WBO-NAO*relative*H","NLMO-BO*relative*H"],
                                                        ["Mulliken-BO-*H"],
                                                        ["Mulliken-BO*relative*H"],
                                                        ["FBO-*H"],
                                                        ["FBO*relative*H"],
                                                        ["LBO-*H"],
                                                        ["LBO*relative*H"],
                                                        ["IBSI-*H"],
                                                        ["IBSI*relative*H"],
                                                        ["FUERZA-FC-*H"],
                                                        ["FUERZA-FC*relative*H"],
                                                        ["BD-*H"],
                                                        ["BD*relative*H"],
                                                        ["*mu*BP-*H"],
                                                        ["*ind*mu*BP-*H"],
                                                        ["diag-e*theta*BP-*H"],
                                                        ["diag-*theta*BP-*H"]                                                  
                                                      ] 


correlated_groups["eq_features_groups_all"]=[]
for g in correlated_groups["eq_features_groups"]:
    correlated_groups["eq_features_groups_all"].append(g)
    correlated_groups["eq_features_groups_all"].append(["protonated "+gg for gg in g])
    correlated_groups["eq_features_groups_all"].append(["deprotonated "+gg for gg in g])

correlated_groups["atomic_features_groups_alpha"]=[]
for g in correlated_groups["atomic_features_groups"]:
    correlated_groups["atomic_features_groups_alpha"].append([gg+"_alpha" for gg in g] )
    correlated_groups["atomic_features_groups_alpha"].append(["protonated "+gg+"_alpha" for gg in g])
    correlated_groups["atomic_features_groups_alpha"].append(["deprotonated "+gg+"_alpha" for gg in g])

correlated_groups["atomic_features_groups_beta"]=[]
for g in correlated_groups["atomic_features_groups"]:
    correlated_groups["atomic_features_groups_beta"].append([gg+"_beta" for gg in g] )
    correlated_groups["atomic_features_groups_beta"].append(["protonated "+gg+"_beta" for gg in g])
    correlated_groups["atomic_features_groups_beta"].append(["deprotonated "+gg+"_beta" for gg in g])

correlated_groups["relative_H"]=[]
for g in correlated_groups["alpha_masked_eq_features_groups"]:
    if g[0].endswith("*relative*H"):
        correlated_groups["relative_H"].append([gg for gg in g] )


#define linear features as passed to composed regressor:
linear_features=['deltaG','deltaE','deltaZPE']

non_linear_features=['SMD-solv', 'expl1wat', 'protonated charge', 
                     'RDG%HB', 'RDG%VdW', 'RDG%st', 'prom-RDG%HB', 
                     'prom-RDG%VdW', 'prom-RDG%st', '*mu', 'tr-e*theta', 
                     'tr-*alpha', 'HLgap', 'Vol', 'Surf', 'Surf+', 
                     'min-ESP', 'max-ESP', 'avg-ESP', 'avg-ESP+', 'avg-ESP-', 
                     'var-ESP', 'var-ESP+', 'var-ESP-', '*PI-ESP', 'MPI', 
                     'min-LEA', 'max-LEA', 'avg-LEA', 'var-LEA', 'avg-ALIE','min-ALIE', 'max-ALIE', 'var-ALIE', 
                     'avg|EF|', 'avgEF*tang', 'avgEF*norm', '0.95q|EF|', '0.95qEF*tang', '0.95qEF*norm', 
                     '0.9q|EF|', '0.9qEF*tang', '0.9qEF*norm','0.75q|EF|', '0.75qEF*tang', '0.75qEF*norm',
                     'avgEF*angle', '0.95qEF*angle', '0.9qEF*angle','0.75qEF*angle',
                     
                     'Hirshfeld_alpha','Voronoy_alpha', 'Lowdin_alpha', 'CM5_alpha', '12CM5_alpha',
                     'Mulliken_alpha', 'Becke_alpha', 
                     'ADCH_alpha', 'CHELPG_alpha', 'MK_alpha', 'RESP_alpha',
                     'PEOE_alpha', 'NBO-chg_alpha', 
                     '(a)Surf_alpha', '(a)Surf-_alpha', '(a)Surf+_alpha',
                     '(a)avg-ESP_alpha', '(a)min-ESP_alpha', '(a)max-ESP_alpha','(a)avg-ESP+_alpha',

                     '(a)avg-ESP-_alpha', '(a)var-ESP_alpha', '(a)var-ESP+_alpha', '(a)var-ESP-_alpha', 
                     '(a)*PI-ESP_alpha', 'ESP-nucl_alpha', 'NMR*delta_alpha', '(a)*mu_alpha', '(a)*mu-ctb_alpha', 
                     '(a)tr-e*theta_alpha', 
                     '(a)avg-ALIE_alpha', '(a)max-ALIE_alpha', '(a)min-ALIE_alpha', '(a)avg-LEA_alpha', '(a)max-LEA_alpha', '(a)min-LEA_alpha',
                     '(a)var-ALIE_alpha', '(a)var-LEA_alpha',

                     'Hirshfeld_beta','Voronoy_beta', 'Lowdin_beta', 'CM5_beta', '12CM5_beta',
                     'Mulliken_beta', 'Becke_beta', 'ADCH_beta', 'CHELPG_beta', 'MK_beta', 'RESP_beta', 'PEOE_beta', 

                     'NBO-chg_beta', '(a)Surf_beta', '(a)Surf-_beta','(a)Surf+_beta', 
                     '(a)avg-ESP_beta',  '(a)min-ESP_beta', '(a)max-ESP_beta','(a)avg-ESP+_beta',

                     '(a)avg-ESP-_beta', '(a)var-ESP_beta', '(a)var-ESP+_beta', '(a)var-ESP-_beta', '(a)*PI-ESP_beta', 
                     'ESP-nucl_beta', 'NMR*delta_beta', '(a)*mu_beta', '(a)*mu-ctb_beta', '(a)tr-e*theta_beta', 
                     '(a)avg-ALIE_beta', '(a)max-ALIE_beta', '(a)min-ALIE_beta', '(a)avg-LEA_beta', '(a)max-LEA_beta', '(a)min-LEA_beta',
                     '(a)var-ALIE_beta', '(a)var-LEA_beta', 

                     'PEOE*relative*H', 'Mulliken*relative*H', 
                     'Hirshfeld*relative*H', 'Voronoy*relative*H', 'Lowdin*relative*H', 'CM5*relative*H', '12CM5*relative*H',
                     'ADCH*relative*H', 'CHELPG*relative*H', 'MK*relative*H', 'RESP*relative*H',
                     'NBO-chg*relative*H', 
                     'NMR*delta*relative*H', 'ESP-nucl*relative*H', 'Mayer-BO*relative*H', 
                     'WBO*relative*H', 'WBO-NAO*relative*H', 'NLMO-BO*relative*H',
                     'Mulliken-BO*relative*H', 'FBO*relative*H', 'LBO*relative*H', 'IBSI*relative*H', 'FUERZA-FC*relative*H', 
                     'BD*relative*H', '*mu*BP-*H', '*ind*mu*BP-*H', 'diag-e*theta*BP-*H', 'diag-*theta*BP-*H'
                     ]

non_linear_features_protonated=['protonated SMD-solv', 'protonated expl1wat', 'protonated charge', 
                     'protonated RDG%HB', 'protonated RDG%VdW', 'protonated RDG%st', 'protonated prom-RDG%HB', 
                     'protonated prom-RDG%VdW', 'protonated prom-RDG%st', 'protonated *mu', 'protonated tr-e*theta', 
                     'protonated tr-*alpha', 'protonated HLgap', 'protonated Vol', 'protonated Surf', 'protonated Surf+', 
                     'protonated min-ESP', 'protonated max-ESP', 'protonated avg-ESP', 'protonated avg-ESP+', 'protonated avg-ESP-', 
                     'protonated var-ESP', 'protonated var-ESP+', 'protonated var-ESP-', 'protonated *PI-ESP', 'protonated MPI', 
                     'protonated min-LEA', 'protonated max-LEA', 'protonated avg-LEA', 'protonated var-LEA', 'protonated avg-ALIE','protonated min-ALIE', 'protonated max-ALIE', 'protonated var-ALIE', 
                     'protonated avg|EF|', 'protonated avgEF*tang', 'protonated avgEF*norm', 'protonated 0.95q|EF|', 'protonated 0.95qEF*tang', 'protonated 0.95qEF*norm', 
                     'protonated 0.9q|EF|', 'protonated 0.9qEF*tang', 'protonated 0.9qEF*norm','protonated 0.75q|EF|', 'protonated 0.75qEF*tang', 'protonated 0.75qEF*norm',
                     'protonated avgEF*angle', 'protonated 0.95qEF*angle', 'protonated 0.9qEF*angle','protonated 0.75qEF*angle',

                     'protonated Hirshfeld_alpha','protonated Voronoy_alpha', 'protonated Lowdin_alpha', 'protonated CM5_alpha', 'protonated 12CM5_alpha',
                     'protonated Mulliken_alpha', 'protonated Becke_alpha', 
                     'protonated ADCH_alpha', 'protonated CHELPG_alpha','protonated MK_alpha', 'protonated RESP_alpha',
                     'protonated PEOE_alpha', 'protonated NBO-chg_alpha', 
                     'protonated (a)Surf_alpha', 'protonated (a)Surf-_alpha', 'protonated (a)Surf+_alpha',
                     'protonated (a)avg-ESP_alpha', 'protonated (a)min-ESP_alpha', 'protonated (a)max-ESP_alpha','protonated (a)avg-ESP+_alpha',
                     'protonated (a)avg-ESP-_alpha', 'protonated (a)var-ESP_alpha', 'protonated (a)var-ESP+_alpha', 'protonated (a)var-ESP-_alpha', 
                     'protonated (a)*PI-ESP_alpha', 'protonated ESP-nucl_alpha', 'protonated NMR*delta_alpha', 'protonated (a)*mu_alpha', 'protonated (a)*mu-ctb_alpha', 
                     'protonated (a)tr-e*theta_alpha', 
                     'protonated (a)avg-ALIE_alpha', 'protonated (a)max-ALIE_alpha', 'protonated (a)min-ALIE_alpha', 'protonated (a)avg-LEA_alpha', 
                     'protonated (a)max-LEA_alpha', 'protonated (a)min-LEA_alpha', 'protonated (a)var-ALIE_alpha', 'protonated (a)var-LEA_alpha', 

                     'protonated Hirshfeld_beta','protonated Voronoy_beta', 'protonated Lowdin_beta', 'protonated CM5_beta', 'protonated 12CM5_beta',                      
                     'protonated Mulliken_beta', 'protonated Becke_beta', 'protonated ADCH_beta', 
                     'protonated CHELPG_beta', 'protonated MK_beta', 'protonated RESP_beta','protonated PEOE_beta', 
                     'protonated NBO-chg_beta', 'protonated (a)Surf_beta','protonated (a)Surf+_beta', 'protonated (a)Surf-_beta', 
                     'protonated (a)avg-ESP_beta', 'protonated (a)min-ESP_beta', 'protonated (a)max-ESP_beta','protonated (a)avg-ESP+_beta', 
                     'protonated (a)avg-ESP-_beta', 'protonated (a)var-ESP_beta', 'protonated (a)var-ESP+_beta', 'protonated (a)var-ESP-_beta', 'protonated (a)*PI-ESP_beta', 
                     'protonated ESP-nucl_beta', 'protonated NMR*delta_beta', 'protonated (a)*mu_beta', 'protonated (a)*mu-ctb_beta', 'protonated (a)tr-e*theta_beta', 
                     'protonated (a)avg-ALIE_beta', 'protonated (a)max-ALIE_beta', 'protonated (a)min-ALIE_beta', 'protonated (a)avg-LEA_beta', 
                     'protonated (a)max-LEA_beta', 'protonated (a)min-LEA_beta', 'protonated (a)var-ALIE_beta', 'protonated (a)var-LEA_beta',
                     'PEOE*relative*H', 'Mulliken*relative*H', 
                     'Hirshfeld*relative*H', 'Voronoy*relative*H', 'Lowdin*relative*H', 'CM5*relative*H', '12CM5*relative*H',
                     'ADCH*relative*H', 'CHELPG*relative*H', 'MK*relative*H', 'RESP*relative*H',
                     'NBO-chg*relative*H', 
                     'NMR*delta*relative*H', 'ESP-nucl*relative*H', 'Mayer-BO*relative*H', 
                     'WBO*relative*H', 'WBO-NAO*relative*H', 'NLMO-BO*relative*H',
                     'Mulliken-BO*relative*H', 'FBO*relative*H', 'LBO*relative*H', 'IBSI*relative*H', 'FUERZA-FC*relative*H', 
                     'BD*relative*H', '*mu*BP-*H', '*ind*mu*BP-*H', 'diag-e*theta*BP-*H', 'diag-*theta*BP-*H'
                     ]

non_linear_features_deprotonated=['deprotonated SMD-solv', 'deprotonated expl1wat', 'protonated charge', 
                     'deprotonated RDG%HB', 'deprotonated RDG%VdW', 'deprotonated RDG%st', 'deprotonated prom-RDG%HB', 
                     'deprotonated prom-RDG%VdW', 'deprotonated prom-RDG%st', 'deprotonated *mu', 'deprotonated tr-e*theta', 
                     'deprotonated tr-*alpha', 'deprotonated HLgap', 'deprotonated Vol', 'deprotonated Surf', 'deprotonated Surf+', 
                     'deprotonated min-ESP', 'deprotonated max-ESP', 'deprotonated avg-ESP', 'deprotonated avg-ESP+', 'deprotonated avg-ESP-', 
                     'deprotonated var-ESP', 'deprotonated var-ESP+', 'deprotonated var-ESP-', 'deprotonated *PI-ESP', 'deprotonated MPI', 
                     'deprotonated min-LEA', 'deprotonated max-LEA', 'deprotonated avg-LEA', 'deprotonated var-LEA', 'deprotonated avg-ALIE','deprotonated min-ALIE', 'deprotonated max-ALIE', 'deprotonated var-ALIE', 
                     'deprotonated avg|EF|', 'deprotonated avgEF*tang', 'deprotonated avgEF*norm', 'deprotonated 0.95q|EF|', 'deprotonated 0.95qEF*tang', 'deprotonated 0.95qEF*norm', 
                     'deprotonated 0.9q|EF|', 'deprotonated 0.9qEF*tang', 'deprotonated 0.9qEF*norm','deprotonated 0.75q|EF|', 'deprotonated 0.75qEF*tang', 'deprotonated 0.75qEF*norm',
                     'deprotonated avgEF*angle', 'deprotonated 0.95qEF*angle', 'deprotonated 0.9qEF*angle','deprotonated 0.75qEF*angle',

                     'deprotonated Hirshfeld_alpha','deprotonated Voronoy_alpha', 'deprotonated Lowdin_alpha', 'deprotonated CM5_alpha', 'deprotonated 12CM5_alpha',
                     'deprotonated Mulliken_alpha', 'deprotonated Becke_alpha', 
                     'deprotonated ADCH_alpha', 'deprotonated CHELPG_alpha','deprotonated MK_alpha', 'deprotonated RESP_alpha',
                     'deprotonated PEOE_alpha', 'deprotonated NBO-chg_alpha', 
                     'deprotonated (a)Surf_alpha', 'deprotonated (a)Surf-_alpha', 'deprotonated (a)Surf+_alpha',
                     'deprotonated (a)avg-ESP_alpha', 'deprotonated (a)min-ESP_alpha', 'deprotonated (a)max-ESP_alpha','deprotonated (a)avg-ESP+_alpha',
                     'deprotonated (a)avg-ESP-_alpha', 'deprotonated (a)var-ESP_alpha', 'deprotonated (a)var-ESP+_alpha', 'deprotonated (a)var-ESP-_alpha', 
                     'deprotonated (a)*PI-ESP_alpha', 'deprotonated ESP-nucl_alpha', 'deprotonated NMR*delta_alpha', 'deprotonated (a)*mu_alpha', 'deprotonated (a)*mu-ctb_alpha', 
                     'deprotonated (a)tr-e*theta_alpha', 
                     'deprotonated (a)avg-ALIE_alpha', 'deprotonated (a)max-ALIE_alpha', 'deprotonated (a)min-ALIE_alpha', 'deprotonated (a)avg-LEA_alpha', 
                     'deprotonated (a)max-LEA_alpha', 'deprotonated (a)min-LEA_alpha', 'deprotonated (a)var-ALIE_alpha', 'deprotonated (a)var-LEA_alpha', 

                     'deprotonated Hirshfeld_beta','deprotonated Voronoy_beta', 'deprotonated Lowdin_beta', 'deprotonated CM5_beta', 'deprotonated 12CM5_beta',                      
                     'deprotonated Mulliken_beta', 'deprotonated Becke_beta', 'deprotonated ADCH_beta', 
                     'deprotonated CHELPG_beta', 'deprotonated MK_beta', 'deprotonated RESP_beta','deprotonated PEOE_beta', 
                     'deprotonated NBO-chg_beta', 'deprotonated (a)Surf_beta','deprotonated (a)Surf+_beta', 'deprotonated (a)Surf-_beta', 
                     'deprotonated (a)avg-ESP_beta', 'deprotonated (a)min-ESP_beta', 'deprotonated (a)max-ESP_beta','deprotonated (a)avg-ESP+_beta', 
                     'deprotonated (a)avg-ESP-_beta', 'deprotonated (a)var-ESP_beta', 'deprotonated (a)var-ESP+_beta', 'deprotonated (a)var-ESP-_beta', 'deprotonated (a)*PI-ESP_beta', 
                     'deprotonated ESP-nucl_beta', 'deprotonated NMR*delta_beta', 'deprotonated (a)*mu_beta', 'deprotonated (a)*mu-ctb_beta', 'deprotonated (a)tr-e*theta_beta', 
                     'deprotonated (a)avg-ALIE_beta', 'deprotonated (a)max-ALIE_beta', 'deprotonated (a)min-ALIE_beta', 'deprotonated (a)avg-LEA_beta', 
                     'deprotonated (a)max-LEA_beta', 'deprotonated (a)min-LEA_beta', 'deprotonated (a)var-ALIE_beta', 'deprotonated (a)var-LEA_beta',
                     'PEOE*relative*H', 'Mulliken*relative*H', 
                     'Hirshfeld*relative*H', 'Voronoy*relative*H', 'Lowdin*relative*H', 'CM5*relative*H', '12CM5*relative*H',
                     'ADCH*relative*H', 'CHELPG*relative*H', 'MK*relative*H', 'RESP*relative*H',
                     'NBO-chg*relative*H', 
                     'NMR*delta*relative*H', 'ESP-nucl*relative*H', 'Mayer-BO*relative*H', 
                     'WBO*relative*H', 'WBO-NAO*relative*H', 'NLMO-BO*relative*H',
                     'Mulliken-BO*relative*H', 'FBO*relative*H', 'LBO*relative*H', 'IBSI*relative*H', 'FUERZA-FC*relative*H', 
                     'BD*relative*H', '*mu*BP-*H', '*ind*mu*BP-*H', 'diag-e*theta*BP-*H', 'diag-*theta*BP-*H'
                     ]


def feature_importances(composed_regressor_params,data,algorithm="shap"): 

    #CV parallelism or model parallelism?
    if composed_regressor_params["l_n_jobs"]!=None or composed_regressor_params["l_n_jobs"]!=1:
        n_jobs=1
    else: n_jobs=14

    results={}
    if algorithm=="shap":
        print ("starting shap:")
        new_regressor_for_shap=composed_regressor( **composed_regressor_params )
        new_regressor_for_shap.do_shap(data,data["pKa"],fraction=1)
        for fi,f in sorted(zip(new_regressor_for_shap.shap_nl_feature_importances,new_regressor_for_shap.non_linear_attributes),reverse=True):
            results[f]=fi

    elif algorithm=="permutation_importances":
        print ("starting permutation_importances")
        new_regressor_for_permutation_importances=composed_regressor( **composed_regressor_params )
        new_regressor_for_permutation_importances.do_permutation_importances(data,data["pKa"],fraction=1,n_jobs=n_jobs)
        for fi,f in sorted(zip(new_regressor_for_permutation_importances.nl_permutation_feature_importances,new_regressor_for_permutation_importances.non_linear_attributes),reverse=True):
            results[f]=fi
    return results





if __name__=="__main__":


    #levels_of_theory=["pbeh3c","swb97xd","wb97xd","M06","sM06"]
    levels_of_theory=["swb97xd"]
    features_included="all"
    if features_included=="all": non_linear_features=list(set(non_linear_features+non_linear_features_protonated+non_linear_features_deprotonated)) #uncomment to include all features, also protonated and deprotonated
    elif features_included=="protonated": non_linear_features= non_linear_features2
    elif features_included=="deprotonated": non_linear_features=non_linear_features3
    elif features_included=="protonated+deprotonated": non_linear_features=list(set(non_linear_features2+non_linear_features3))
    elif features_included=="difference": non_linear_features=non_linear_features


    for lot in levels_of_theory:
        #name of files:
        csv_file=extracted_data_route+"values_extracted-gibbs-"+lot+".25.csv" 
        json_file=extracted_data_route+"/molecular_graphs-gibbs-"+lot+".25.json"

        data_file=csv_file[:-4]+"_with_graph_data.csv"
        data_file=data_file.split("/")[-1]
        #test_data_file=csv_file[:-4]+"_std_test_with_graph_data.csv"
        #test_data_file=test_data_file.split("/")[-1]

        #prepare_eq_data(file_name=csv_file,drop_compounds=drop_compounds,test_size=0.0,correlated_groups=correlated_groups,standarize=False,train_suffix="_all.csv")
        #prepare_graph_data_to_ML(json_file=json_file,csv_file_name=csv_file,correlated_groups=correlated_groups,test_suffix="",train_suffix="_all.csv",prepare_test_set=False) 

        atom_data=pd.read_csv(data_file,low_memory=True)
        atom_data.dropna(axis=0)
        atom_data.dropna()
        eq_data=pd.read_csv(csv_file,low_memory=True)
        eq_data.dropna(axis=0)
        eq_data.dropna()
        #data=pd.merge(eq_data,atom_data,on="compn")
        data=atom_data
        for d in drop_compounds:    data =data[data["compn"].str.startswith(d)==False]
        #test_data=pd.read_csv(test_data_file,low_memory=True)

        composed_regressor_params={
            "linear_attributes":linear_features,
            "non_linear_attributes":non_linear_features,
            "l_n_jobs":14, "nl_n_jobs":14, "dr_n_jobs":14,  #for model paralelism instead of CV paralelism
            "combination":"sum",
            "linear_regressor":"HuberRegressor",
            "dimensionality_reduction":"None",
            "nl_inner_n_estimators":100,
            #"non_linear_regressor":"XGBRegressor",
            "non_linear_regressor":"BaggingRegressor-XGB",
            "l_ramdom_state": 42, "dr_random_state":42, "nl_random_state": 42,
             }

        for algorithm in ["shap","permutation_importances"]:
            text=""
            feat_importances=feature_importances(composed_regressor_params,data, algorithm=algorithm)
            
            aggregated_feature_importances=[]
            for cg in correlated_groups["eq_features_groups_all"]+correlated_groups["atomic_features_groups_alpha"]+correlated_groups["atomic_features_groups_beta"]+correlated_groups["relative_H"]:

                agg_feat_importance=0
                for cgg in cg:
                    agg_feat_importance+=feat_importances[cgg]
                aggregated_feature_importances.append([cg,agg_feat_importance])

            aggregated_feature_importances.sort(key=lambda x: x[1],reverse=True)

            for a in aggregated_feature_importances: 
                dot="."
                if len(str(a[0]))<80:
                    text+= f"{str(a[0]).strip('[').strip(']'):{dot}<80} {str(a[1]):.9}"
                    text+="\n"
                else:
                    if  len(str(a[0][:3]))<80: 
                        text+= f"{str([aa for aa in a[0][:3]]).strip('[').strip(']'):<80}"
                        text+="\n" 
                        if len(str(a[0][4:]))<80: 
                            text+=  f"{str([aa for aa in a[0][4:]]).strip('[').strip(']'):{dot}<80}{str(a[1]):.9}"
                            text+="\n" 
                        
                        else:
                            text+= f"{str([aa for aa in a[0][4:6]]).strip('[').strip(']'):<80}\n" 
                            text+="\n"
                            text+= f"{str([aa for aa in a[0][6:]]).strip('[').strip(']'):{dot}<80} {str(a[1]):.9}"
                            text+="\n" 
                    else:
                        text+=  f"{str([aa for aa in a[0][:2]]).strip('[').strip(']'):<80}"
                        text+="\n" 

                        if len(str(a[0][2:]))<80: 
                            text+=  f"{str([aa for aa in a[0][2:]]).strip('[').strip(']'):{dot}<80}{str(a[1]):.9}"
                            text+="\n" 
                        else:
                            text+=  f"{str([aa for aa in a[0][2:4]]).strip('[').strip(']'):<80}" 
                            text+="\n"
                            if len(str(a[0][4:]))<80: 
                                text+=  f"{str([aa for aa in a[0][4:]]).strip('[').strip(']'):{dot}<80} {str(a[1]):.9}" 
                                text+="\n"
                            else:
                                text+=  f"{str([aa for aa in a[0][4:6]]).strip('[').strip(']'):<80}"
                                text+="\n" 
                                text+=  f"{str([aa for aa in a[0][6:]]).strip('[').strip(']'):{dot}<80} {str(a[1]):.9}"
                                text+="\n" 
                text+= "\n" 
                with open(algorithm+"_results.txt","w") as f: f.write(text)       



