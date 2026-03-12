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


#include in linear features as long as I obtain it'lf_SMD energy difference':[True,False]
linear_features={"lf_energy":['deltaZPE','deltaE','deltaG'],
                #"lf_SMD-solv":['SMD-solv'],
                #"lf_explicit_water":['expl1wat'],        
                }

non_linear_features={   
                        "nlf_SMD-solv":['SMD-solv'],
                        "nlf_explicit_water":['expl1wat'],
                        "nlf_protonated charge":['protonated charge'],
                        "nlf_RDG%HB": ['RDG%HB'],
                        "nlf_RDG%VdW": ['RDG%VdW'],
                        "nlf_RDG%st": ['RDG%st'],
                        "nlf_prom-RDG%HB": ['prom-RDG%HB'],
                        "nlf_prom-RDG%VdW": ['prom-RDG%VdW'],
                        "nlf_prom-RDG%st": ['prom-RDG%st'],
                        "nlf_*mu": ['*mu'],
                        "nlf_tr-e*theta": ['tr-e*theta'],
                        "nlf_tr-*alpha": ['tr-*alpha'],
                        "nlf_HLgap": ['HLgap'],
                        "nlf_Vol": ['Vol'],
                        "nlf_Surf": ['Surf'],
                        "nlf_Surf+": ['Surf+', 'Surf-'],
                        "nlf_min-ESP": ['min-ESP'],
                        "nlf_max-ESP": ['max-ESP'],
                        "nlf_avg-ESP": ['avg-ESP', 'avg-ALIE'],
                        "nlf_avg-ESP+": ['avg-ESP+'],
                        "nlf_avg-ESP-": ['avg-ESP-'],
                        "nlf_var-ESP": ['var-ESP'],
                        "nlf_var-ESP+": ['var-ESP+'],
                        "nlf_var-ESP-": ['var-ESP-'],
                        "nlf_*PI-ESP": ['*PI-ESP'],
                        "nlf_MPI": ['MPI'],
                        "nlf_min-LEA": ['min-LEA'],
                        "nlf_max-LEA": ['max-LEA'],
                        "nlf_avg-LEA": ['avg-LEA'],
                        "nlf_var-LEA": ['var-LEA'],
                        "nlf_min-ALIE": ['min-ALIE'],
                        "nlf_max-ALIE": ['max-ALIE'],
                        "nlf_var-ALIE": ['var-ALIE'],
                        "nlf_avg|EF|": ['avg|EF|', '0.75q|EF|', '0.5q|EF|'],
                        "nlf_avgEF*tang": ['avgEF*tang', '0.75qEF*tang', '0.5qEF*tang'],
                        "nlf_avgEF*norm": ['avgEF*norm', '0.75qEF*norm', '0.5qEF*norm'],
                        "nlf_0.95q|EF|": ['0.95q|EF|', '0.9q|EF|'],
                        "nlf_0.95qEF*tang": ['0.95qEF*tang', '0.9qEF*tang'],
                        "nlf_0.95qEF*norm": ['0.95qEF*norm', '0.9qEF*norm'],
                        "nlf_avgEF*angle": ['avgEF*angle', '0.75qEF*angle', '0.5qEF*angle'],
                        "nlf_0.95qEF*angle": ['0.95qEF*angle', '0.9qEF*angle'],

                        "nlf_Hirshfeld_alpha":['Hirshfeld_alpha', 'Voronoy_alpha', 'Lowdin_alpha', 'CM5_alpha', '12CM5_alpha'],
                        "nlf_Mulliken_alpha":['Mulliken_alpha'],
                        "nlf_Becke_alpha":['Becke_alpha'],
                        "nlf_ADCH_alpha":['ADCH_alpha'],
                        "nlf_CHELPG_alpha":['CHELPG_alpha', 'MK_alpha', 'RESP_alpha'],
                        "nlf_PEOE_alpha":['PEOE_alpha'],
                        "nlf_NBO-chg_alpha":['NBO-chg_alpha'],
                        "nlf_(a)Surf_alpha":['(a)Surf_alpha'],
                        "nlf_(a)Surf-_alpha":['(a)Surf-_alpha', '(a)Surf+_alpha'],
                        "nlf_(a)avg-ESP_alpha":['(a)avg-ESP_alpha', '(a)min-ESP_alpha', '(a)max-ESP_alpha'],
                        "nlf_(a)avg-ESP+_alpha":['(a)avg-ESP+_alpha'],
                        "nlf_(a)avg-ESP-_alpha":['(a)avg-ESP-_alpha'],
                        "nlf_(a)var-ESP_alpha":['(a)var-ESP_alpha'],
                        "nlf_(a)var-ESP+_alpha":['(a)var-ESP+_alpha'],
                        "nlf_(a)var-ESP-_alpha":['(a)var-ESP-_alpha'],
                        "nlf_(a)*PI-ESP_alpha":['(a)*PI-ESP_alpha'],
                        "nlf_ESP-nucl_alpha":['ESP-nucl_alpha'],
                        "nlf_NMR*delta_alpha":['NMR*delta_alpha'],
                        "nlf_(a)*mu_alpha":['(a)*mu_alpha'],
                        "nlf_(a)*mu-ctb_alpha":['(a)*mu-ctb_alpha'],
                        "nlf_(a)tr-e*theta_alpha":['(a)tr-e*theta_alpha'],
                        "nlf_(a)avg-ALIE_alpha":['(a)avg-ALIE_alpha', '(a)max-ALIE_alpha', '(a)min-ALIE_alpha', '(a)avg-LEA_alpha', '(a)max-LEA_alpha', '(a)min-LEA_alpha'],
                        "nlf_(a)var-ALIE_alpha":['(a)var-ALIE_alpha'],
                        "nlf_(a)var-LEA_alpha":['(a)var-LEA_alpha'],

                        "nlf_Hirshfeld_beta":['Hirshfeld_beta', 'Voronoy_beta', 'Lowdin_beta', 'CM5_beta', '12CM5_beta'],
                        "nlf_Mulliken_beta":['Mulliken_beta'],
                        "nlf_Becke_beta":['Becke_beta'],
                        "nlf_ADCH_beta":['ADCH_beta'],
                        "nlf_CHELPG_beta":['CHELPG_beta', 'MK_beta', 'RESP_beta'],
                        "nlf_PEOE_beta":['PEOE_beta'],
                        "nlf_NBO-chg_beta":['NBO-chg_beta'],
                        "nlf_(a)Surf_beta":['(a)Surf_beta'],
                        "nlf_(a)Surf-_beta":['(a)Surf-_beta', '(a)Surf+_beta'],
                        "nlf_(a)avg-ESP_beta":['(a)avg-ESP_beta', '(a)min-ESP_beta', '(a)max-ESP_beta'],
                        "nlf_(a)avg-ESP+_beta":['(a)avg-ESP+_beta'],
                        "nlf_(a)avg-ESP-_beta":['(a)avg-ESP-_beta'],
                        "nlf_(a)var-ESP_beta":['(a)var-ESP_beta'],
                        "nlf_(a)var-ESP+_beta":['(a)var-ESP+_beta'],
                        "nlf_(a)var-ESP-_beta":['(a)var-ESP-_beta'],
                        "nlf_(a)*PI-ESP_beta":['(a)*PI-ESP_beta'],
                        "nlf_ESP-nucl_beta":['ESP-nucl_beta'],
                        "nlf_NMR*delta_beta":['NMR*delta_beta'],
                        "nlf_(a)*mu_beta":['(a)*mu_beta'],
                        "nlf_(a)*mu-ctb_beta":['(a)*mu-ctb_beta'],
                        "nlf_(a)tr-e*theta_beta":['(a)tr-e*theta_beta'],
                        "nlf_(a)avg-ALIE_beta":['(a)avg-ALIE_beta', '(a)max-ALIE_beta', '(a)min-ALIE_beta', '(a)avg-LEA_beta', '(a)max-LEA_beta', '(a)min-LEA_beta'],
                        "nlf_(a)var-ALIE_beta":['(a)var-ALIE_beta'],
                        "nlf_(a)var-LEA_beta":['(a)var-LEA_beta'],


                        "nlf_PEOE*relative*H": ['PEOE*relative*H'],
                        "nlf_Mulliken-*H": ['Mulliken-*H'],
                        "nlf_Mulliken*relative*H": ['Mulliken*relative*H'],
                        "nlf_Hirshfeld*relative*H": ['Hirshfeld*relative*H', 'Voronoy*relative*H', 'Lowdin*relative*H', 'CM5*relative*H', '12CM5*relative*H'],
                        "nlf_ADCH*relative*H": ['ADCH*relative*H'],
                        "nlf_CHELPG*relative*H": ['CHELPG*relative*H', 'MK*relative*H', 'RESP*relative*H'],
                        "nlf_NBO-chg*relative*H": ['NBO-chg*relative*H'],
                        "nlf_NMR*delta*relative*H": ['NMR*delta*relative*H'],
                        "nlf_ESP-nucl*relative*H": ['ESP-nucl*relative*H'],
                        "nlf_Mayer-BO*relative*H": ['Mayer-BO*relative*H'],
                        "nlf_WBO*relative*H": ['WBO*relative*H', 'WBO-NAO*relative*H', 'NLMO-BO*relative*H'],
                        "nlf_Mulliken-BO*relative*H": ['Mulliken-BO*relative*H'],
                        "nlf_FBO*relative*H": ['FBO*relative*H'],
                        "nlf_LBO*relative*H": ['LBO*relative*H'],
                        "nlf_IBSI*relative*H": ['IBSI*relative*H'],
                        "nlf_FUERZA-FC*relative*H": ['FUERZA-FC*relative*H'],
                        "nlf_BD*relative*H": ['BD*relative*H'],
                        "nlf_*mu*BP-*H": ['*mu*BP-*H'],
                        "nlf_*ind*mu*BP-*H": ['*ind*mu*BP-*H'],
                        "nlf_diag-e*theta*BP-*H": ['diag-e*theta*BP-*H'],
                        "nlf_diag-*theta*BP-*H": ['diag-*theta*BP-*H'],
                        }

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

non_linear_features2=['protonated SMD-solv', 'protonated expl1wat', 'protonated charge', 
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

non_linear_features3=['deprotonated SMD-solv', 'deprotonated expl1wat', 'protonated charge', 
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


def compare_model_performance(composed_regressor_params,train_data,test_data1,test_data2,file_name="composed_model"): 



    new_regressor=composed_regressor( **composed_regressor_params )
    new_regressor.fit(train_data,train_data["pKa"])

    test1_pka_predictions=new_regressor.predict(test_data1)
    test2_pka_predictions=new_regressor.predict(test_data2)


    font=plotly.graph_objects.layout.annotation.Font(size=36,weight=1000)
    fontG=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="green")
    fontR=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="red")

    text=[n+" ("+"%+d" %c+" -> "+"%+d" %(c-1)+")" for n,c in zip(test_data1['correct name'],test_data1['protonated charge'])]
    compare_predictions_plot_trace= go.Scatter(x=test1_pka_predictions, y=test2_pka_predictions,
                                                text=text, 
                                                mode='markers', showlegend=False,
                                                marker={"color":test_data1['protonated charge'],"size":8,
                                                        "colorscale":'Bluered',"cmin":-2,"cmax":3,                                  
                                                        "line":{"width":0.5},"showscale":True,
                                                        "colorbar":{"y":0.80,"x":0.15, "orientation":"h",
                                                            "title":{"text": "charge of acid","side":"top",
                                                            "font":{"size":42,"weight":500}},
                                                        "tickvals":[-3,-2,-1,0,1,2,3],
                                                        "tickfont":{"size":32},
                                                        "thickness":18,"len":0.25,}},    
                                                )

    std_predictions_plot_trace= go.Scatter(x=test_data1["pKa"], y=test1_pka_predictions,
                                            text=text,
                                                mode='markers', showlegend=False,
                                                marker={"color":test_data1['protonated charge'],"size":8,
                                                        "colorscale":'Bluered',"cmin":-2,"cmax":3,                                  
                                                        "line":{"width":0.5},"showscale":True,
                                                        "colorbar":{"y":0.80,"x":0.15, "orientation":"h",
                                                            "title":{"text": "charge of acid","side":"top",
                                                            "font":{"size":42,"weight":500}},
                                                        "tickvals":[-3,-2,-1,0,1,2,3],
                                                        "tickfont":{"size":32},
                                                        "thickness":18,"len":0.25,}},    
                                                )

    singlemic_predictions_plot_trace= go.Scatter(x=test_data2["pKa"], y=test2_pka_predictions,
                                            text=text,
                                                mode='markers', showlegend=False,
                                                marker={"color":test_data1['protonated charge'],"size":8,
                                                        "colorscale":'Bluered',"cmin":-2,"cmax":3,                                  
                                                        "line":{"width":0.5},"showscale":True,
                                                        "colorbar":{"y":0.80,"x":0.15, "orientation":"h",
                                                            "title":{"text": "charge of acid","side":"top",
                                                            "font":{"size":42,"weight":500}},
                                                        "tickvals":[-3,-2,-1,0,1,2,3],
                                                        "tickfont":{"size":32},
                                                        "thickness":18,"len":0.25,}},    
                                                ) 

    line_trace=go.Scatter(y=[-5, 16],x=[-5,16],mode="lines",showlegend=False,line=dict(color='black', width=1,dash='dash'))
    fill_05=go.Scatter(y=[-8.5,-9.5,18.5,19.5],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.3)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")
    fill_1=go.Scatter(y=[-8.0,-10.0,18.0,20.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.2)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")
    fill_2=go.Scatter(y=[-7.0,-11.0,17.0,21.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.1)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")                                                

    fig1,fig2,fig3=go.Figure(data=line_trace),go.Figure(data=line_trace),go.Figure(data=line_trace)
    fig1.add_trace(fill_05),fig2.add_trace(fill_05),fig3.add_trace(fill_05)
    fig1.add_trace(fill_1),fig2.add_trace(fill_1),fig3.add_trace(fill_1)
    fig1.add_trace(fill_2),fig2.add_trace(fill_2),fig3.add_trace(fill_2)
    fig1.add_trace(compare_predictions_plot_trace),fig2.add_trace(std_predictions_plot_trace),fig3.add_trace(singlemic_predictions_plot_trace)


    fig1.update_xaxes(title_text="pKa predicted std. procedure",title_font={'size': 48, 'weight': 1000},tickfont={"size":36})
    fig1.update_yaxes(title_text="pKa predicted single tautomer",title_font={'size': 48, 'weight': 1000},tickfont={"size":36})
    fig2.update_xaxes(title_text="pKa predicted std. procedure",title_font={'size': 48, 'weight': 1000},tickfont={"size":36})
    fig2.update_yaxes(title_text="exp. pKa",title_font={'size': 2, 'weight': 1000},tickfont={"size":36})
    fig3.update_xaxes(title_text="pKa predicted single tautomer",title_font={'size': 48, 'weight': 1000},tickfont={"size":36})
    fig3.update_yaxes(title_text="exp. pKa",title_font={'size': 48, 'weight': 1000},tickfont={"size":36})

    from sklearn.metrics import r2_score
    mean_absolute_error_std=np.mean(abs(test1_pka_predictions-test_data1["pKa"]))
    root_mean_squared_error_std=np.mean((test1_pka_predictions-test_data1["pKa"])**2)**0.5
    r_2_std=r2_score(test1_pka_predictions,test_data1["pKa"])
    mean_absolute_error_single_tautomer=np.mean(abs(test2_pka_predictions-test_data2["pKa"]))
    root_mean_squared_error_single_tautomer=np.mean((test2_pka_predictions-test_data2["pKa"])**2)**0.5
    r_2_single_tautomer=r2_score(test2_pka_predictions,test_data2["pKa"])


    fig2.add_annotation(x=0.8,y=0.3,xref="paper", yref="paper",text="  M.U.E.: "+"{:.3f}".format(mean_absolute_error_std),
                        font = font, showarrow=False )
    fig2.add_annotation(x=0.8,y=0.25,xref="paper", yref="paper",text="R.M.S.E.: "+"{:.3f}".format(root_mean_squared_error_std),
                        font = font, showarrow=False )
    fig2.add_annotation(x=0.8,y=0.2,xref="paper", yref="paper",text="      r\u00b2: "+"{:.3f}".format(r_2_std),
                        font = font, showarrow=False )

    fig3.add_annotation(x=0.8,y=0.3,xref="paper", yref="paper",text="  M.U.E.: "+"{:.3f}".format(mean_absolute_error_single_tautomer),
                        font = font, showarrow=False )
    fig3.add_annotation(x=0.8,y=0.25,xref="paper", yref="paper",text="R.M.S.E.: "+"{:.3f}".format(root_mean_squared_error_single_tautomer),
                        font = font, showarrow=False )
    fig3.add_annotation(x=0.8,y=0.2,xref="paper", yref="paper",text="      r\u00b2: "+"{:.3f}".format(r_2_single_tautomer),
                        font = font, showarrow=False )

    fig1.update_layout(height=1600,width=1600,yaxis_range=[-1,15],xaxis_range=[-1,15])
    fig2.update_layout(height=1600,width=1600,yaxis_range=[-1,15],xaxis_range=[-1,15])
    fig3.update_layout(height=1600,width=1600,yaxis_range=[-1,15],xaxis_range=[-1,15])
    fig1.write_html(file_name+"_comparison.html")
    fig1.write_image(file_name+"_comparison.png", width=1600, height=1600,scale=1)
    fig2.write_html(file_name+"_standard.html")
    fig2.write_image(file_name+"_standard.png", width=1600, height=1600,scale=1)
    fig3.write_html(file_name+"_single_tautomer.html")
    fig3.write_image(file_name+"_single_tautomer.png", width=1600, height=1600,scale=1)


if __name__=="__main__":

    #levels_of_theory=["swb97xd","pbeh3c","wb97xd","M06","sM06"]
    levels_of_theory=["swb97xd"]
    #features_included="protonated+deprotonated"
    features_included=sys.argv[1]    
    if features_included=="all": non_linear_features=list(set(non_linear_features+non_linear_features2+non_linear_features3)) #uncomment to include all features, also protonated and deprotonated
    elif features_included=="protonated": non_linear_features= non_linear_features2
    elif features_included=="deprotonated": non_linear_features=non_linear_features3
    elif features_included=="protonated+deprotonated": non_linear_features=list(set(non_linear_features2+non_linear_features3))
    elif features_included=="difference": non_linear_features=non_linear_features


    for lot in levels_of_theory:
        #name of files:
        csv_file=extracted_data_route+"values_extracted-gibbs-"+lot+".25.csv" 
        json_file=extracted_data_route+"/molecular_graphs-gibbs-"+lot+".25.json"

        #prepare the data; this is time consuming, so if it has already been prepared, comment out the instructions and read "several_tautomers" from file
        #prepare_eq_data(file_name=csv_file,drop_compounds=drop_compounds,test_size=0.0,correlated_groups=correlated_groups,standarize=False,train_suffix="_all.csv")
        #d=prepare_graph_data_to_ML(json_file=json_file,csv_file_name=csv_file,correlated_groups=correlated_groups,test_suffix="",train_suffix="_all.csv",prepare_test_set=False, use_standard_scalers="g_standard_scalers.txt",
        #                        weighting="only_most_populated_microequilibrium",to_return=["names of equilibriums with several tautomers"],output_file_suffix="_1mic_with_graph_data") 
        #d=prepare_graph_data_to_ML(json_file=json_file,csv_file_name=csv_file,correlated_groups=correlated_groups,test_suffix="",train_suffix="_all.csv",prepare_test_set=False, use_standard_scalers="g_standard_scalers.txt",
        #                        weighting="auto",to_return=["names of equilibriums with several tautomers"],output_file_suffix="_with_graph_data") 
        #d=prepare_graph_data_to_ML(json_file=json_file,csv_file_name=csv_file,correlated_groups=correlated_groups,test_suffix="",train_suffix="_all.csv",prepare_test_set=False, use_standard_scalers="g_standard_scalers.txt",
        #                        weighting="never",to_return=["names of equilibriums with several tautomers"],output_file_suffix="_noweight_with_graph_data") 

        #several_tautomers=d["names of equilibriums with several tautomers"]

        from several_tautomers import several_tautomers

        #default params
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
        
        
        #optimized params for difference features
        composed_regressor_params_diff={
            "linear_attributes":linear_features,
            "non_linear_attributes":non_linear_features,
            "l_n_jobs":14, "nl_n_jobs":14, "dr_n_jobs":14,  #for model paralelism instead of CV paralelism
            "combination":"sum",
            "linear_regressor":"HuberRegressor",
            "dimensionality_reduction":"None",


            'lxalpha': 1.4927667722761333, 
            'lxl1_ratio': 0.7435036507881647, 
            'nl_bootstrap': False, 
            'nl_max_features': 0.7503990446667513, 
            'nl_booster': 'gbtree', 
            'nl_max_depth': 6, 
            'nl_subsample': 0.6559357845904604, 
            'nl_gamma': 0.011703594068894641, 
            'nl_reg_lambda': 0.7730862185886191, 
            'nl_reg_alpha': 0.36974300613319844, 
            'nl_tree_method': 'auto', 
            'nl_refresh_leaf': True, 
            'nl_max_bin': 135, 
            'nl_eta': 0.21274963957992102, 
            'nl_inner_n_estimators': 144, 
            'nl_n_estimators': 16, 


            "non_linear_regressor":"BaggingRegressor-XGB",
            "l_ramdom_state": 42, "dr_random_state":42, "nl_random_state": 42,
        }

        #optimized params for difference features
        composed_regressor_params_all={
            "linear_attributes":linear_features,
            "non_linear_attributes":non_linear_features,
            "l_n_jobs":14, "nl_n_jobs":14, "dr_n_jobs":14,  #for model paralelism instead of CV paralelism
            "combination":"sum",
            "linear_regressor":"HuberRegressor",
            "dimensionality_reduction":"None",

            #optimized parameters
            'lxalpha': 0.2568484324204346, 
            'lxl1_ratio': 1.7103982517155383, 
            'nl_bootstrap': False, 
            'nl_max_features': 0.5544257839350846, 
            'nl_booster': 'gbtree', 
            'nl_max_depth': 6, 
            'nl_subsample': 0.5609777841019409, 
            'nl_gamma': 0.0036326761011042418, 
            'nl_reg_lambda': 1.06421799794472, 
            'nl_reg_alpha': 1.5334098326457903, 
            'nl_tree_method': 'hist', 
            'nl_refresh_leaf': False, 
            'nl_max_bin': 206, 
            'nl_eta': 0.2478014105328064, 
            'nl_inner_n_estimators': 170, 
            'nl_n_estimators': 14,

            "non_linear_regressor":"BaggingRegressor-XGB",
            "l_ramdom_state": 42, "dr_random_state":42, "nl_random_state": 42,
        }
    
        if features_included=="all": composed_regressor_params=composed_regressor_params_all
        elif features_included=="diff": composed_regressor_params=composed_regressor_params_diff

        data_file=csv_file[:-4]+"_all_with_graph_data.csv"
        data_file=data_file.split("/")[-1] #remove route and work on local file
        data_1mic_file=csv_file[:-4]+"_all_1mic_with_graph_data.csv"
        data_1mic_file=data_1mic_file.split("/")[-1]

        atom_data=pd.read_csv(data_file,low_memory=True)
        atom_data.dropna(axis=0)
        atom_data.dropna()

        atom_data_1mic=pd.read_csv(data_1mic_file,low_memory=True)
        atom_data_1mic.dropna(axis=0)
        atom_data_1mic.dropna()

        for d in drop_compounds:    
            atom_data =atom_data[atom_data["compn"].str.startswith(d)==False]
            atom_data_1mic =atom_data_1mic[atom_data_1mic["compn"].str.startswith(d)==False]

        test_set_indexes=[int(np.where(atom_data["compn"]==d)[0]) for d in several_tautomers]
        train_set_indexes=[i for i in range(len(atom_data)) if i not in test_set_indexes]
        test_data_std=atom_data.iloc[test_set_indexes]
        test_data_1mic=atom_data_1mic.iloc[test_set_indexes]
        train_data=atom_data.iloc[train_set_indexes]


        compare_model_performance(composed_regressor_params,train_data=train_data,
                                    test_data1=test_data_std,
                                    test_data2=test_data_1mic,
                                    file_name="compare_tautomers_treatment_"+features_included+"_"+lot)






