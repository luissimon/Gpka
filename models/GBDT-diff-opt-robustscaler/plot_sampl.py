#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go
import plotly
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import copy


import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)

from correlated_groups import correlated_groups
from drop_compounds import drop_compounds
from routes import extracted_data_route
from routes import sampl_extracted_data_route

from prepare_data import prepare_eq_data
from prepare_data import prepare_graph_data_to_ML

from composed_regressor import composed_regressor

drop_compounds=["sm11-21_cation->neut","sm11-23_cation->neut"]

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


def score_model(composed_regressor_params,train_data,test_data,file_name="composed_model"): 

    #CV parallelism or model parallelism?
    if composed_regressor_params["l_n_jobs"]!=None or composed_regressor_params["l_n_jobs"]!=1:
        n_jobs=1
    else: n_jobs=14

    new_regressor=composed_regressor( **composed_regressor_params )
    new_regressor.fit(X_train=train_data,y_train=train_data["pKa"])

    for d in drop_compounds:    train_data =train_data[train_data["compn"].str.startswith(d)==False]


    test_data["deprotonated charge"]=test_data["protonated charge"]-1
    predicted_pka=new_regressor.predict(test_data)
    test_data["predicted pKa"]=predicted_pka

    multiprotic_pairs=[["sm11-23_cation->neut","sm11-23_neut->an"]]
    multiprotic_pairs=[]
    m_brackets=[]

    for multiprotic_pair in multiprotic_pairs:
                
                i,j=test_data.index[test_data["compn"]==multiprotic_pair[0]].tolist()[0],test_data.index[test_data["compn"]==multiprotic_pair[1]].tolist()[0]
                a,aa=test_data.iloc[i],test_data.iloc[j]
                m_brackets.append({"x":[a["pKa"],aa["pKa"]],"y":[a["predicted pKa"],aa["predicted pKa"]],"names":[a["compn"],aa["compn"]]}) 
                new_name=a["compn"].split("_")[0]+a["compn"].split("_")[1].split("->")[0]+"->"+aa["compn"].split("_")[1].split("->")[1]
                #change value of first equilibrium in pair...
                test_data.loc[i,"pKa"]=np.average([a["pKa"],aa["pKa"]])
                test_data.loc[i,"compn"]=new_name
                test_data.loc[i,"protonated charge"]=a["protonated charge"]
                test_data.loc[i,"deprotonated charge"]=aa["deprotonated charge"]
                test_data.loc[i,"predicted pKa"]=np.average([a["predicted pKa"],aa["predicted pKa"]])
                #and remove the other
                test_data=test_data[test_data["compn"]!=multiprotic_pair[1]] 


    test_data_sm6=test_data[test_data["compn"].str.startswith("sm6")==True]
    test_data_sm7=test_data[test_data["compn"].str.startswith("sm7")==True]
    test_data_sm8=test_data[test_data["compn"].str.startswith("sm8")==True]
    test_data_sm11=test_data[test_data["compn"].str.startswith("sm11")==True]


    brackets=[]
    for m_bracket in m_brackets:
        brackets.append(go.Scatter(x=m_bracket["x"],y=m_bracket["y"],text=m_bracket["names"],mode="lines+markers",
                        marker_symbol="x-thin",marker_line_width=2,marker_size=8,marker_color="black",
                        line=dict(color='black', width=1,dash='dash')
                        ))


    font=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000)
    fontR=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="red")
    fontG=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="green")
    fontB=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="blue")
    fontO=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="orange")
    fonts=[fontR,fontG,fontB,fontO]
    colors=["red","green","blue","orange"]  

    texts=[[n+" ("+"%+d" %c+" -> "+"%+d" %(c_d)+")" for n,c,c_d in zip(test_data['compn'],test_data['protonated charge'],test_data['deprotonated charge'])] for test_data in [test_data_sm6,test_data_sm7,test_data_sm8,test_data_sm11]]
    pka_traces=[]
    predicted_pkas=[test_data_sm6["predicted pKa"],test_data_sm7["predicted pKa"],test_data_sm8["predicted pKa"],test_data_sm11["predicted pKa"]]
    experimental_pkas=[test_data_sm6["pKa"],test_data_sm7["pKa"],test_data_sm8["pKa"],test_data_sm11["pKa"]]
    for p_pka,exp_pka,color,text in zip(predicted_pkas,experimental_pkas,colors,texts): 
        pka_traces.append(go.Scatter(x=exp_pka,y=p_pka,mode='markers', showlegend=False,
                            marker={"color":color,"size":8,
                                    #"colorscale":'Rainbow',"cmin":-3,"cmax":3,                                  
                                    "line":{"width":1.0},"showscale":True,
                                    },    
                            text=text )
                            )

    
    line_trace=go.Scatter(y=[0, 14],x=[0,14],mode="lines",showlegend=False,line=dict(color='black', width=1,dash='dash'))
    fill_05=go.Scatter(y=[-8.5,-9.5,18.5,19.5],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.3)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")
    fill_1=go.Scatter(y=[-8.0,-10.0,18.0,20.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.2)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")
    fill_2=go.Scatter(y=[-7.0,-11.0,17.0,21.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.1)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")


    
    from plotly.subplots import make_subplots
    fig1 = make_subplots( rows=1,cols=1,#specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        #subplot_titles=["mue is:"+str(mean_unsigned_error),"residuals","errors"],
                        subplot_titles=["model"]#,"residuals","errors"],
                        #row_heights=[0.6,0.35,0.05],
                        #vertical_spacing=0.03,horizontal_spacing=0.05
                        )
    fig1.update_annotations(font=font)
    #fig1.update_layout(legend=dict(font=dict(size=18),yanchor="top",xanchor="right",y=0.2,x=0.2))


    for bracket in brackets: fig1.add_trace(bracket,row=1,col=1)
    for pka_trace in pka_traces: fig1.add_trace(pka_trace,row=1,col=1)
    fig1.add_trace(line_trace,row=1,col=1)
    fig1.add_trace(fill_05,row=1,col=1)
    fig1.add_trace(fill_1,row=1,col=1)
    fig1.add_trace(fill_2,row=1,col=1)

    fig1.update_layout(height=800,yaxis_range=[0,14],xaxis_range=[0,14])

    fig1.update_xaxes(title_text="pKa",row=1,col=1,title_font={'size': 22, 'weight': 1000},tickfont={"size":16})
    fig1.update_yaxes(title_text="pKa pred.",row=1,col=1,title_font={'size': 22, 'weight': 1000},tickfont={"size":16})

    mean_absolute_errors=[np.mean(abs(predicted_pka-experimental_pka))   for predicted_pka,experimental_pka in zip(predicted_pkas,experimental_pkas)]
    neg_mean_squared_errors=[np.mean((predicted_pka-experimental_pka)**2)**0.5 for predicted_pka,experimental_pka in zip(predicted_pkas,experimental_pkas)]
    from sklearn.metrics import r2_score
    r_2_scores=[r2_score(predicted_pka,experimental_pka) for predicted_pka,experimental_pka in zip(predicted_pkas,experimental_pkas)]

    x_position=0.55
    fig1.add_annotation(x=x_position,y=0.4,xref="paper", yref="paper",text="  M.U.E.: ",font = font, showarrow=False )
    fig1.add_annotation(x=x_position,y=0.33,xref="paper", yref="paper",text="  R.M.S.E.: ",font = font, showarrow=False )
    fig1.add_annotation(x=x_position,y=0.285,xref="paper", yref="paper",text="  r\u00b2: ",font = font, showarrow=False )
    fig1.add_annotation(x=0.1,y=0.9,xref="paper", yref="paper",text="  SAMPL6 ",font=fontR,showarrow=False)
    fig1.add_annotation(x=0.1,y=0.83,xref="paper", yref="paper",text="  SAMPL7 ",font=fontG,showarrow=False)
    fig1.add_annotation(x=0.1,y=0.76,xref="paper", yref="paper",text="  SAMPL8 ",font=fontB,showarrow=False)
    fig1.add_annotation(x=0.1,y=0.70,xref="paper", yref="paper",text="  EuroSAMPL11 ",font=fontO,showarrow=False)

    x_positions=[0.64,0.75,0.83,0.91]
    for mae,rmse,r2,f,x_position in zip(mean_absolute_errors,neg_mean_squared_errors,r_2_scores,fonts,x_positions):
        #x_position+=0.10
        fig1.add_annotation(x=x_position,y=0.4,xref="paper", yref="paper",text="{:.3f}".format(mae),font = f, showarrow=False )
        fig1.add_annotation(x=x_position,y=0.33,xref="paper", yref="paper",text="{:.3f}".format(rmse),font = f, showarrow=False )
        fig1.add_annotation(x=x_position,y=0.285,xref="paper", yref="paper",text="{:.3f}".format(r2),font = f, showarrow=False )        

    
    #fig1.update_layout(legend=dict(y=0.35,x=0.9))



    fig1.write_html(file_name+".html")

    fig1.write_image(file_name+".png", width=1200, height=900,scale=4)



if __name__=="__main__":

    levels_of_theory=["pbeh3c","swb97xd","wb97xd","M06","sM06"]
    levels_of_theory=["swb97xd"]
    features_included="difference"
    if features_included=="all": non_linear_features=list(set(non_linear_features+non_linear_features2+non_linear_features3)) #uncomment to include all features, also protonated and deprotonated
    elif features_included=="protonated": non_linear_features= non_linear_features2
    elif features_included=="deprotonated": non_linear_features=non_linear_features3
    elif features_included=="protonated+deprotonated": non_linear_features=list(set(non_linear_features2+non_linear_features3))
    elif features_included=="difference": non_linear_features=non_linear_features


    for lot in levels_of_theory:
        #name of files:
        train_csv_file=extracted_data_route+"values_extracted-gibbs-"+lot+".25.csv" 
        train_json_file=extracted_data_route+"/molecular_graphs-gibbs-"+lot+".25.json"

        test_csv_file=sampl_extracted_data_route+"values_extracted_sampl-gibbs-"+lot+".25.csv" 
        test_json_file=sampl_extracted_data_route+"/molecular_graphs-sampl-gibbs-"+lot+".25.json"

        train_data_file=train_csv_file[:-4]+"_all_with_graph_data.csv"
        train_data_file=train_data_file.split("/")[-1]
        test_data_file=test_csv_file[:-4]+"_all_with_graph_data.csv"
        test_data_file=test_data_file.split("/")[-1]

        """
        prepare_eq_data(file_name=test_csv_file,drop_compounds=drop_compounds,
                        test_size=0.0,correlated_groups=correlated_groups,train_suffix="_all.csv",
                        standarize=False,use_standard_scalers="e_standard_scalers.txt")

        prepare_graph_data_to_ML(json_file=test_json_file,csv_file_name=test_csv_file,correlated_groups=correlated_groups,
                                test_suffix="",train_suffix="_all.csv",prepare_test_set=False,
                                standarize=True,use_standard_scalers="g_standard_scalers.txt") 
        """
        
        train_atom_data=pd.read_csv(train_data_file,low_memory=True)
        train_atom_data.dropna(axis=0)
        train_atom_data.dropna()
        #train_eq_data=pd.read_csv(train_csv_file,low_memory=True)
        #train_eq_data.dropna(axis=0)
        #train_eq_data.dropna()
        #train_data=pd.merge(train_eq_data,train_atom_data,on="compn")
        train_data=train_atom_data
        for d in drop_compounds:    traindata =train_data[train_data["compn"].str.startswith(d)==False]
        #test_data=pd.read_csv(test_data_file,low_memory=True)

        test_atom_data=pd.read_csv(test_data_file,low_memory=True)
        test_atom_data.dropna(axis=0)
        test_atom_data.dropna()
        #test_eq_data=pd.read_csv(test_csv_file,low_memory=True)
        #test_eq_data.dropna(axis=0)
        #test_eq_data.dropna()
        #test_data=pd.merge(test_eq_data,test_atom_data,on="compn")
        test_data= test_atom_data

        composed_regressor_params={
            "linear_attributes":linear_features,
            "non_linear_attributes":non_linear_features,
            "l_n_jobs":14, "nl_n_jobs":14, "dr_n_jobs":14,  #for model paralelism instead of CV paralelism
            "combination":"sum",
            "linear_regressor":"HuberRegressor",
            "dimensionality_reduction":"None",

'lxalpha': 0.39563931136636404, 'lxl1_ratio': 0.6431492672600965, 'nl_bootstrap': False, 'nl_max_features': 0.7580306924159148, 'nl_booster': 'gbtree', 'nl_max_depth': 6, 'nl_subsample': 0.7163453678772236, 'nl_gamma': 0.00912479024687513, 
'nl_reg_lambda': 1.24326911968816, 'nl_reg_alpha': 0.5302720761832531, 'nl_tree_method': 'approx', 'nl_refresh_leaf': True, 'nl_max_bin': 130, 'nl_eta': 0.26396140499039084, 'nl_inner_n_estimators': 154, 'nl_n_estimators': 12,


            "non_linear_regressor":"BaggingRegressor-XGB",
            "l_ramdom_state": 42, "dr_random_state":42, "nl_random_state": 42,
        }

        score_model(composed_regressor_params,train_data,test_data,file_name="sam"+features_included+"_"+lot)



