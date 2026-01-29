#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import time
import sys
import copy
import json
import joblib 


import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
#sys.path.insert(0,"../../Gpka/scripts/import")
from correlated_groups import correlated_groups
from drop_compounds import drop_compounds
from prepare_data import prepare_eq_data
from prepare_data import prepare_graph_data_to_ML

from flock_of_finches import flock_of_finches
from composed_regressor import composed_regressor

from datetime import date, datetime, timedelta
from os.path import getmtime


import NpEncoder

#required to jsonize numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.string_) or isinstance(obj,np.str_):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, (pd.Series,pd.DataFrame )):
            return obj.to_json()
        return super(NpEncoder, self).default(obj)



#import ray
#ray.init(address="212.128.132.49:6379",_redis_password="34b2317c-3adf-45d8-a31a-8b4a7fc8b459")


start_time=time.time()


#from features_diff_abs import linear_features
#from features_diff_abs import non_linear_features
#from features_diff_prot_deprot import linear_features
#from features_diff_prot_deprot import non_linear_features

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





from hyperparameters import hyperparameters
from hyperparameters import bounds




#prepare the parameters for composed_regressor from a genotype
#note that only the first element in list genotype (the fenotype) is used.
def prepare_params(genotype):
    parameters={}
    for k in genotype.keys(): parameters[k]=genotype[k][0]
    if "nl_n_estimators_inner_outer_ratio" in parameters.keys() and "nl_total_n_estimators" in parameters.keys():
        parameters["nl_inner_n_estimators"]=round( (parameters["nl_total_n_estimators"]*parameters["nl_n_estimators_inner_outer_ratio"])**0.5)
        parameters["nl_n_estimators"]=round( (parameters["nl_total_n_estimators"]/parameters["nl_n_estimators_inner_outer_ratio"])**0.5)
        for pk in ["nl_n_estimators_inner_outer_ratio","nl_total_n_estimators"]:
            if pk in parameters.keys(): parameters.pop(pk)
    parameters["linear_attributes"]=linear_features
    parameters["non_linear_attributes"]=non_linear_features#+non_linear_features2+non_linear_features3
    parameters["l_n_jobs"],parameters["nl_n_jobs"],parameters["dr_n_jobs"]=7,7,7  #for model paralelism instead of CV paralelism
    parameters["combination"]="sum"
    parameters["linear_regressor"]="HuberRegressor"
    parameters["dimensionality_reduction"]="None"
    parameters["non_linear_regressor"]="BaggingRegressor-XGB"
    parameters["l_ramdom_state"],parameters["dr_random_state"],parameters["nl_random_state"]=42,42,42

    print ("parameters:")
    print(parameters)   
    return parameters



def score_model(genotype,cv_file="cv_splitter.json",data="with_graphs.csv"): 

    import time
    imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
    sys.path.insert(0,imports_path)
    from composed_regressor import composed_regressor
    #import composed_regressor
    ctime=time.time()
    data=pd.read_csv(data,low_memory=False)
    data.drop(columns=["compn",],inplace=True)
    with open(cv_file) as f: cv_splitter = json.load(f)

    params=prepare_params(genotype)
    print (params)
    #CV parallelism or model parallelism?
    if params["l_n_jobs"]!=None or params["l_n_jobs"]!=1:
        n_jobs=1
    else: n_jobs=7
    
    new_regressor=composed_regressor( **params )
        
    scoring={"r2":"r2",
            "neg_mean_squared_error":"neg_mean_squared_error",
            "neg_mean_absolute_error":"neg_mean_absolute_error",
            }

    #compound_scores=sklearn.model_selection.cross_val_score(new_regressor,data,data["pKa"], 
    #                                    scoring=scoring, cv=7,n_jobs=n_jobs,verbose=0)
    from sklearn.model_selection import cross_validate
    print ("about to start cross_validation") 
   
    #new_regressor.fit(data,data["pKa"])#borrame
    #print ("error while fitting?")
    #print (new_regressor.predict(data))#borrame
    cross_validated_model=cross_validate(new_regressor,
                                        data,
                                        data["pKa"], 
                                        scoring=scoring, 
                                        cv=cv_splitter,
                                        n_jobs=n_jobs,
                                        verbose=0,
                                        return_estimator=False,
                                        return_train_score=True,
                                        )    

    computation_time=-1*(time.time()-ctime)#negative, so bigger is better
    print("cross validation finished after" +str(computation_time))

    #compound_scores=sklearn.cross_val_predict(new_regressor,data,data["pKa"],cv=5)
    score=[cross_validated_model["test_r2"].mean(axis=0),
            cross_validated_model["test_neg_mean_squared_error"].mean(axis=0),
            cross_validated_model["test_neg_mean_absolute_error"].mean(axis=0),
            cross_validated_model["train_neg_mean_absolute_error"].mean(axis=0),
            computation_time,
            ]
    if any(np.isnan(np.array(score))):
            score=[-1000,-1000,-1000,-1000,-1000]
    if new_regressor.fit_failure:
        for s in score[:-2]: s=-1000
        print("fit failed")
    else: 
        print ("scoring terminated after: "+str(-1*computation_time))
        print ("score: "+str(score))
    return {"score":score,"genotype":genotype}

def print_model(genotype,file_name):
    params=[prepare_params(g) for g in genotype]
    with open (file_name+".txt","w") as f: f.write(str(params))

#calculate the consensus results from models built from the list of genotypes in "genotype". 
#note that if len(genotypes) is 1, it will calculate only the performance of this model.
def model_results(finches,data_file="all_with_graphs.csv", n_procs=28,cv_file="cv_splitter.json",save_model="",save_model_parameters="",force_recalculation=False):
    
    data=pd.read_csv(data_file,low_memory=False)
    with open(cv_file) as f: cv_splitter = json.load(f)

    params=[prepare_params(f.genotype) for f in finches]
    memes=[f.memotype for f in finches]

    #CV parallelism or model parallelism?
    if params[0]["l_n_jobs"]!=None or params[0]["l_n_jobs"]!=1:
        n_procs=int(np.floor(n_procs/params[0]["l_n_jobs"]))

    n_procs=50
    print (cv_file) 
    def foo(param,meme):
        imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
        sys.path.insert(0,imports_path)
        from composed_regressor import composed_regressor
        print("inside foo")
        #a copy of the parameters but setting non_linear_regressor to "None", used in linear model evaluation
        param_only_linear=copy.deepcopy(param)
        param_only_linear["non_linear_regressor"]="None"
        regressor_only_linear= composed_regressor( **param_only_linear ) 
        regressor= composed_regressor( **param )
        
        if (force_recalculation==False and isinstance(meme,dict) and 
            "linear_pka_prediction" in meme.keys() and  "pka_prediction" in meme.keys()): 
            linear_pka_prediction,pka_prediction=meme["linear_pka_prediction"],meme["pka_prediction"]

        else:
            linear_pka_prediction=sklearn.model_selection.cross_val_predict(regressor_only_linear,data,data["pKa"],cv=cv_splitter,n_jobs=n_procs)
            print(linear_pka_prediction)
            pka_prediction=sklearn.model_selection.cross_val_predict(regressor,data,data["pKa"],cv=cv_splitter,n_jobs=n_procs) 
            print (pka_prediction)
            
            #regressor.fit(data,data["pKa"])
            #regressor_only_linear.fit(data,data["pKa"])

        #return [linear_pka_prediction,pka_prediction,regressor,regressor_only_linear]
        return [linear_pka_prediction,pka_prediction]

    jobs=[]
    import submitit
    executor = submitit.AutoExecutor(folder="slurm_temp_folder")
    executor.update_parameters(timeout_min=40,ntasks_per_node=56,name="consensus")  
    jobs=[]
    with executor.batch():
        for param,meme in zip(params,memes):
            job=executor.submit(foo,param,meme)
            jobs.append(job)
    outputs=[]
    for job in jobs:
        try:outputs.append(job.results())
        except:
            print ("Job {job} failed with error")
            continue

    #outputs = [job.results() for job in jobs]
    print (str(len(outputs))+" jobs finished")

    linear_pka_predictions,pka_predictions=[],[],
    #regressors,regressors_only_linear=[],[]
    for output,f  in zip(outputs,finches):
        linear_pka_predictions.append(np.array(output[0][0]))
        pka_predictions.append(np.array(output[0][1]))
        #regressors.append(output[0][2])
        #regressors_only_linear.append(output[0][3])
        f.memotype["linear_pka_prediction"]=np.array(linear_pka_predictions[-1])
        f.memotype["pka_prediction"]=np.array(pka_predictions[-1])
        f.memotype["residual"]=data["pKa"]-f.memotype["linear_pka_prediction"]
        f.memotype["predicted_residual"]=f.memotype["pka_prediction"]-f.memotype["linear_pka_prediction"]

    residuals=[data["pKa"]-linear_pka_pred for linear_pka_pred in linear_pka_predictions]
    predicted_residuals=[(pka_pred-linear_pka_pred) for pka_pred, linear_pka_pred in zip(pka_predictions,linear_pka_predictions)]

    #clean-up temporary files
    #clean-up temporary files
    for f in os.listdir(os.getcwd()+"/slurm_temp_folder/"):
        if f[0]!="." and time.time()-getmtime(os.getcwd()+"/slurm_temp_folder/"+f)>3600:
            os.remove(os.getcwd()+"/slurm_temp_folder/"+f) #comment for debggin

    #save parameters of evaluated models
    if save_model_parameters!="":
        for i,p in enumerate(params):
            if len(params)>1: 
                with open(save_model_parameters+'-'+str(i)+'.parameters.txt',"w") as f: json.dump(p,f,cls=NpEncoder)
            else: 
                with open(save_model_parameters+'.parameters.txt',"w") as f: json.dump(p,f,cls=NpEncoder)

    #prepare results for output
    results={}
    results["consensus_linear_pka_pred"]=np.array(linear_pka_predictions)
    results["consensus_residuals"]=np.array(residuals)
    results["consensus_pka_pred"]=np.array(pka_predictions)
    results["consensus_pred_residuals"]=np.array(predicted_residuals)
    results["linear_mean_unsigned_error"]=np.array(  [abs(r-data["pKa"]) for r in  results["consensus_linear_pka_pred"]])
    results["mean_unsigned_error"]=np.array(  [abs(r-data["pKa"]) for r in  results["consensus_pka_pred"]]) 
    results["pka"]=data["pKa"]
    results["names"]=data["compn"]

    return results     



#prepare graphics
def plot_html(consensus_linear_pka_pred,consensus_residuals,consensus_pka_pred,consensus_pred_residuals,mean_unsigned_error,mean_linear_unsigned_error,real_pka,names,file_name):

    font=go.layout.annotation.Font(size=24,weight=1000)
    fontG=go.layout.annotation.Font(size=24,weight=1000,color="green")
    fontR=go.layout.annotation.Font(size=24,weight=1000,color="red")
    pka_plot_trace=go.Scatter(x=consensus_pka_pred,y=real_pka,mode='markers', showlegend=False,
                            marker={"color":consensus_residuals,"size":4,
                                    "colorscale":'Rainbow',"cmin":-3,"cmax":3,                                  
                                    "line":{"width":1},"showscale":True,
                                    "colorbar":{"y":0.88,"x":0.15, "orientation":"h",
                                                "title":{"text": "lin. model\nerror","side":"top"},
                                                "tickvals":[-3,-2,-1,0,1,2,3],
                                                "thickness":18,"len":0.25,}},    
                            text=names )

    
    line_trace=go.Scatter(y=[-5, 16],x=[-5,16],mode="lines",showlegend=False,line=dict(color='black', width=1,dash='dash'))
    fill_05=go.Scatter(y=[-8.5,-9.5,18.5,19.5],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.3)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")
    fill_1=go.Scatter(y=[-8.0,-10.0,18.0,20.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.2)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")
    fill_2=go.Scatter(y=[-7.0,-11.0,17.0,21.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.1)',
                            line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip")


    #fig2=go.Figure()
    residuals_plot_trace=go.Scatter(x=consensus_residuals,y=consensus_pred_residuals,mode='markers', showlegend=False,
                            marker={"color":real_pka,"size":4,
                                    "colorscale":'Rainbow',"cmin":-1,"cmax":15,                                  
                                    "line":{"width":1},"showscale":True,
                                    "colorbar":{"y":.88, "x":0.6,"orientation":"h", 
                                                "title":{"text":"exp. pKa","side":"top"},
                                                "tickvals":[0,4,7,10,14],
                                                "thickness":18,"len":0.15,}},  
                            text=names) 



    #using go.Histogram
    std_dev_error_lineal=np.std(real_pka-consensus_linear_pka_pred)
    std_dev_error_all=np.std(real_pka-consensus_pka_pred)
    
    final_residuals_histogram=go.Histogram(x=(real_pka-consensus_pka_pred),opacity=0.75,xbins={"size":0.25},marker_color="red",name="final model",legendgroup=1,legendrank=2)
    linear_residuals_histogram=go.Histogram(x=(real_pka-consensus_linear_pka_pred),opacity=0.5,xbins={"size":0.25},marker_color="green",name="linear model",legendgroup=1,legendrank=1)
    rug_final_residuals=go.Scatter(x=(real_pka-consensus_pka_pred),y=[2.0]*len((real_pka-consensus_pka_pred)),mode='markers', showlegend=False,
                                    marker={"symbol":142,"color":["#D62728","#DD4477"]*len(real_pka/2)},
                                    text=names,hoverinfo="text")
    rug_linear_residuals=go.Scatter(x=(real_pka-consensus_linear_pka_pred),y=[1.0]*len((real_pka-consensus_pka_pred)),mode='markers', showlegend=False,
                                    marker={"symbol":142,"color":["#00CC96","#2CA02C"]*len(real_pka/2)},
                                    text=names,hoverinfo="text")


    from plotly.subplots import make_subplots
    fig1 = make_subplots( rows=3,cols=2,specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        #subplot_titles=["mue is:"+str(mean_unsigned_error),"residuals","errors"],
                        subplot_titles=["composed model","residuals","errors"],
                        row_heights=[0.6,0.35,0.05],
                        vertical_spacing=0.03,horizontal_spacing=0.05)
    fig1.update_annotations(font=font)
    fig1.update_layout(legend=dict(font=dict(size=18),yanchor="top",xanchor="right",y=0.2,x=0.2))


    fig1.add_trace(pka_plot_trace,row=1,col=1)
    fig1.add_trace(line_trace,row=1,col=1)
    fig1.add_trace(fill_05,row=1,col=1)
    fig1.add_trace(fill_1,row=1,col=1)
    fig1.add_trace(fill_2,row=1,col=1)
    fig1.add_trace(residuals_plot_trace,row=1,col=2)

    fig1.add_trace(final_residuals_histogram,row=2,col=2)
    fig1.add_trace(linear_residuals_histogram,row=2,col=2)
    fig1.add_trace(rug_final_residuals,row=3,col=2)
    fig1.add_trace(rug_linear_residuals,row=3,col=2)
    #fig1.add_trace(residuals_histogram,row=2,col=2)
    fig1.update_layout(height=800,yaxis_range=[-5,16],xaxis_range=[-5,16])


    fig1.update_xaxes(title_text="pKa",row=1,col=1,title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
    fig1.update_yaxes(title_text="pKa pred.",row=1,col=1,title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
    #fig1.update_xaxes(title_text="linear resicuals",row=1,col=2,title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
    fig1.update_yaxes(title_text="predicted",row=1,col=2,title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
    #fig1['layout']['xaxis1'].update(title_text="pKa")
    #fig1['layout']['yaxis1'].update(title_text="pKa pred.")
    fig1.update_xaxes(visible=False,row=3,col=2,range=[-4,5],title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
    fig1.update_yaxes(visible=False,row=3,col=2,title_font={'size': 24, 'weight': 1000})
    fig1.update_xaxes(visible=True,row=2,col=2,range=[-4,5],title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
    fig1.update_yaxes(visible=False,row=2,col=2,title_font={'size': 24, 'weight': 1000})

    fig1.add_annotation(x=0.4,y=0.35,xref="paper", yref="paper",text="M.U.E: "+"{:.3f}".format(mean_unsigned_error),
                        font = fontR, showarrow=False )
    fig1.add_annotation(x=0.4,y=0.25,xref="paper", yref="paper",text="M.U.E: "+"{:.3f}".format(mean_linear_unsigned_error),
                        font = fontG, showarrow=False )
    fig1.add_annotation(x=0.6,y=0.35,xref="paper", yref="paper",text="std.dev: "+"{:.3f}".format(std_dev_error_lineal),
                        font = fontG, showarrow=False )
    fig1.add_annotation(x=0.6,y=0.25,xref="paper", yref="paper",text="std.dev: "+"{:.3f}".format(std_dev_error_all),
                        font = fontR, showarrow=False )
    
    fig1.update_layout(legend=dict(y=0.35,x=0.9))

    fig1.write_html(file_name+".html")
    fig1.write_image(file_name+".png",width=1600, height=800,scale=4)



def report(esu,print_time=True,score_statistics_indexes=[]):
        
        text_to_report=""

        if print_time:
            text_to_report=str(len(esu.generation))+" evaluations ended at:"+str(datetime.now())+"\n---------------\n"

        #report hyperparameters histograms
        import termcharts
        from termcharts.colors import Color,default_colors

        def mytermcharts_bar(dhists):
            bars=[]
            for dhist in dhists:
                if np.sum(list(dhist.values()))!=0:
                    col=next(default_colors)
                    while col!=Color.red: col=next(default_colors)
                    b=termcharts.bar(dhist,title="",mode="h").split("\n")[4:-1:2]
                else:
                    b=["/n"]*len(dhist.values())
                for i in range(len(b)): 
                    if b[i].strip().endswith("3"):b[i]+="9m"
                    if b[i].strip().endswith("m")==False: b[i]+="m"
                bars.append(b ) 
            lines=""
            for i in range(len(bars[0])):
                line=Color.RESET+str(list(dhist.keys())[i]).rjust(30)
                for b in bars: line+="{:56}".format(b[i])
                lines+=line+"\n"
            return lines

        def draw_hist(key,fraction,bins=6):
            r=int(len(esu.generation)*fraction)            
            if isinstance(hyperparameters[key][0],(str)):
                dhist=dict(zip(hyperparameters[key],[ [esu.generation[i].genotype[key][0] for i in range(r)].count(k) for k in hyperparameters[key] ]   ) )
            elif isinstance(hyperparameters[key][0],(bool,np.bool)):
                trues= [esu.generation[i].genotype[key][0] for i in range(r)].count(True)
                falses=[esu.generation[i].genotype[key][0] for i in range(r)].count(False)
                dhist= dict(zip(["true ","false"],[trues,falses]))
            elif isinstance(hyperparameters[key][0],(int,np.int,np.int64,np.int32,float,np.floating)): 
                if isinstance(hyperparameters[key][0],(int,np.int,np.int64,np.int32)):
                    rng=(min(hyperparameters[key]),max(hyperparameters[key]))
                    for i in range(1000):
                        if (max(hyperparameters[key])-min(hyperparameters[key]))%(bins+i)==0:
                            bins=bins+i
                            break
                        if bins>i and (max(hyperparameters[key])-min(hyperparameters[key]))%(bins-i)==0:
                            bins=bins-i
                            break                        
                    hist=np.histogram([esu.generation[i].genotype[key][0] for i in range(r)],bins=bins,range=rng)
                    k=[  "["+str(hist[1][i])+"-"+str(hist[1][i+1])+"]" for i in range(len(hist[0]))]
                else:
                    all_values=[ esu.generation[i].genotype[key][0] for i in range(len(esu.generation)) ]
                    hist=np.histogram([esu.generation[i].genotype[key][0] for i in range(r)],bins=bins,range=(min(all_values),max(all_values)))
                    k=[  "["+"{:5.4g}".format(hist[1][i])+"-"+"{:5.4g}".format(hist[1][i+1])+"]" for i in range(len(hist[0]))]
                dhist=dict(zip(k,hist[0]))
            return dhist

        for h_key in hyperparameters.keys():
            if len(hyperparameters[h_key])>1:
                hists=[draw_hist(h_key,f) for f in [0.1,0.25,0.5,1.0]]
                text_to_report+="\n\nfrequency of "+str(h_key)+" in the first decile/quartile/half and overall population:\n"
                text_to_report+="_"*80+"\n"
                text_to_report+=mytermcharts_bar(hists)
                text_to_report+="_"*80+"\n"

        #report scores statistics
        if len(score_statistics_indexes)>0:
            text_to_report+="\n\n=====================================scores statistics==============================================\n"
            text_to_report+="r2, rmse, mae(CV test data), mae(training data), computational time\n"
            text_to_report+=esu.scores_statistics(report_scores_indexes=score_statistics_indexes,score_index_for_best=2)


        return text_to_report



if __name__=="__main__":

    from routes import extracted_data_route
    #name of file
    csv_file=extracted_data_route+"values_extracted-gibbs-swb97xd.25.csv" 
    json_file=extracted_data_route+"molecular_graphs-gibbs-swb97xd.25.json"

    data_file=csv_file[:-4]+"_all_with_graph_data.csv"
    data_file=data_file.split("/")[-1]


    #some parameters:
    population=100
    copies_per_gen=1
    time_limit=300
    cv_splits=7

    score_function_args={"cv_file":"cv_splitter"+str(cv_splits)+".json",
                        "data":data_file,
                        } 
    score_statistics_indexes=[0,1,2,3,4]
    i=0
    while i<len(sys.argv): 
        if sys.argv[i]=="population": population=int(sys.argv[i+1])
        if sys.argv[i]=="time_limit": time_limit=sys.argv[i+1]
        i+=1

    alleles={**hyperparameters}
    
    #prepare the data
    prepare_eq_data(file_name=csv_file,drop_compounds=drop_compounds,test_size=0.0,correlated_groups=correlated_groups,train_suffix="_all.csv",
                    standarize=False) #standarization will take place on the graph data
                    #standarize=True,save_standard_scalers_to_file="e_standard_scalers.txt")
    prepare_graph_data_to_ML(json_file=json_file,csv_file_name=csv_file,correlated_groups=correlated_groups,test_suffix="",train_suffix="_all.csv",
                            save_standard_scalers_to_file="g_standard_scalers.txt",prepare_test_set=False,standarize=True,std_transformer="RobustScaler") 
    sys.exit()
    data=pd.read_csv(data_file,low_memory=True)

    all_features= linear_features + non_linear_features+non_linear_features2+non_linear_features3
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=5,random_state=42)
    cluster_index=kmeans.fit_predict(data[list(all_features)])
    skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_splitter=json.dumps( list(skf.split(cluster_index,cluster_index)) ,cls=NpEncoder) 
    with open("cv_splitter"+str(cv_splits)+".json","w") as f: f.write(cv_splitter)

    #with open("summary"+str(i)+".txt","w") as f: f.write("")
    for i in range(0,1000):

        def fake_score_model(genotype,**kwargs):
                print("fake function")
                r=[np.random.random()]*5 
                return {"score":r, "genotype": genotype}#to test the code, just return random values
            
        if i==0: #the first time, create a generation 

            esu=flock_of_finches(alleles=alleles,
                                    copies_per_gen=copies_per_gen,
                                    bounds=bounds,
                                    score_function=score_model,
                                    score_function_args=score_function_args,
                                    population=population,
                                    initial_mix=True,                 
                                    valid_score_range=[-10.0,10.0] ) 
            print ("generation created")          
        else:
            #esu.Lokta_Volterra_new_generation(mix_parents="norm_distribution",age_halving_life_expectation=100,hunt_criteria=["score-0"],hunt_criteria_combination="sum",score_index=2,
            #                                      reproduction_rate=0.15,inmigration_rate=0.1,predation_rate=0.05,falcon_death_rate=0.4,falcon_feeding_yield=0.06,min_finches=100,
            #                                      min_falcons=1) #using default values of parameters
            esu.fixed_population_new_generation(mix_parents="norm_distribution",diversity_entropy_kept=0.0,score_index=2,probability_factor=2,reproduction_rate=0.75)
            #for e in esu.generation: print (e.genotype)
            with open("summary"+str(esu.generation_number)+".txt","w") as f: 
                s=""
                f.write(s+"_______________\ngeneration # "+str(i+1)+":\n---------------\nstarting at:"+str(datetime.now())+"\n")
                f.write("number of finches in current population:" +str(esu.population)+"\n")
                f.write("number of falcons in current population:" +str(esu.falcons)+"\n")
            #evaluate models in generation

        evaluation=esu.evaluate_generation(nodes="slurm-7",scores_file="borrame",fail_score=[-1000]*7,time_limit=time_limit,force_reevaluation=False)            
        #fake evaluation (for debuggin):
        #for e in esu.generation: e.score=[np.random.random(1)[0],np.random.random(1)[0],np.random.random(1)[0]]
        sortable=esu.sort_by_scores()
        #save to report
        with open("summary"+str(esu.generation_number)+".txt","a") as f: f.write(report(esu,score_statistics_indexes=score_statistics_indexes))
        #save scores
        with open("allscores.txt","w") as f: f.write(str([gen.score for gen in esu.generation]  ))

        #wrap it in a function so it can be called by multiprocess without interrupting the other processes
        def print_out(consensus=[1,5],score_index=0,file_prefix="",file_suffix=""):
                n_models=consensus[-1]
                scores=[f.score[score_index] for f in esu.generation]
                sorted_finches=[f for _,f in sorted(zip(scores,esu.generation),reverse=True)] 
                # a flag to state if png and html files will be printed (only if there is any difference with respect to previous generation's html and png files)
                already_printed=all([(isinstance(f.memotype,dict) and "linear_pka_prediction" in f.memotype.keys() and  "pka_prediction" in f.memotype.keys()) for f in sorted_finches[0:n_models]]) 
                #calculate result summary of the model
                results=model_results(finches=sorted_finches[0:n_models],data_file=data_file,cv_file="cv_splitter"+str(cv_splits)+".json",
                                                    save_model_parameters=file_prefix+"best_model"+file_suffix,
                                                    force_recalculation=False)
                print ("results:"+str(results))

                if already_printed==False:  
                    print ("consensus"+str(consensus))
                    for n in consensus:
                        if len(consensus)>1: file_name=file_prefix+"models-"+"_"+str(n)+"best_models_score"+str(score_index)+"_generation"+file_suffix
                        elif len(consensus)==1: file_name=file_prefix+"best_model_score"+str(score_index)+"_generation"+file_suffix
                        plot_html(consensus_linear_pka_pred=np.mean(results["consensus_linear_pka_pred"][0:n],axis=0),
                                consensus_residuals=np.mean(results["consensus_residuals"][0:n],axis=0),
                                consensus_pka_pred=np.mean(results["consensus_pka_pred"][0:n],axis=0),
                                consensus_pred_residuals=np.mean(results["consensus_pred_residuals"][0:n],axis=0),
                                mean_unsigned_error=np.mean(results["mean_unsigned_error"][0:n]),
                                mean_linear_unsigned_error=np.mean(results["linear_mean_unsigned_error"][0:n]),
                                real_pka=results["pka"],
                                names=results["names"],
                                file_name=file_name)

                with open(file_prefix+"summary"+file_suffix+".txt","a") as f:
                    f.write("\n--------------------------------------------------\n")
                    f.write("result of best finches using as criteria score# "+str(score_index)+":")
                    f.write("\n--------------------------------------------------\n") 
                    for n in consensus:
                        if n==1:
                            f.write("\n MUE with best model:                {:15.3}".format((results["mean_unsigned_error"][0].mean())))
                        else:
                            f.write("\n MUE with "+str(n)+" models:                {:15.3}".format(results["mean_unsigned_error"][0:n].mean()))
                        f.write("\n")    

                        #for debuggin...
                        f.write("using finches with id:")
                        for fnch in sorted_finches[0:n_models]: f.write("\n "+str(fnch.finch_id))
                        #f.write("\n")
                        #f.write(str(sorted_finches[0].memotype))
                        import termcharts
                        from termcharts.colors import Color
                        def mytermcharts_bar(data,title):
                            s=title+"\n"
                            termcharts_output=termcharts.bar(list(data[0]),title=title,mode="h").split("\n")
                            for i,t in enumerate(termcharts_output[4:-1:2]):
                                s+=Color.RESET+("  ["+str(data[1][i])+"-"+str(data[1][i+1])+"] ").rjust(20)+t+"\n\n"
                            return s
                        
                        if n==1: 
                            title="ABS ERROR HISTOGRAM with best model"
                            errors_hist=np.histogram(np.abs(results["consensus_pka_pred"][0]-results["pka"]),bins=[0,0.25,0.5,0.75,1.0,1.5,2,10]) 
                        else:    
                            title="ABS ERROR HISTOGRAM with "+str(n)+" models"
                            errors_hist=np.histogram(np.abs(np.mean(results["consensus_pka_pred"][0:n],axis=0)-results["pka"]),bins=[0,0.25,0.5,0.75,1.0,1.5,2,10]) 
                        f.write("_"*80+"\n")
                        f.write(mytermcharts_bar(list(errors_hist),title=title))
                        f.write("_"*80+"\n")

                    f.write("\n\ngeneration saved to: last_generation"+str(esu.generation_number) +" "*80+"\n\n")


        #print_out(consensus=[1],score_index=2,file_prefix="",file_suffix=str(esu.generation_number))


        #print best model
        esu.sort_by_scores()
        print_model(genotype=[esu.generation[0].genotype],file_name="best_model_generation"+str(esu.generation_number))
        name_of_file=esu.save_generation(file_name="last_generation"+str(esu.generation_number))    

        if i==0: aggregated_esu=copy.copy(esu)
        else: 
                aggregated_esu.add_generation(esu)
                with open("all_gens.txt","a") as f: f.write("\n-----AGGREGATING ALL MODELS FROM ALL GENERATIONS:----\n"+report(aggregated_esu))
 








