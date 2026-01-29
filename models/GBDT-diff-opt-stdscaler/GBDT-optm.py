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


    from sklearn.model_selection import cross_validate
    print ("about to start cross_validation") 

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
    """
    prepare_eq_data(file_name=csv_file,drop_compounds=drop_compounds,test_size=0.0,correlated_groups=correlated_groups,standarize=False,train_suffix="_all.csv")
    prepare_graph_data_to_ML(json_file=json_file,csv_file_name=csv_file,correlated_groups=correlated_groups,test_suffix="",train_suffix="_all.csv",prepare_test_set=False) 


    data=pd.read_csv(data_file,low_memory=True)
    print (len(data))

    all_features= linear_features + non_linear_features+non_linear_features2+non_linear_features3
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=5,random_state=42)
    cluster_index=kmeans.fit_predict(data[list(all_features)])
    skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_splitter=json.dumps( list(skf.split(cluster_index,cluster_index)) ,cls=NpEncoder) 
    with open("cv_splitter"+str(cv_splits)+".json","w") as f: f.write(cv_splitter)
    """

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
            esu.fixed_population_new_generation(mix_parents="norm_distribution",diversity_entropy_kept=0.0,score_index=2,probability_factor=2,reproduction_rate=0.75)

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


        #print best model
        print_model(genotype=[esu.generation[0].genotype],file_name="best_model_generation"+str(esu.generation_number))
        name_of_file=esu.save_generation(file_name="last_generation"+str(esu.generation_number))    

        if i==0: aggregated_esu=copy.copy(esu)
        else: 
                aggregated_esu.add_generation(esu)
                with open("all_gens.txt","a") as f: f.write("\n-----AGGREGATING ALL MODELS FROM ALL GENERATIONS:----\n"+report(aggregated_esu))
 





