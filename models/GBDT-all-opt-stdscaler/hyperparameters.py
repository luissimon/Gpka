#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

hyperparameters = {
        "lxalpha":[0.01,2.0],
        "lxl1_ratio":[0.01,2.0],

        "nl_bootstrap":[True,False],
        "nl_max_features":[0.5,1.0],  

        "nl_n_estimators_inner_outer_ratio":[5,10,20],
        "nl_total_n_estimators":[500,1000,1500],
        #"nl_num_parallel_tree":[1,4,12,20,30,35,40,50],  # gor XGBoost, default None
        "nl_booster":["gbtree","dart"],   #str XGBoost, default: gbtree
        "nl_max_depth":[6,8],   #int XGBoost, default: 6
        "nl_subsample":[0.7,0.9],   #float XGBoost, default: 1.0, (0,1] 
        #"nl_max_delta_step":[0.0,0.1,0.2,0.4,0.6,1.0,2.0,4.0,6.0,8.0,10.0],   #float XGBoost, default: 0, [0,inf]
        "nl_gamma":[0.0,0.4],   #float XGBoost, default: 0, [0,inf]
        "nl_reg_lambda":[0.1,1.0,2.0],   #float XGBoost, default: 1, [0,inf]
        "nl_reg_alpha":[0.0,2.0],   #float XGBoost, default: 0, [0,inf]
        "nl_tree_method":["exact","approx","hist","auto"],   #str XGBoost, default: auto
        "nl_refresh_leaf":[True,False],   #bool XGBoost, default: True
        #"nl_max_leaves":[0,10,20,30,50,60,80],   #int XGBoost, default: 0
        "nl_max_bin":[128,256,512],   #int XGBoost, default: 256
        #"nl_rate_drop":[0.0,1.0],   #float XGBoost, default: 0.0
        #"nl_skip_drop":[0.0,1.0],   #float XGBoost, default: 0.0
        "nl_eta": [0.2,0.4]
            }


#bounds of hyperparameters
bounds={}
for k in hyperparameters.keys(): bounds[k]=[0,None]  # all hyperparameters must be>0:
#some more strict bounds:
bounds["lxalpha"]=[0.01,2.0]
bounds["lxl1_ratio"]=[0.01,2.0]
bounds["nl_total_n_estimators"]=[100,2500]
bounds["nl_n_estimators_inner_outer_ratio"]=[2,40]
bounds["nl_max_depth"]=[2,18]
bounds["nl_subsample"]=[0.5,1.0]
bounds["nl_max_bin"]=[64,512]


bounds["nl_max_features"]=[0.01,1.0]
bounds["nlxlearning_rate"]=[0.0001,None]
bounds["nl_subsample"]=[0.01,1]
bounds["nl_rate_drop"]=[0.0,1]
bounds["nl_skip_drop"]=[0.0,1]
bounds["nl_inner_n_estimators"]=[4,None]


bounds["l_eta"]=[0.0,1.0]
