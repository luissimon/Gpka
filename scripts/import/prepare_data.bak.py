#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-



import spektral
import json
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
import joblib
import sklearn 
import os
import types
import tensorflow_model_optimization as tfmot
import copy
import tfkan

import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
from correlated_groups import correlated_groups
from drop_compounds import drop_compounds
from drop_compounds import force_in_test_set_compounds
from drop_compounds import force_in_train_set_compounds

import gpka_spektral_dataset


#method to prepare data: standarize, split in train and test set, and saves the result in a csv file.
def prepare_eq_data(file_name,drop_compounds,test_size=0.2,correlated_groups={},test_suffix="_std_test.csv",train_suffix="_std_train.csv",all_suffix="_std_all.csv",standarize=True):

    #features that will be keept, extracted from groups of correlated features
    selected_lin_eq_features= [i for j in correlated_groups["lin_eq_features_groups"] for i in j if "H_difference" not in i]
    selected_eq_features= [i for j in correlated_groups["eq_features_groups"] for i in j if "H_difference" not in i]
    selected_protonated_eq_features=["protonated "+i for i in selected_eq_features]
    selected_deprotonated_eq_features=["deprotonated "+i for i in selected_eq_features]
    categorical_features=correlated_groups["categorical_features"]
    attributes_for_clustering=selected_lin_eq_features+selected_eq_features+selected_protonated_eq_features+selected_deprotonated_eq_features

    #file names:
    all_data_file_name=file_name[:-4]+all_suffix
    all_data_file_name=all_data_file_name.split("/")[-1] #remove the route (so the new files are not created in the "extracted_data" directory
    all_train_data_file_name=file_name[:-4]+train_suffix
    all_train_data_file_name=all_train_data_file_name.split("/")[-1]  #remove the route (so the new files are not created in the "extracted_data" directory
    test_data_file_name=file_name[:-4]+test_suffix
    test_data_file_name=test_data_file_name.split("/")[-1]  #remove the route (so the new files are not created in the "extracted_data" directory

    #read the data:
    data=pd.read_csv(file_name,low_memory=False,encoding='latin-1')
    keep_columns=["compn"]+["pKa"]+categorical_features+attributes_for_clustering
    data=data[keep_columns]
    #data.drop(data.columns.difference(not_to_drop),inplace=True)
    data.dropna(axis=0)
    data.dropna()
    #eliminate discarded compounds, move datat forced to be in test or train set
    for d in drop_compounds:    data =data[data["compn"].str.startswith(d)==False]
    force_in_test_data=pd.DataFrame([])
    for d in force_in_test_set_compounds:
        force_in_test_data=pd.concat([force_in_test_data,   data[data["compn"].str.startswith(d)==True]  ])
        data =data[data["compn"].str.startswith(d)==False]
    force_in_train_data=pd.DataFrame([])
    for d in force_in_train_set_compounds: 
        force_in_train_data=pd.concat([force_in_train_data, data[data["compn"].str.startswith(d)==True] ])
        data =data[data["compn"].str.startswith(d)==False]
    #clustering data to stratified split train and test set
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=4,random_state=42,n_init=10)
    cluster_index=kmeans.fit_predict(data[attributes_for_clustering])
    #print (attributes_for_clustering)
    data["cluster_index"]=cluster_index
    """
    print (len(cluster_index))
    print(list(cluster_index))
    zeros,ones,twos,threes,fours,fives=0,0,0,0,0,0
    i=0
    for ci in cluster_index: 
        if ci==0: zeros+=1
        if ci==1: ones+=1
        if ci==2: twos+=1
        if ci==3: threes+=1
        if ci==3: fours+=1
        if ci==4: 
           fives+=1
           #print(i)
           #print(data["compn"][i])
        i+=1
    print ("cluster zero:"+str(zeros))
    print ("cluster one:"+str(ones))
    print ("cluster two:"+str(twos))
    print ("cluster three:"+str(threes))
    print ("cluster four:"+str(fours))
    print ("cluster five:"+str(fives))
    """
    #split data 
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size,stratify=cluster_index,random_state=42)
    #add to training set those data that is forced to be in it
    #train_data.reset_index(inplace=True, drop=True)
    #force_in_train_data.reset_index(inplace=True, drop=True)
    force_in_train_data["cluster_index"]=1000 
    if len(force_in_train_data)>0: train_data=pd.concat([train_data, force_in_train_data], axis=0, ignore_index=True)
    #add to test set those data that is forced to be in it
    #test_data.reset_index(inplace=True, drop=True)
    #force_in_test_data.reset_index(inplace=True, drop=True)  
    force_in_test_data["cluster_index"]=1000 
    if len(force_in_test_data)>0: test_data=pd.concat([test_data, force_in_test_data], axis=0,ignore_index=True)
    
    #standarize the data
    if standarize:
        standard_scalers={} #using this dictionarry the standard scaler used during training or model selection could also be called during prediction of new data
        not_standarize=selected_lin_eq_features+["compn"]+["pKa"]+categorical_features+["cluster_index"]
        from sklearn.preprocessing import StandardScaler
        for trd in train_data:
            if trd not in not_standarize: 
                standard_scalers[trd]= StandardScaler()
                t=np.asarray(train_data[trd]).reshape(-1,1)
                train_data[trd]=standard_scalers[trd].fit_transform(t) 
                tt=np.asarray(test_data[trd]).reshape(-1,1)
                test_data[trd]=standard_scalers[trd].transform(tt) 
                ttt=np.asarray(data[trd]).reshape(-1,1)
                data[trd]=standard_scalers[trd].transform(ttt)
    #save data
    data.to_csv(all_data_file_name)
    train_data.to_csv(all_train_data_file_name)
    test_data.to_csv(test_data_file_name)

def prepare_graph_data_to_ML(json_file,csv_file_name,correlated_groups,test_suffix="_std_test.csv",train_suffix="_std_train.csv"):

    train_data_file_name=csv_file_name[:-4]+train_suffix
    #train_data_file_name=train_data_file_name.split("/")[-1]   #remove the route, since the files are in local directory
    test_data_file_name=csv_file_name[:-4]+test_suffix
    #test_data_file_name=test_data_file_name.split("/")[-1]     #remove the route, since the files are in local directory


    atomic_features=[ff for f in correlated_groups["atomic_features_groups"] for ff in f]
    edge_features=[ff for f in correlated_groups["edge_groups"] for ff in f]
    eq_features=[ff for f in correlated_groups["eq_features_groups"] for ff in f]

    protonated_eq_features=["protonated "+ff for f in correlated_groups["eq_features_groups"] for ff in f ]
    if "protonated protonated charge" in protonated_eq_features: protonated_eq_features.remove("protonated protonated charge")
    protonated_eq_features.append("protonated expl1wat")
    protonated_eq_features.append("protonated SMD-solv")

    deprotonated_eq_features=["deprotonated "+ff for f in correlated_groups["eq_features_groups"] for ff in f]
    if "deprotonated protonated charge" in deprotonated_eq_features: deprotonated_eq_features.remove("deprotonated protonated charge")
    deprotonated_eq_features.append("deprotonated expl1wat")
    deprotonated_eq_features.append("deprotonated SMD-solv")

    eq_features=eq_features+protonated_eq_features+deprotonated_eq_features
    categorical_features=correlated_groups["categorical_features"]
    linear_eq_features=[i for j in correlated_groups["lin_eq_features_groups"] for i in j]
    #features that will be transofmed to eq_features
    transformed_eq_features=[ff for f in correlated_groups["alpha_masked_eq_features_groups"] for ff in f]
    transformed_eq_features_no_replace=[f for f in transformed_eq_features if f in edge_features or f in atomic_features]
    transformed_eq_features_replace=[f for f in transformed_eq_features if f not in edge_features and f not in atomic_features]

    if test_suffix==train_suffix: prepare_files=[train_data_file_name]
    else: prepare_files=[train_data_file_name,test_data_file_name]
        


    for d in prepare_files: 
        print (d) #borrame
        #load dataset
        
        dataset=gpka_spektral_dataset.gpka_spektral_dataset(json_file,csv_file=d,equilibrium_keys=['correct name']+categorical_features+eq_features,
                                linear_equilibrium_keys=linear_eq_features,label_key="pKa")
        #generate features corresponding to bonds with H and alpha atoms
        if len(transformed_eq_features_replace)>0: dataset.aply_mask_to_atom_features( transformed_eq_features_replace, replace=True )
        if len(transformed_eq_features_no_replace)>0: dataset.aply_mask_to_atom_features( transformed_eq_features_no_replace,replace=False)

        dataset.features_n_bonds_away(suffixes=["alpha","beta","gamma"],exclude_ending="_H_difference")
        pd_dataframe=dataset.eq_features_to_pd_series()
        new_file_name=d.split("/")[-1]#remove the route (so the new files are not created in the "extracted_data" directory
        pd_dataframe.to_csv(new_file_name[:-4]+"_with_graph_data.csv",index=False)
    



#prepare graph data, reading the json file, train_csv_file and tes_csv_file
def prepare_graph_data(json_file,csv_file_name,correlated_groups,test_suffix="_std_test.csv",train_suffix="_std_train.csv"):

    train_data_file_name=csv_file_name[:-4]+train_suffix
    train_data_file_name=train_data_file_name.split("/")[-1]  #remove the route (so the new files are not created in the "extracted_data" directory
    test_data_file_name=csv_file_name[:-4]+test_suffix
    test_data_file_name=test_data_file_name.split("/")[-1]  #remove the route (so the new files are not created in the "extracted_data" directory
    
    #list of flatted features
    atomic_features=[ff for f in correlated_groups["atomic_features_groups"] for ff in f]
    edge_features=[ff for f in correlated_groups["edge_groups"] for ff in f]
    eq_features=[ff for f in correlated_groups["eq_features_groups"] for ff in f]
    categorical_features=correlated_groups["categorical_features"] 
    categorical_features+=["cluster_index"]
    linear_eq_features=[i for j in correlated_groups["lin_eq_features_groups"] for i in j]
    #features that will be transofmed to eq_features
    transformed_eq_features=[ff for f in correlated_groups["alpha_masked_eq_features_groups"] for ff in f]
    transformed_eq_features_no_replace=[f for f in transformed_eq_features if f in edge_features or f in atomic_features]
    transformed_eq_features_replace=[f for f in transformed_eq_features if f not in edge_features and f not in atomic_features]
    #features that will be keept, extracted from groups of correlated features
    all_kept_features=list(set(atomic_features+edge_features+eq_features+categorical_features+transformed_eq_features))

    for d in [train_data_file_name,test_data_file_name]:  
            dataset_file_name=d[:-4]+".dataset"
            #load dataset
            dataset=gpka_spektral_dataset.gpka_spektral_dataset(json_file,csv_file=d,equilibrium_keys=categorical_features+eq_features,
                                        linear_equilibrium_keys=linear_eq_features,label_key="pKa")
            #generate atom features accumulating on each atom the properties of the H bound to it (specified as H_added)
            H_added_keys=[k for k in atomic_features if "H_added" in k]
            for k in H_added_keys:
                k_het=k.replace("_H_added_","_")
                k_H=k.replace("_H_added_","_H_")
                H_added_feature=[f_het+f_H for f_het,f_H in zip(dataset.get_values_of_feature(k_het),dataset.get_values_of_feature(k_H))]
                dataset.add_atom_feature(H_added_feature,k)

            #generate features corresponding to bonds with H and alpha atoms
            if len(transformed_eq_features_replace)>0: dataset.aply_mask_to_atom_features( transformed_eq_features_replace, replace=True )
            if len(transformed_eq_features_no_replace)>0: dataset.aply_mask_to_atom_features( transformed_eq_features_no_replace,replace=False)
            
            #add edge features: related to connectivity
            dataset.add_n_bonds_away_matrix(n=0,name="diagonal")
            dataset.add_n_bonds_away_matrix(n=1,name="adj")
            dataset.add_n_bonds_away_matrix(n=2,name="adj_2")
            dataset.add_n_bonds_away_matrix(n=3,name="adj_3")

            #add edge features: for substituting vector_keys with positive and negative sparsemax, minmax, etc 
            pos_vector_keys=[k for k in edge_features if k.endswith("pos")]
            neg_vector_keys=[k for k in edge_features if k.endswith("neg")]
            for pvk in pos_vector_keys:
                if pvk.endswith("_sparsemax_pos"): dataset.add_sparsemax_matrix(pvk.replace("_sparsemax_pos",""),sign="pos")
                if pvk.endswith("_minmax_pos"):    dataset.add_minmax_matrix(pvk.replace("_minmax_pos",""),sign="pos")
                #(add more cases)
            for pvk in neg_vector_keys:
                if pvk.endswith("_sparsemax_neg"): dataset.add_sparsemax_matrix(pvk.replace("_sparsemax_neg",""),sign="neg")
                if pvk.endswith("_minmax_neg"):    dataset.add_minmax_matrix(pvk.replace("_minmax_neg",""),sign="neg")
                #add more cases
           
            #remove all features except those in all_keep_features
      
            dataset.keep_features(all_kept_features)
            #save file
            joblib.dump(dataset,dataset_file_name)
            

