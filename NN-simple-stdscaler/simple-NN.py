#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-


import json
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
import joblib 
import os
import tensorflow_model_optimization as tfmot
import copy


import sys
#imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
#sys.path.insert(0,imports_path)
sys.path.insert(0,"../scripts/import")
from correlated_groups import correlated_groups
from drop_compounds import drop_compounds
from drop_compounds import force_in_test_set_compounds
from drop_compounds import force_in_train_set_compounds
from prepare_data import prepare_eq_data
from prepare_data import prepare_graph_data
from prepare_data import prepare_graph_data_to_ML
from NpEncoder import NpEncoder
from pka_NN_model import pka_model
from routes import extracted_data_route,sampl_extracted_data_route
 
import gpka_spektral_dataset

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

np.random.seed(16)
tf.random.set_seed(16)




    
def fit_and_evaluate_model(node,file_name,batch_size=64,epochs=2000,initial_learning_rate=0.01,snapshots=7,optimizer=keras.optimizers.Adam(0.05),loss="mse",file_prefix="",out_of_bag_size=0.5,data_augmentation=1.1,fraction_features_in=0.8,evaluate_test_set=True):

    #prepare datasets and loaders 
    test_file_name=file_name[:-4]+"_std_test.dataset"
    train_file_name=file_name[:-4]+"_std_train.dataset"


    train_dataset=joblib.load(train_file_name)
    #loaders for pre-training
    pre_train_loader=gpka_spektral_dataset.acbase_BatchLoader(train_dataset,batch_size=batch_size)

    if evaluate_test_set:
        test_dataset=joblib.load(test_file_name)
        test_loader=gpka_spektral_dataset.acbase_BatchLoader(test_dataset,batch_size=batch_size,epochs=1,shuffle=False)

    #find out indexes of correlated groups (needed for creating the neural network)
    eq_correlated_group_indexes=[train_dataset.get_index_of_eq_feature(g) for g in correlated_groups["eq_features_groups"]+correlated_groups["alpha_masked_eq_features_groups"]] 
    atomic_correlated_group_indexes=[train_dataset.get_index_of_atom_feature(g) for g in correlated_groups["atomic_features_groups"]] 
    edge_correlated_group_indexes=[train_dataset.get_index_of_atom_vector(g) for g in correlated_groups["edge_groups"]] 
    categorical_features_indexes=[train_dataset.get_index_of_eq_feature(g)[0] for g in correlated_groups["categorical_features"]]


    # prepare the model (uses vaues defined as global variables):
    pka_predictor= pka_model( dense_modules_params=dense_modules_params, 
                            eConv_modules_params=eConv_modules_params,
                            reduce_processed_atm_feat_matrix_params=reduce_processed_atm_feat_matrix_params,
                            reduce_eq_features_params=reduce_eq_features_params,
                            embedding_params=embedding_params,
                            linear_params=linear_params,
                            output_module_params=output_module_params,
                            outputs=outputs,
                            eq_correlated_groups=eq_correlated_group_indexes,
                            atomic_correlated_groups=atomic_correlated_group_indexes,
                            edge_correlated_groups=edge_correlated_group_indexes,
                            categorical_features=categorical_features_indexes
                            )
    #compile
    pka_predictor.compile(optimizer=optimizer,loss=loss,loss_weights=loss_weights,metrics="mae")

    #tensorboard working directory
    if file_prefix=="": 
        from time import strftime
        file_prefix=strftime("run_%Y_%m_%d_%H_%M")
    tensor_board_dir=file_prefix+"/TB"
    try:os.makedirs(file_prefix)
    except FileExistsError: pass
    try:os.makedirs(tensor_board_dir)
    except FileExistsError: pass    

    #pre-train the model with all features for a small (10) number of epochs
    #callbacks for pre-training 
    if dense_modules_params["pruning"] or eConv_modules_params["pruning"] or reduce_eq_features_params[0]["pruning"]: 
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    else: callbacks=[]
    #pre-training
    pre_train_history=pka_predictor.fit(pre_train_loader.load(),steps_per_epoch=pre_train_loader.steps_per_epoch,verbose=2, callbacks=callbacks,epochs=10) 

    #train the model using snapshots
    counter=0
    for i in range(snapshots):

        n_inputs=len(correlated_groups["eq_features_groups"])+len(correlated_groups["alpha_masked_eq_features_groups"])
        n_inputs+=len(correlated_groups["atomic_features_groups"])+len(correlated_groups["edge_groups"])
        #randomize the features that are introduced in the model; this is done by setting to 0 non-trainable weights of the input layer in the model
        input_layers=[pka_predictor.get_layer(name="input_layer_"+str(i)) for i in range(n_inputs)] #get the names of the input layers
        for layer in input_layers:
            #when the input layer is a single neuron
            if layer.units==1:
                layer.trainable=False
                #print (layer.get_weights())
                if np.random.random()<fraction_features_in:  layer.set_weights(np.ones_like(layer.get_weights()))
                else: layer.set_weights(np.zeros_like(layer.get_weights()))
            #when the input layer consist of several units that combine correlated features
            else: 
                #print(layer.get_weights())
                if np.random.random()<fraction_features_in: 
                    layer.set_weights(np.ones_like(layer.get_weights()))
                    layer.trainable=True
                else:
                    layer.set_weights(np.zeros_like(layer.get_weights()))
                    layer.trainable=False

        counter+=1

        train_dataset=joblib.load(train_file_name)

        #"manual" c-fold cross validation: the folds are stored in json files, so they can be shared between different computing nodes (see: function cv_split)
        #here, the indexes "node" corresponds to train and validation set for each node.
        valid_indexes,test_indexes=[],[]
        if "cv_folds_indexes_test.json" in os.listdir(file_prefix) and "cv_folds_indexes_train.json" in os.listdir(file_prefix):
            with open(file_prefix+"/cv_folds_indexes_test.json","r") as f:  test_indexes=json.loads(f.readline())[node]
            with open(file_prefix+"/cv_folds_indexes_train.json","r") as f:  valid_indexes=json.loads(f.readline())[node]            

        #use indexes to get datasets
        baggin_dataset,valid_dataset=train_dataset.train_test_split(test_size=out_of_bag_size,stratify="cluster_index",with_replacement=True,remove_stratify_feature=False,data_augmentation=data_augmentation,train_indexes=valid_indexes,test_indexes=test_indexes)
        #prepare data loaders
        train_loader=gpka_spektral_dataset.acbase_BatchLoader(baggin_dataset,batch_size=batch_size)
        validation_loader=gpka_spektral_dataset.acbase_BatchLoader(valid_dataset,batch_size=batch_size,shuffle=False)
        #two loaders are needed, since these are iterators and are "consumed" when they are first used
        validation_loader2=gpka_spektral_dataset.acbase_BatchLoader(valid_dataset,batch_size=batch_size,shuffle=False,epochs=1)
        if evaluate_test_set: test_loader=gpka_spektral_dataset.acbase_BatchLoader(test_dataset,batch_size=batch_size,epochs=1,shuffle=False)
        train_names=[g.name for g in baggin_dataset.graphs]
        validation_names=[g.name for g in valid_dataset.graphs]


        #callbacks:
        #model checkpoint: save best model in each snapshot
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath= file_prefix+"/pka_predictor-"+str(counter)+"-"+str(node)+".h5",  
                                                monitor="val_mae", save_weights_only=True, save_best_only=True )]
        
        #cosine annealing of learning schedule, so after each snapshot learning schedule is increased to escape local gradients minima
        def _cosine_anneal_schedule(self, t):
            counter=int(t/epochs)+1
            cos = np.cos((np.pi * t )/epochs)  +1.0
            factor_decay=max(  [(2**(1-counter)),1/2])
            return float(0.5*initial_learning_rate * cos *factor_decay)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=_cosine_anneal_schedule))

        #for KAN
        class UpdateGridCallback(tf.keras.callbacks.Callback):
            
            def __init__(self,grid_update_data):
                self.grid_update_data=grid_update_data

            def on_epoch_begin(self, epoch, logs=None):
                """
                update grid before new epoch begins
                """
                if epoch > 0:
                    for layer in self.model.KAN_layers: layer.update_grid_from_samples(self.grid_update_data)
                    """
                    for layer in self.model.layers:
                        if hasattr(layer, 'update_grid_from_samples'):
                            layer.update_grid_from_samples(x_batch)
                        x_batch = layer(x_batch)
                    """    
                    print(f"Call update_grid_from_samples at epoch {epoch}")
#        grid_update_loader=gpka_spektral_dataset.acbase_BatchLoader(valid_dataset,batch_size=batch_size,shuffle=False,epochs=1)
#        grid_update_data=grid_update_loader.load()
        
        #tensorboard callbacks
        if tensor_board_params["use_tensor_board"]:
            tbcb=tf.keras.callbacks.TensorBoard(log_dir="tensor_board_dir", 
                                                    write_images=tensor_board_params["write_images"],
                                                    write_graph=tensor_board_params["write_graph"],
                                                    histogram_freq=tensor_board_params["histogram_freq"])
            callbacks.append(tbcb)

        #pruning callbacks
        if dense_modules_params["pruning"] or eConv_modules_params["pruning"] or reduce_eq_features_params[0]["pruning"]:
            callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
            callbacks.append(tfmot.sparsity.keras.PruningSummaries(file_prefix))


        #fit
        history=pka_predictor.fit(train_loader.load(),steps_per_epoch=train_loader.steps_per_epoch,
                        validation_data=validation_loader,validation_steps=validation_loader.steps_per_epoch,
                        #validation_split=0.1,
                        callbacks=callbacks, 
                        verbose=2, epochs=epochs) 

        #save model
        h5_file_name=file_prefix+"/pka_predictor-"+str(counter)+"-"+str(node)+".h5"
        pka_predictor.load_weights(h5_file_name) #recover best val_mae weights saved by checkpoint callback

        pka_predictor.strip_pruning()
        #only save model architecture once
        if counter==1 and node==1: 
            pka_predictor.save(file_prefix+"/pkapredictor.keras")
            with open(file_prefix+"/pkapredictor.json","w") as f: f.write(pka_predictor.to_json())
        h5_pruned_file_name=file_prefix+"/pka_predictor-pruned-"+str(counter)+"-"+str(node)+".h5"
        pka_predictor.save_weights(h5_pruned_file_name)
        
        import zipfile
        zipped_pruned_file_name=file_prefix+"/pka_predictor-pruned-"+str(counter)+"-"+str(node)+".h5.zip"
        with zipfile.ZipFile(zipped_pruned_file_name, 'w', compression=zipfile.ZIP_DEFLATED) as f: f.write(h5_pruned_file_name)

        
        #for reading weights from the zipfile:
        with zipfile.ZipFile(zipped_pruned_file_name, 'r') as f: 
            for filename in f.namelist(): pka_predictor.load_weights(filename)

        #remove h5 files to save disk space
        os.remove(h5_pruned_file_name)
        os.remove(h5_file_name)


        #pka_predictor.load_weights(read_zipfile)
        #keras.models.save_model(pka_predictor,h5_pruned_file_name,include_optimizer=False)
        #converter=tf.lite.TFLiteConverter.from_keras_model(pka_predictor)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #pruned_pka_predictor=converter.convert()
        #tflite_file_name=file_prefix+"/pka_predictor-"+str(counter)+"-"+str(node)+".tflite"
        #with open(tflite_file_name,"wb") as f: f.write(pruned_pka_predictor)
        #with open(tflite_file_name,"rb") as f: f.read()
        #interpreter=tf.lite.Interpreter(model_content=tflite_model)
        #interpreter.allocate_tensors()
        #input_index = interpreter.get_input_details()[0]["index"]
        #output_index = interpreter.get_input_details()[0]["index"]
	


        """
        if dense_modules_params["pruning"] or eConv_modules_params["pruning"] or reduce_eq_features_params[0]["pruning"]:
            pka_predictor=tfmot.sparsity.keras.strip_pruning(pka_predictor)
            keras.models.save_model(pka_predictor,h5_file_name,include_optimizer=False)
            converter=tf.lite.TFLiteConverter.from_keras_model(pka_predictor)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            pruned_pka_predictor=converter.convert()

            tflite_file_name=file_prefix+"/pka_predictor-"+str(counter)+"-"+str(node)+".tflite"
            with open(tflite_file_name,"wb") as f: f.write(pruned_pka_predictor)
            # read the tflite model:
            with open(tflite_file_name,"rb") as f: f.read()
            interpreter=tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            input_index = interpreter.get_input_details()[0]["index"]
            output_index = interpreter.get_input_details()[0]["index"]
        """

        #evaluate the model on validation and test sets; save results in csv files 
        if evaluate_test_set:
            loaders=[validation_loader2,test_loader]
            files=[file_prefix+"/pka_predictor-"+str(counter)+"-"+str(node)+"val.csv",file_prefix+"/pka_predictor-"+str(counter)+"-"+str(node)+"test.csv"]
        else:
            loaders=[validation_loader2]
            files=[file_prefix+"/pka_predictor-"+str(counter)+"-"+str(node)+"val.csv"]
        for loader,filename in zip(loaders,files): 
            pka_real,names,pka_predicted=[],[],[]
            results_dict={}
            for batch in loader:
                inputs,target = batch
                pka_real+=list(target)
                names+=list(inputs[-1])
                
                #if dense_modules_params["pruning"] or eConv_modules_params["pruning"] or reduce_eq_features_params[0]["pruning"]:
                #interpreter.set_tensor(input_index, inputs)
                #interpreter.invoke()
                #predicted_values=interpreter.get_tensor(output_index)
                #else: predicted_values=pka_predictor.predict(inputs)
                predicted_values=pka_predictor.predict(inputs)
                if isinstance(predicted_values,tuple): predicted_values=predicted_values[0]
                pka_predicted+=list(np.ndarray.flatten(predicted_values))
            results_dict["names"]= names 
            results_dict["pKa"]= pka_real
            results_dict["pKa predicted"]=pka_predicted
            dataframe=pd.DataFrame.from_dict(results_dict)
            dataframe.to_csv(filename)

#manual split of data in cv folds. Ensures that every instance is tested  a number of times = "times_in_test_folds" 
#it creates json files with the indexes of the data for test and validation set in two separate files. Each file contains 
#a list of lists, so each node can read an element of these lists, that contains indexes for the validation and training set used in this node
def cv_split(data_file_name,stratify="",cv_folds=5,times_in_test_folds=1,data_augmentation=1.0,json_file_name=""):
    import random

    data=pd.read_csv(data_file_name,low_memory=False,encoding='latin-1')

    #dictionary with "cluster index" as key and the np.array of instance indexes (in random order) as values
    strates_dict={}
    for i,c in enumerate(data["cluster_index"]):
        stratify_value=str(c)
        if stratify_value not in strates_dict.keys(): strates_dict[  stratify_value]=[i]
        else: strates_dict[ stratify_value ].append(i)  
    for s in strates_dict.keys(): strates_dict[s]=np.array(strates_dict[s]) 

    for s in strates_dict.keys(): 
        random.shuffle(strates_dict[s])

    #initialize the folds
    cv_final_train_sets,cv_final_test_sets=[None]*cv_folds*times_in_test_folds,[None]*cv_folds*times_in_test_folds


    for r in range(times_in_test_folds):
        for s in strates_dict.keys():
            random.shuffle(strates_dict[s])
            cv_test_sets=np.array_split(strates_dict[s],cv_folds)
            cv_train_sets=[[ j for j in strates_dict[s] if j not in vs   ] for vs in cv_test_sets ]

            for i,cv_train_set,cv_test_set in zip(range(0,cv_folds*(times_in_test_folds+1)),cv_train_sets,cv_test_sets):
                repeated=np.random.choice(cv_train_set,len(cv_test_set))
                if cv_final_train_sets[i+r*cv_folds]==None: cv_final_train_sets[i+r*cv_folds]=list(cv_train_set)+list(repeated)
                else: cv_final_train_sets[i+r*cv_folds]+=list(cv_train_set)+list(repeated)
                if cv_final_test_sets[i+r*cv_folds]==None: cv_final_test_sets[i+r*cv_folds]=list(cv_test_set)
                else: cv_final_test_sets[i+r*cv_folds]+=list(cv_test_set)

    if json_file_name!="": 
        import json
        with open(json_file_name+"_train.json","w") as f: f.write(json.dumps(cv_final_train_sets,cls=NpEncoder))
        with open(json_file_name+"_test.json","w") as f: f.write(json.dumps(cv_final_test_sets,cls=NpEncoder))

    return cv_final_train_sets, cv_final_test_sets

#creates slurm scripts to launch calculations and run them.
#first, it creates the train and test folds and saves them in json files, so they can be shared by all nodes
#note that these scripts run this script -os.path.basename(__file__)- with "run-node" option
def launch(nodes,n_cpu=14,train_file_name="",times_in_test_folds=2):
    import slurmpy
    import os
    from time import strftime
    
    file_prefix=strftime("run_%Y_%m_%d_%H_%M")
    try:os.makedirs(file_prefix)
    except FileExistsError: pass

    if train_file_name!="":
        if train_file_name.endswith(".csv") and train_file_name.endswith("_std_train.csv")==False: 
           train_file_name=train_file_name[:-4]+"_std_train.csv"
        _,_=cv_split(data_file_name=train_file_name,stratify="cluster_index",cv_folds=int(nodes/times_in_test_folds),times_in_test_folds=times_in_test_folds,data_augmentation=1.0,json_file_name=file_prefix+"/cv_folds_indexes")

    slurms=[slurmpy.Slurm("job_"+str(n),{"ntasks-per-node":n_cpu,"nodes":"1","exclude":"atwood"}) for n  in range(nodes)]
    commands=[" ./"+str( os.path.basename(__file__))+" run_node "+str(n) +" "+file_prefix for n in range(nodes)]
    job_ids=[]
    for s,c in zip(slurms,commands): job_ids.append(s.run(c))
    analysis=slurmpy.Slurm("analysis",{"ntasks-per-node":n_cpu,"nodes":"1"})
    command=" ./"+str( os.path.basename(__file__))+" analyze "
    analysis.run(command,depends_on=job_ids)



def analyze(file_name="pka_predictor",include_predictions=[],folder="",n_files=0,labels_file=""):

    print(include_predictions);print(n_files) 
    all_files=os.listdir(folder)
    csv_validation_files=[f for f in all_files if f.endswith("val.csv") and int(f.split("-")[1].split("-")[0])  in include_predictions ]
    csv_validation_files.sort()
    csv_test_files=[f for f in all_files if f.endswith("test.csv") and int(f.split("-")[1].split("-")[0]) in include_predictions ]
    csv_test_files.sort()
    if n_files!=0 and len(csv_validation_files)>n_files:
        import random
        n_files=random.sample(range(0,len(csv_validation_files)),k=n_files)
        csv_validation_files=[csv_validation_files[n] for n in n_files]
        csv_test_files=[csv_test_files[n] for n in n_files]

    validation_dataframes=[pd.read_csv(folder+"/"+f) for f in csv_validation_files]
    test_dataframes=[pd.read_csv(folder+"/"+f) for f in csv_test_files]
    validation_dict,test_dict={},{}

    if len(test_dataframes)>0:
        dataframes=[validation_dataframes,test_dataframes]
        dicts=[validation_dict,test_dict]
        dataframes_names=["val","test"]
    else: 
        dataframes=[validation_dataframes]
        dicts=[validation_dict]
        dataframes_names=["val"]


    maes=[ np.mean(np.abs(df["pKa predicted"]- df["pKa"]  )) for df in validation_dataframes]
    maes=1.0/(np.array(maes)**2)
    maes=maes/np.sum(maes)

    for d,data_dict,w in zip(dataframes,dicts,[maes]*len(dicts)): # zip([validation_dataframes,test_dataframes],[validation_dict,test_dict],[maes,maes]):
        for df,ww in zip(d,w):
            for name,pka,pka_pred in zip(df["names"].tolist(),df["pKa"].tolist(),df["pKa predicted"].tolist()):
                if name not in data_dict.keys():
                    data_dict[name]=[pka,[pka_pred],[ww]]
                else: data_dict[name][1].append(pka_pred); data_dict[name][2].append(ww)

    
    for d,n in zip(dicts,dataframes_names): #zip([validation_dict,test_dict],["val","test"]):
        for k in d.keys():
            predicted_values=d[k][1]
            weights=d[k][2]
            d[k].insert(1,np.mean(predicted_values))
            d[k].insert(2,np.std(predicted_values))
            d[k].insert(3,len(predicted_values))
            if n=="test": d[k].insert(4,np.average (predicted_values,weights=weights) )
        if n=="test":
            text="compn,real pka,pred pka,std dev pka,number of predictions,weighted pred pka,predictions\n"
            for k in d.keys(): text+=k+","+",".join(map(str,d[k][0:5]))+","+",".join(map(str,d[k][5]))+"\n"
        else:
            text="compn,real pka,pred pka,std dev pka,number of predictions,predictions\n"
            for k in d.keys(): text+=k+","+",".join(map(str,d[k][0:4]))+","+",".join(map(str,d[k][4]))+"\n"            
        with open(folder+"/"+n+"_summary.csv","w") as f: f.write(text)


    valid_names=[k for k in validation_dict.keys()]
    if labels_file!="":
        labels=pd.read_csv(labels_file)
        valid_protonated_charges=[labels[labels["compn"]==cmp]["protonated charge"].item() for cmp in valid_names]
        valid_correct_names=[labels[labels["compn"]==cmp]["correct name"].item()+" "+str(chrg)+"->"+str(chrg-1) for cmp,chrg in zip(valid_names,valid_protonated_charges)]
    else: 
        valid_correct_names=valid_names
        valid_protonated_charges=np.zeros(len(valid_names))
    valid_real_pka=[ validation_dict[k][0] for k in validation_dict.keys()]
    valid_pred_pka=[ validation_dict[k][1] for k in validation_dict.keys()]
    valid_std_pka=[ validation_dict[k][2] for k in validation_dict.keys()]

    if len(test_dataframes)>0:
        test_names=[k for k in test_dict.keys()]
        if labels_file!="":
            test_protonated_charges=[labels[labels["compn"]==cmp]["protonated charge"].item() for cmp in test_names]
            test_correct_names=[labels[labels["compn"]==cmp]["correct name"].item()+" "+str(chrg)+"->"+str(chrg-1) for cmp,chrg in zip(test_names,test_protonated_charges)]
        else:
            test_correct_names=test_names
            test_protonated_charges=np.zeros(len(test_names))
        test_real_pka=[ test_dict[k][0] for k in test_dict.keys()]
        test_pred_pka=[ test_dict[k][1] for k in test_dict.keys()]
        test_pred_pka_w=[ test_dict[k][4] for k in test_dict.keys()]
        test_std_pka=[ test_dict[k][2] for k in test_dict.keys()]


    valid_errors=np.abs(np.array(valid_real_pka)-np.array(valid_pred_pka))
    valid_errors_hist=np.histogram(valid_errors,bins=[0,0.25,0.5,0.75,1.0,1.5,2,10])
    if len(test_dataframes)>0:
        test_errors=np.abs(np.array(test_real_pka)-np.array(test_pred_pka))
        test_errors_hist=np.histogram(test_errors,bins=[0,0.25,0.5,0.75,1.0,1.5,2,10])
        test_w_errors=np.abs(np.array(test_real_pka)-np.array(test_pred_pka_w))
        test_w_errors_hist=np.histogram(test_w_errors,bins=[0,0.25,0.5,0.75,1.0,1.5,2,10])

    #valid_dataframe=pd.read_csv(folder+"/val_summary.csv")
    #test_dataframe=pd.read_csv(folder+"/test_summary.csv")
    #cv_mae_v=np.mean(np.abs(np.array(valid_dataframe["pred pka"])-np.array(valid_dataframe["pka"])))
    #cv_mae_t=np.mean(np.abs(np.array(test_dataframe["pred pka"])-np.array(test_dataframe["pka"])))
    cv_mae_v=np.mean(np.abs(np.array(valid_pred_pka)-np.array(valid_real_pka)))
    from sklearn.metrics import r2_score,mean_squared_error
    r_2_v= np.mean (    r2_score(np.array(valid_pred_pka),np.array(valid_real_pka)) )
    cv_rmse_v=np.mean(mean_squared_error(np.array(valid_pred_pka),np.array(valid_real_pka)))
    if len(test_dataframes)>0:
        cv_mae_t=np.mean(np.abs(np.array(test_pred_pka)-np.array(test_real_pka)))
        cv_mae_tw=np.mean(np.abs(np.array(test_pred_pka_w)-np.array(test_real_pka)))
        cv_rmse_t=np.mean( mean_squared_error(np.array(test_pred_pka),np.array(test_real_pka)))**0.5
        cv_rmse_tw=np.mean( mean_squared_error(np.array(test_pred_pka_w),np.array(test_real_pka)))**0.5
        r_2_t= np.mean (    r2_score(np.array(test_pred_pka),np.array(test_real_pka)) )
        r_2_tw= np.mean (    r2_score(np.array(test_pred_pka_w),np.array(test_real_pka)) )



    #print in the terminal a histogram of the absolute errors
    import termcharts
    from termcharts.colors import Color
    def mytermcharts_bar(data,title):
        s=title+"\n"
        termcharts_output=termcharts.bar(list(data[0]),title=title,mode="h").split("\n")
        for i,t in enumerate(termcharts_output[4:-1:2]):
            s+=Color.RESET+("  ["+str(valid_errors_hist[1][i])+"-"+str(valid_errors_hist[1][i+1])+"] ").rjust(20)+t+"\n\n"
        return s

    if len(test_dataframes)>0:
        print("_"*80)
        print ("TEST MAE: "+str(cv_mae_t))
        print (mytermcharts_bar(list(test_errors_hist[0]),title="TEST ABS ERROR HISTOGRAM"))
        print("_"*80)
        print("_"*80)
        print ("TEST w-MAE:"+str(cv_mae_tw))
        print (mytermcharts_bar(list(test_w_errors_hist[0]),title="TEST(weighted) ABS ERROR HISTOGRAM"))
        print("_"*80)
        print("_"*80)
    print ("TRAIN CV MAE:  "+str(cv_mae_v))
    print ("TRAIN CV RMSE: "+str(cv_rmse_v))
    print ("TRAIN CV R2:   "+str(r_2_v))
    print (mytermcharts_bar(valid_errors_hist,title="VALIDATIOM ABS ERROR HISTOGRAM"))
    print("_"*80)

    import plotly.graph_objects as go
    def __get_marker(std_list,charges_list,color_by_charge=False,include_colorbar=True):
        marker={"size":4,"colorscale":'Rainbow',                                  
                "line":{"width":1},"showscale":True,
                } 
        colorbar={"y":0.80,"x":0.15, "orientation":"h","tickfont":{"size":18},"thickness":18,"len":0.25}

        if all([s==0 for s in std_list]) or color_by_charge:  #if it is specified or if all standard deviations values are 0, will color points based on the charge
            marker["color"]=charges_list
            marker["cmax"],marker["cmin"]=3,-3
            colorbar.update({"title":{"text": "protonated charge","side":"top"},
                    "tickvals":[-3,-2,-1,0,1,2,3]})
            #marker["colorbar"]["ticktext"]=["{:1f}".format(-3),"{:1f}".format(0),"{:1f}".format(3)]

        else:
            std=np.mean(std_list)
            std_max=np.std(std_list)*2
            marker["color"]=(std_list)
            marker["cmin"],marker["cmax"]=std-std_max,std+std_max
            colorbar.update({"title":{"text": "std. dev.","side":"top"},
                    "tickvals":[std-std_max,std,std+std_max],
                    "ticktext":["{:.2f}".format(std-std_max),"{:.2f}".format(std),"{:.2f}".format(std+std_max)]
                    })
        if not include_colorbar: marker["showscale"]=False
        marker["colorbar"]=colorbar
        return marker


    if len(include_predictions)>1: range_in_file_name=str(include_predictions[0])+"-"+str(include_predictions[-1])    
    else: range_in_file_name=str(include_predictions[0])
    if len(test_dataframes)>0:
        #figs=[fig1,fig2,fig3]
        #hists=[hist1,hist2,hist3]
        cv_maes=[cv_mae_v,cv_mae_t,cv_mae_tw]
        cv_rmses=[cv_rmse_v,cv_rmse_t,cv_rmse_tw]
        cv_r2s=[r_2_v,r_2_t,r_2_tw]
        std_data=[valid_std_pka,test_std_pka,test_std_pka]
        protonated_charges=[valid_protonated_charges,test_protonated_charges,test_protonated_charges]
        names=[valid_correct_names,test_correct_names,test_correct_names]
        predicted_data=[valid_pred_pka,test_pred_pka,test_pred_pka_w]
        real_data=[valid_real_pka,test_real_pka,test_real_pka]
        file_names=[folder+"/trainCV"+range_in_file_name,folder+"/testCV"+range_in_file_name,folder+"/testCVw"+range_in_file_name]
    else:
        #figs=[fig1]
        #hist=[hist1]
        cv_maes=[cv_mae_v]
        cv_rmse_ts=[cv_rmse_v]
        cv_r2s=[r_2_v]
        std_data=[valid_std_pka]
        protonated_charges=[valid_protonated_charges]
        names=[valid_correct_names]
        predicted_data=[valid_pred_pka]
        real_data=[valid_real_pka]
        file_names=[folder+"/trainCV"+range_in_file_name]


    for i in range(len(cv_maes)):
        marker=__get_marker(std_data[i],protonated_charges[i])
        fig=go.Figure(data=go.Scatter(y=np.array(predicted_data[i]),x=np.array(real_data[i]),mode='markers',text=names[i], marker=marker ))
        fig.add_trace(go.Scatter(y=[-8.5,-9.5,18.5,19.5],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.3)',line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(y=[-8.0,-10.0,18.0,20.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.2)',line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip") )
        fig.add_trace(go.Scatter(y=[-7.0,-11.0,17.0,21.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.1)',line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo="skip") )
        fig.update_xaxes(title_text="pKa",title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
        fig.update_yaxes(title_text="pKa pred.",title_font={'size': 24, 'weight': 1000},tickfont={"size":16})
        fig.update_layout(height=800,yaxis_range=[-5,16],xaxis_range=[-5,16],showlegend=False)
        font=go.layout.annotation.Font(size=24,weight=1000)
        fontR=go.layout.annotation.Font(size=24,weight=1000,color="red")
        fontG=go.layout.annotation.Font(size=24,weight=1000,color="green")
        fontB=go.layout.annotation.Font(size=24,weight=1000,color="blue")
        fontO=go.layout.annotation.Font(size=24,weight=1000,color="orange")
        fig.add_annotation(x=0.1,y=0.70,xref="paper", yref="paper",text= "M.U.E: ",font = font, showarrow=False )
        fig.add_annotation(x=0.3,y=0.70,xref="paper", yref="paper",text="{:.3f}".format(cv_maes[i]),font = fontR, showarrow=False )
        fig.add_annotation(x=0.1,y=0.62,xref="paper", yref="paper",text="R.M.S.E: ",font = font, showarrow=False )
        fig.add_annotation(x=0.3,y=0.62,xref="paper", yref="paper",text="{:.3f}".format(cv_rmse_ts[i]),font = fontG, showarrow=False )
        fig.add_annotation(x=0.1,y=0.57,xref="paper", yref="paper",text= "R\u00b2:      ",font = font, showarrow=False )
        fig.add_annotation(x=0.3,y=0.57,xref="paper", yref="paper",text= "{:.3f}".format(cv_r2s[i]),font = fontB, showarrow=False )
        fig.write_html(file_names[i]+".html")
        fig.write_image(file_names[i]+".png", width=800, height=800,scale=4)

        hist_data=np.array(real_data[i])-np.array(predicted_data[i])
        residuals_histogram=go.Histogram(x=hist_data,opacity=0.75,xbins={"size":0.25},showlegend=False,
                                    marker_color="red",name="final model",legendgroup=1,legendrank=2,
                                    hoverinfo='skip'
                                    #xaxis='x2',yaxis='y2',
                                )
        marker=__get_marker(std_data[i],protonated_charges[i],color_by_charge=True,include_colorbar=False)
        marker["symbol"]=142
        marker["size"]=20
        rug_residuals=go.Scatter(x=hist_data,y=[1.0]*len(hist_data),
                                mode='markers', text=text,showlegend=False,marker=marker,
                                #xaxis='x2',yaxis='y2',
                                #marker=dict(color=data["protonated charge"],
                                #            colorscale='Rainbow',cmin=-5, cmax=3,
                                #            symbol=142,size=20)          
                                )
        from plotly.subplots import make_subplots
        hist = make_subplots( rows=2,cols=1,
                            #subplot_titles=["mue is:"+str(mean_unsigned_error),"residuals","errors"],
                            #subplot_titles=["histogram",],
                            row_heights=[0.8,0.2],
                            vertical_spacing=0.1,horizontal_spacing=0.0)


        hist.add_trace(residuals_histogram,row=1,col=1)
        hist.add_trace(rug_residuals,row=2,col=1)
        hist.update_layout(height=400,width=400,plot_bgcolor='rgba(255,255,255,1)',bargap=0.2)
        hist.update_xaxes(visible=True,row=1,col=1,range=[-4,4],griddash="dot",linewidth=2,
                        linecolor="#444",gridcolor="#D62728",showticklabels=True,mirror=True,tickvals=np.arange(-4,5,1),tickfont={"size":24})
        hist.update_yaxes(visible=True,row=1,col=1,griddash="dot",linewidth=2,linecolor="#444",gridcolor="#D62728",mirror=True,tickfont={"size":24})
        hist.update_xaxes(visible=False,row=2,col=1,range=[-4,4])
        hist.update_yaxes(visible=False,row=2,col=1,griddash="dot")
        hist.write_html(file_names[i]+"-hist.html")
        hist.write_image(file_names[i]+"-hist.png", width=800, height=800,scale=4)




def load_model(weight_file_name,model_json_file):

    with open(model_json_file) as json_file: config= json.load(json_file)
    sample_input=config.pop("sample_input")
    samp=[]
    for s in sample_input: samp.append(tf.constant(s,dtype=tf.float32))
    sample_input=tuple(samp)
    restored_model=pka_model(**config)
    restored_model(samp)
    #for reading weights from the zipfile:
    import zipfile
    with zipfile.ZipFile(weight_file_name, 'r') as f: 
        for filename in f.namelist(): f.extract(filename)
        restored_model.load_weights(weight_file_name.split(".zip")[0])
        os.remove(weight_file_name.split(".zip")[0])

    return restored_model





#parameters of the model architecture and training
batch_size=256
epochs=200 #200
nodes=10 #24
snapshots = 100 # number of snapshots  
data_augmentation=0.0
fraction_features_in_model=1.0#0.8
out_of_bag_size=0.25 #ignored, since it is used in: acbase_sekeptral_dataset.train_test_split only when test and train indexes are not explicitely given
initial_learning_rate=0.02
optimizer=keras.optimizers.Adam(initial_learning_rate)
#loss=["mse","mse"]
#loss_weights=[1.0,0.0]
#outputs=["pKa","aux_at_pKa"]
#monitor="val_output_1_mae"

tensor_board_params={"use_tensor_board":False,
                    "histogram_freq":0, #if 0, will not save histograms
                    "write_images":True,
                    "write_graph":True,
                    }

loss_weights=[1.0]
outputs=["pKa"]
monitor=["val_mae"]
loss="mse"

linear_params=dict(
    l1=0.005,
    l2=0.005)

embedding_params=[dict(
    embedding_sizes=2,
    embedding_vocab_sizes=10
    )]      


eConv_modules_params=dict (

        type="eGIN", #"eGATv2", # or "eGIN"

        #for eGATv2
        key_dim=0.3,
        query_dim=0.3,
        value_dim=0.3,
        use_key_bias=True,
        use_query_bias=True,
        use_value_bias=True,
        num_heads=4,
        concat_heads=True,
        wq_is_wk=True,
        wk_is_wv=False,

        #for eGIN using attention
        #num_heads=2,
        #key_dim=2.0,
        #use_bias=False,
        #output_shape= 1.0,

        #batchnorm=True,
        #non_ortogonal_penalty=0.00,

        #for eGIN:
        mlp_shape=[[0.75,0.5]],#[[0.2,0.05],[0.2,0.05]], #without KAN: [[1.0,0.75],[0.75,0.75*0.75]]
        KAN_gridsize=[[0,0],[0,0]],#[[8,8],[8,8]],
        batchnorm=True,
        activation="relu", #ignored if KAN
        dropout_rate=0,    #ignored if KAN
        l1=0.00001,        #ignored if KAN
        l2=0.00001,        #ignored if KAN


        pruning=False,
        pruning_schedule= "PolynomialDecay",
        pruning_initial_sparsity= 0.0,
        pruning_final_sparsity= 0.75,
        pruning_begin_step= int(0.1*epochs),
        pruning_end_step= int(1.0*epochs),
        pruning_frequency= 10, #int(epochs),  

        consecutive_units=1,#2,
        pooling=["GlobalSumPool","mask"] #["mask","GlobalMaxPool"]#,"GlobalSumPool"],
)

dense_modules_params=dict (
        dense_shape=[0.75,0.5,0.5],#[0.75,0.5,0.5],#[0.5,0.5],   #without KAN: [0.75,0.5,0.5]
        KAN_gridsize=[0,0,0],#[5,5,5],
        dropout_factor=[0,0,0], #ignored if KAN
        activation="prelu",  #ignored if KAN
        dense_l1=0.00001,    #ignored if KAN
        dense_l2=0.00001,    #ignored if KAN
        feature_dropout_rate=0.0,
        batchnorm=True,

        input_noise=0.0, #0.05,
        output_noise=0.0, #0.2,
        layer_noise=[0.0],#[0.01,0.01,0.01,0.01,0.01,0.01,0.01],

        cancelout=False,
        cancelout_activation="sigmoid",
        cancelout_l1=0.001,
        cancelout_l2=0.002,
        leakygate=False,
        leakygate_alpha=-0.01,

        use_skip_layer=True,
        skip_insert_point=3,
        skip_filter="none",

        TANGOS=False,
        TANGOS_subsample=0,
        TANGOS_sp=[0.0001]*80, #a list large enough so there is a value for each layer
        TANGOS_ort=[0.00001]*80,  #a list large enough so there is a value for each layer

        pruning=False,
        pruning_schedule= "PolynomialDecay",
        pruning_initial_sparsity= 0.0,
        pruning_final_sparsity= 0.75,
        pruning_begin_step= int(0.1*epochs),
        pruning_end_step= int(1.0*epochs),
        pruning_frequency= 10, #int(epochs),  

)


#two consecutive dense modules to reduce the size of the eq features
reduce_eq_features_params=[copy.deepcopy(dense_modules_params) for i in range(2)]
#reduce_eq_features_params=[copy.deepcopy(dense_modules_params)]

#four consecutive dense modules acting on alternating axis to reduce the size of the atom features after eConv
reduce_processed_atm_feat_matrix_params=[copy.deepcopy(dense_modules_params) for i in range(4)]
#reduce_processed_atm_feat_matrix_params=[copy.deepcopy(dense_modules_params) for i in range(2)]
reduce_processed_atm_feat_matrix_params[0]["reduction_axis"]=1
reduce_processed_atm_feat_matrix_params[1]["reduction_axis"]=2
reduce_processed_atm_feat_matrix_params[2]["reduction_axis"]=1
reduce_processed_atm_feat_matrix_params[3]["reduction_axis"]=2

#final dense module
output_module_params=copy.deepcopy(dense_modules_params)
final_dense_modules_params=copy.deepcopy(dense_modules_params)




if __name__=="__main__":
    import sys

    #parameters of this run (should be options in the command line?)
    #name of files:
    lot="swb97xd"
    csv_file=extracted_data_route+"values_extracted-gibbs-"+lot+".25.csv" 
    json_file=extracted_data_route+"molecular_graphs-gibbs-"+lot+".25.json"
    local_csv_file=csv_file.split("/")[-1]
    test_csv_file=sampl_extracted_data_route+"values_extracted_sampl-gibbs-"+lot+".25.csv" 
    test_json_file=sampl_extracted_data_route+"molecular_graphs-sampl-gibbs-"+lot+".25.json"    

    #options for standarizing data
    std_transformer="StandardScaler"
    standarize= (std_transformer!="")

    drop_compounds=["sm11-21_cation->neut","sm11-23_cation->neut"]

    if len(sys.argv)>1 and sys.argv[1]=="run_node":
        node=int(sys.argv[2])
        folder_name=sys.argv[3]
        fit_and_evaluate_model(node=node,file_name=local_csv_file,batch_size=batch_size,epochs=epochs,
                                initial_learning_rate=initial_learning_rate,snapshots=snapshots,
                                optimizer=optimizer,loss=loss,file_prefix=folder_name,
                                data_augmentation=data_augmentation,out_of_bag_size=out_of_bag_size,
                                fraction_features_in=fraction_features_in_model,evaluate_test_set=False)

    elif len(sys.argv)>1 and sys.argv[1]=="analyze":
        print (sys.argv)
        dirs=[d for d in os.listdir() if os.path.isdir(d)]
        dirs.sort(key=lambda x: os.path.getmtime(x))
        folder=dirs[-1]
        if len(sys.argv)>2 and sys.argv[2] in dirs: folder=sys.argv[2]

        if len(sys.argv)>3 and sys.argv[3].startswith("-"):
                include_predictions=range(0,int(sys.argv[3].split("-")[1]))
        elif len(sys.argv)>3 and sys.argv[3].endswith("-"): 
                include_predictions=range(int(sys.argv[3].split("-")[0]),999)
        elif len(sys.argv)>3  and "-" in sys.argv[3]:
                include_predictions=range(int(sys.argv[3].split("-")[0]),int(sys.argv[3].split("-")[1]))
        else: include_predictions=range(100)#[0,1,2,3]

        if len(sys.argv)>4: 
            n_files=int(sys.argv[4]);print("sys:argv"+str(sys.argv[4]))
        else: n_files=0
        
        analyze(folder=folder,include_predictions=include_predictions,n_files=n_files,labels_file=csv_file)



    elif len(sys.argv)>1 and sys.argv[1]=="test":

        prepare_eq_data(file_name=test_csv_file,drop_compounds=drop_compounds,
                        test_size=0.0,correlated_groups=correlated_groups,train_suffix="",test_suffix="",all_suffix="_all.csv",
                        standarize=standarize,save_standard_scalers_to_file="e_standard_scalers.txt",std_transformer=std_transformer)
        prepare_graph_data(json_file=test_json_file,csv_file_name=test_csv_file,correlated_groups=correlated_groups,prepare_test_set=False,all_suffix="_all.csv",test_suffix="",train_suffix="")

        test_dataset_file=test_csv_file.split("/")[-1][:-4]+"_all.dataset"
        test_dataset=joblib.load(test_dataset_file)

        dirs=[d for d in os.listdir() if os.path.isdir(d)]
        dirs.sort(key=lambda x: os.path.getmtime(x))
        folder=dirs[-1]
        if len(sys.argv)>2 and sys.argv[2] in dirs: folder=sys.argv[2]

        if len(sys.argv)>3 and sys.argv[3].startswith("-"):
                include_models=range(0,int(sys.argv[3].split("-")[1]))
        elif len(sys.argv)>3 and sys.argv[3].endswith("-"): 
                include_models=range(int(sys.argv[3].split("-")[0]),999)
        elif len(sys.argv)>3  and "-" in sys.argv[3]:
                include_models=range(int(sys.argv[3].split("-")[0]),int(sys.argv[3].split("-")[1]))
        else: include_models=range(100)#[0,1,2,3]

        models_included=[f for f in os.listdir(folder) if f.endswith(".h5.zip") if int(f.split("pka_predictor-pruned-")[1].split("-")[0]) in include_models]

        model_json_file="pka_predictor_config.json"

        #results_dict={}
        #pka_real_dict={}
        #predicted_dicts=[]
        results=pd.DataFrame()
        for m in models_included:
            filename=folder+"/test_result"+m.split("pka_predictor-pruned-")[1].split(".h5.zip")[0]+".csv"
            #loaded_models.append(load_model(weight_file_name=folder+"/"+m,model_json_file=model_json_file))
            model=load_model(weight_file_name=folder+"/"+m,model_json_file=model_json_file)
            test_loader=gpka_spektral_dataset.acbase_BatchLoader(test_dataset,batch_size=batch_size,epochs=1,shuffle=False)
            
            pka_real,names,pka_predicted=[],[],[]
            for batch in test_loader:
                inputs,target = batch
                pka_real+=list(target)
                names+=list(inputs[-1])
                
                #if dense_modules_params["pruning"] or eConv_modules_params["pruning"] or reduce_eq_features_params[0]["pruning"]:
                #interpreter.set_tensor(input_index, inputs)
                #interpreter.invoke()
                #predicted_values=interpreter.get_tensor(output_index)
                #else: predicted_values=model.predict(inputs)
                predicted_values=model.predict(inputs)

                if isinstance(predicted_values,tuple): predicted_values=predicted_values[0]
                pka_predicted+=list(np.ndarray.flatten(np.array(predicted_values)))

            #fill the dictionary with values
            if "compn" not in results.keys() and "exp. pKa" not in results.keys(): results["compn"],results["exp. pKa"]=pd.Series(names), pd.Series(pka_real) #only do this once
            results["pKa predicted by models:"+m.split("pka_predictor-pruned-")[1].split(".h5.zip")[0]]=pd.Series([ dict(zip(names,pka_predicted))[n] for n in  results["compn"]  ]) #ensure the order using n from results["compn"]

        results["consensus"]=results.mean(numeric_only=True,axis=1)
        results["std. dev."]=results.std(numeric_only=True,axis=1)
        file_name=folder+"/sampl_results"+str(include_models[0])+"-"+str(include_models[-1])
        print("save results to: ",file_name+".csv")
        results.to_csv(file_name+".csv")

        #get charges from name:
        results["protonated charge"]= pd.Series([ (["5an","4an","3an","2an","an","neut","cation","2cation","3cation","4cation","5cation"].index(n.split("->")[1])-4) for n in results["compn"] ])
        results["deprotonated charge"]=pd.Series([ (["5an","4an","3an","2an","an","neut","cation","2cation","3cation","4cation","5cation"].index(n.split("->")[1])-5) for n in results["compn"]])   

        predicted_pka=results["consensus"]

        import plotly
        import plotly.graph_objects as go
        font=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000)
        fontR=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="red")
        fontG=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="green")
        fontB=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="blue")
        fontO=plotly.graph_objects.layout.annotation.Font(size=24,weight=1000,color="orange")
        fonts=[fontR,fontG,fontB,fontO]
        colors=["red","green","blue","orange"] 

        #multiprotic_pairs=[["sm11-23_cation->neut","sm11-23_neut->an"]]
        multiprotic_pairs=[]
        m_brackets=[]
        """
        for multiprotic_pair in multiprotic_pairs:
                
                i,j=results.index[results["compn"]==multiprotic_pair[0]].tolist()[0],results.index[results["compn"]==multiprotic_pair[1]].tolist()[0]
                a,aa=results.iloc[i],results.iloc[j]
                print([multiprotic_pair[0].startswith("sm6"),multiprotic_pair[0].startswith("sm7"),multiprotic_pair[0].startswith("sm8"),multiprotic_pair[0].startswith("sm11")])
                testset=[multiprotic_pair[0].startswith("sm6"),multiprotic_pair[0].startswith("sm7"),multiprotic_pair[0].startswith("sm8"),multiprotic_pair[0].startswith("sm11")]
                color=np.array(colors)[[multiprotic_pair[0].startswith("sm6"),multiprotic_pair[0].startswith("sm7"),multiprotic_pair[0].startswith("sm8"),multiprotic_pair[0].startswith("sm11")]][0]
                m_brackets.append({"x":[a["exp. pKa"],aa["exp. pKa"]],"y":[a["consensus"],aa["consensus"]],
                                    "names":[a["compn"],aa["compn"]],"std. dev.":[a["std. dev."],aa["std. dev."]],"color":color}) 
                new_name=a["compn"].split("_")[0]+a["compn"].split("_")[1].split("->")[0]+"->"+aa["compn"].split("_")[1].split("->")[1]
                #change value of first equilibrium in pair...
                results.loc[i,"exp. pKa"]=np.average([a["exp. pKa"],aa["exp. pKa"]])
                results.loc[i,"compn"]=new_name
                results.loc[i,"protonated charge"]=a["protonated charge"]
                results.loc[i,"deprotonated charge"]=aa["deprotonated charge"]
                results.loc[i,"consensus"]=np.average([a["consensus"],aa["consensus"]])
                results.loc[i,"std. dev."]="_"#np.average([a["std. dev."],aa["std. dev."]])
                #and remove the other
                results=results[results["compn"]!=multiprotic_pair[1]] 
        """
        for d in drop_compounds:    results =results[results["compn"].str.startswith(d)==False]

        test_data_sm6=results[results["compn"].str.startswith("sm6")==True]
        test_data_sm7=results[results["compn"].str.startswith("sm7")==True]
        test_data_sm8=results[results["compn"].str.startswith("sm8")==True]
        test_data_sm11=results[results["compn"].str.startswith("sm11")==True]

        brackets=[]
        """
        m_brackets=[]
        for m_bracket in m_brackets:
            brackets.append(go.Scatter(x=m_bracket["x"],y=m_bracket["y"],text=m_bracket["names"],mode="lines+markers",showlegend=False,
                            error_y=dict(type="data",array=m_bracket["std. dev."],visible=True,thickness=1,width=3),
                            marker_symbol="x-thin",marker_line_width=2,marker_size=8,marker_color=m_bracket["color"],marker_line_color=m_bracket["color"],
                            line=dict(color=m_bracket["color"], width=3.5,dash='dot')
                            ))
        """


        texts=[[n+" ("+"%+d" %c+" -> "+"%+d" %(c_d)+")" for n,c,c_d in zip(test_data['compn'],test_data['protonated charge'],test_data['deprotonated charge'])] for test_data in [test_data_sm6,test_data_sm7,test_data_sm8,test_data_sm11]]
        pka_traces=[]
        predicted_pkas=[test_data_sm6["consensus"],test_data_sm7["consensus"],test_data_sm8["consensus"],test_data_sm11["consensus"]]
        experimental_pkas=[test_data_sm6["exp. pKa"],test_data_sm7["exp. pKa"],test_data_sm8["exp. pKa"],test_data_sm11["exp. pKa"]]
        errors=[test_data_sm6["std. dev."],test_data_sm7["std. dev."],test_data_sm8["std. dev."],test_data_sm11["std. dev."]]
        for p_pka,exp_pka,color,text,error in zip(predicted_pkas,experimental_pkas,colors,texts,errors): 
            pka_traces.append(go.Scatter(x=exp_pka,y=p_pka,mode='markers', showlegend=False, 
                                error_y=dict(type="data",array=error,visible=True,thickness=1,width=3),
                                marker={"color":color,"size":8,
                                        #"colorscale":'Rainbow',"cmin":-3,"cmax":3,                                  
                                        "line":{"width":1.0},"showscale":False,
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
        fig1.add_annotation(x=0.1,y=0.70,xref="paper", yref="paper",text="  EuroSAMPL ",font=fontO,showarrow=False)

        x_positions=[0.63,0.75,0.83,0.91]
        for mae,rmse,r2,f,x_position in zip(mean_absolute_errors,neg_mean_squared_errors,r_2_scores,fonts,x_positions):
            #x_position+=0.10
            fig1.add_annotation(x=x_position,y=0.4,xref="paper", yref="paper",text="{:.3f}".format(mae),font = f, showarrow=False )
            fig1.add_annotation(x=x_position,y=0.33,xref="paper", yref="paper",text="{:.3f}".format(rmse),font = f, showarrow=False )
            fig1.add_annotation(x=x_position,y=0.285,xref="paper", yref="paper",text="{:.3f}".format(r2),font = f, showarrow=False )        

        
        #fig1.update_layout(legend=dict(y=0.35,x=0.9))

        fig1.write_html(file_name+".html")
        fig1.write_image(file_name+".png", width=1200, height=800,scale=4)






    else:
        prepare_eq_data(file_name=csv_file,drop_compounds=drop_compounds,test_size=0.0,
                        correlated_groups=correlated_groups,prepare_test_set=False,
                        test_suffix="",train_suffix="_std_train.csv",all_suffix="",
                        standarize=standarize,save_standard_scalers_to_file="e_standard_scalers.txt",std_transformer=std_transformer)
        prepare_graph_data(json_file=json_file,csv_file_name=csv_file,correlated_groups=correlated_groups,prepare_test_set=False,
                            test_suffix="",train_suffix="_std_train.csv",all_suffix="")
        launch(nodes,train_file_name=local_csv_file,times_in_test_folds=1)























