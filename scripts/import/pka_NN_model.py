#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-


import json
import numpy as np
import keras
import tensorflow as tf
import copy

import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
from NpEncoder import NpEncoder
from layers import zero_layer,eGATv2_module,eGIN_module,DenseModule,skipped_attention


#architecture (subclassing)
@keras.saving.register_keras_serializable()
class pka_model(tf.keras.Model):
   
    def get_config(self):
            return  {  **super().get_config(),
            "eq_correlated_groups":     self.eq_correlated_groups,
            "atomic_correlated_groups": self.atomic_correlated_groups,
            "edge_correlated_groups":   self.edge_correlated_groups,
            "categorical_features":     self.categorical_features,

            "eConv_modules_params":     self.eConv_modules_params,
            "dense_modules_params":     self.dense_modules_params,
            "reduce_processed_atm_feat_matrix_params": self.reduce_processed_atm_feat_matrix_params,
            "reduce_eq_features_params": self.reduce_eq_features_params,
            "output_module_params":     self.output_module_params,
            "embedding_params":         self.embedding_params,
            "linear_params":            self.linear_params,

            "outputs":                  self.outputs,
  
            "verbose":                  self.verbose
            }


 
    def __init__(self,**kwargs):
        super().__init__()

        self.eq_correlated_groups=[]
        self.atomic_correlated_groups=[]
        self.edge_correlated_groups=[]
        self.categorical_features=[]
        self.eq_dim_red_layers=[]
        self.atomic_dim_red_layers=[]
        self.edge_dim_red_layers=[]
        self.eConv_layers=[]
        self.embedding_layers=[]
        self.reduce_processed_atm_feat_matrix=[]
        self.reduce_eq_features=[]
        self.Dense_layers=[]

        self.verbose=True

        self.pruned_layers=[]
        self.KAN_layers=[]

        for key,value in kwargs.items():
            
            if key=="eq_correlated_groups":     self.eq_correlated_groups=value
            if key=="atomic_correlated_groups": self.atomic_correlated_groups=value
            if key=="edge_correlated_groups":   self.edge_correlated_groups=value
            if key=="categorical_features":     self.categorical_features=value

            if key=="linear_params":            self.linear_params=value
            if key=="embedding_params":         self.embedding_params=value
            if key=="eConv_modules_params":     self.eConv_modules_params=value
            if key=="dense_modules_params":     self.dense_modules_params=value
            if key=="reduce_processed_atm_feat_matrix_params": self.reduce_processed_atm_feat_matrix_params=value
            if key=="reduce_eq_features_params": self.reduce_eq_features_params=value
            if key=="output_module_params":     self.output_module_params=value

            if key=="outputs":                  self.outputs=value
            if key=="verbose":                  self.verbose=value
            
        if self.dense_modules_params["cancelout_l1"]==0 and self.dense_modules_params["cancelout_l2"]==0: self.cancelout_loss=False    
        else:    self.cancelout_loss=True

        if type(self.dense_modules_params["TANGOS_sp"])==int:  self.dense_modules_params["TANGOS_sp"]=[0.1]*80 #a list large enough so there is a value for each layer
        if type(self.dense_modules_params["TANGOS_ort"])==int: self.self.dense_modules_params["TANGOS_ort"]=[0.01]*80

        self.position_of_skip_connection=-1

                                     
    def build(self,input_shape):
        if self.verbose: print ("starting build method...") 

        x_eq_linear_shape,x_eq_shape,x_atm_shape,a_shape,e_shape,mask_shape,batch_shape=input_shape
        g,z=0,0 #counters

        config_dict=self.get_config()
        config_dict["sample_input"]=[np.ones(i) for i in input_shape]
        with open("pka_predictor_config.json","w") as jsonfile: jsonfile.write(  json.dumps(config_dict,cls=NpEncoder))


        #number_of_atomic_features=len(self.atomic_correlated_groups)
        #if number_of_atomic_features==0: number_of_atomic_features=x_atm_shape[-1]
        number_of_edge_matrices=len(self.edge_correlated_groups)
        if number_of_edge_matrices==0: number_of_edge_matrices=e_shape[-1]
        #number_of_eq_features=len(self.eq_correlated_groups)  #actually, not needed
        #if number_of_eq_features==0: number_of_eq_features=x_eq_shape[-1]

        #linear fit of deltaGs and other energies
        self.linear_fit=keras.layers.Dense(units=1,
                                           kernel_regularizer=keras.regularizers.L1L2(self.linear_params["l1"],self.linear_params["l2"]),
                                           name="linear_fit")
       
        #reducing dimmensionality layers, consisting on a single neurone that produces a linear combination of its inputs; These inputs are the features in the same correlated groups of features 
        #for those features that are not correlated with other (groups of correlated features with only one feature), a trivial non trainable layer with weight=1 is used
        for group,red_dim_layer,type_of_group in zip([self.eq_correlated_groups,self.atomic_correlated_groups,self.edge_correlated_groups],
                                               [self.eq_dim_red_layers,self.atomic_dim_red_layers,self.edge_dim_red_layers],
                                               [" eq features"," atomic features"," edge features"]):

            for prl in group:
                if len(prl)==1:
                    if g==1 and prl[0] in self.categorical_features: #if the feature is categorical, multiply it by a non-trainable weight initialized as 0
                        #the effect is to kill the feature, as it will be embedded and the embeddings will be added as features
                        #not the most elegant code, but much much simpler than removing it from the tf tensor, that requires slicing and a headache
                        red_dim_layer.append(zero_layer(name="zero_cat_feature_"+str(z)));z+=1
                    else:
                        red_dim_layer.append(keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer="ones",name="input_layer_"+str(g),trainable=False))
                        g+=1
                        #red_dim_layer.append(keras.layers.Identity(name="trivial_layer_"+str(g)+"."+str(i)))
                else: 
                      #red_dim_layer.append(keras.layers.Dense(1,activation=None,use_bias=False,name="red_dim_layer_"+str(g)+"."+str(i)))
                      red_dim_layer.append(keras.layers.Dense(1,activation=None,use_bias=False,name="input_layer_"+str(g),trainable=True,kernel_initializer="ones"))
                      if self.verbose:
                          print("creating dimensional reduction layer for correlated groups of "+str(len(prl))+type_of_group)
                      g+=1


        #add embedding layers for categorical features
        for i in range(len(self.categorical_features)): 
            self.embedding_layers.append(
                keras.layers.Embedding(input_dim=self.embedding_params[i]["embedding_vocab_sizes"],
                                       output_dim=self.embedding_params[i]["embedding_sizes"],
                                       name="embedding_"+str(i)) )

        #create eConv_modules (wether eGIN or eGATv2) for transforming atomic features 
        for n in range(number_of_edge_matrices):
            for k,pool_layer in enumerate(self.eConv_modules_params["pooling"]):
                eGClayer_params=copy.deepcopy(self.eConv_modules_params)
                eGClayer_params["pooling"]=pool_layer
                if self.eConv_modules_params["type"]=="eGIN":
                    print("edge matrix number: "+str(n))
                    print("pooling: "+str (eGClayer_params["pooling"]))
                    l=eGIN_module(**eGClayer_params, name="eGIN_module_"+str(n)+"-"+str(k)) 
                    self.eConv_layers.append(l)
                    self.pruned_layers.append(l) 
                    self.KAN_layers.append(l)
                elif self.eConv_modules_params["type"]=="eGATv2":
                    l=eGATv2_module(**eGClayer_params, name="eGATv2_module_"+str(n)+"-"+str(k))
                    self.eConv_layers.append(l)
                    self.pruned_layers.append(l)
                if self.verbose: print("creating "+self.eConv_modules_params["type"]+"module "+str(n)+"-"+str(k)+" with global pooling: "+str(pool_layer))


        #create  layers to reduce dimmension of the atom features matrix
        for i,params in enumerate(self.reduce_processed_atm_feat_matrix_params):
            
            if any( k in params.keys() for k in ["dense_shape"]):
                l=DenseModule(**params,return_one_value=False,name="reducing_atom_matrix_dimension_"+str(i))
                self.reduce_processed_atm_feat_matrix.append(l)
                self.pruned_layers.append(l)
            elif any( k in params.keys() for k in ["key_dim","num_heads","output_shape"]): #still have to prune it
                self.reduce_processed_atm_feat_matrix.append(skipped_attention(**params,
                                                            name="at_feat_skipped_attention_layer_"+str(i) ) )                

        #add batch normalization layer for processing tranformed atom features prior to processing in a dense module
        self.batchnorm_transformed_atom_features=keras.layers.BatchNormalization(scale=False,center=False,name="batchnorm_transformed_atom_features")   

        #create dense layers to reduce dimmension of the eq features
        for i, params in enumerate(self.reduce_eq_features_params):
            if any( [k in params.keys() for k in ["dense_shape"]]):
                l=DenseModule(**params,return_one_value=False,name="reducing_eq_features_dimension_MLP"+str(i))
                self.reduce_eq_features.append(l)
                self.pruned_layers.append(l)
                self.KAN_layers.append(l)
            elif any( k in params.keys() for k in ["key_dim","num_heads","output_shape"]):
                self.reduce_eq_features.append(skipped_attention(**params,
                                                            name="reducing_eq_features_dimension_multihead_attention_layer"+str(i) ))

        #create last module
        self.output_module=DenseModule(**self.output_module_params,return_one_value=True,name="output_module_non_linear")

        #create auxiliary output module for processing atomic data only7
        if "aux_at_pKa" in self.outputs:
            self.aux_output_module=DenseModule(**self.output_module_params,return_one_value=True,name="aux_output_module_att")
            self.pruned_layers.append(self.aux_output_module)

        if self.verbose: print ("finished build() method")
        self.built=True

    def call(self,inputs):
       
        #(values of eq. linear features,values of eq. features, atom features matrix ,adjacency matrix, edge matrices, mask,name)
        x_eq_linear,x_eq,x_atm,a,e,mask,name = inputs

        #a=tf.cast(a,dtype=tf.float32) #needed?
        
        # calculate linear prediction of pka
        pka_linear=self.linear_fit(x_eq_linear)

        #resolve categorical features in embeddings
        #note that the features are not removed from x_eq vector: they are ignored because they are multiplied by 0 (see build method)
        if len(self.categorical_features)>0:
            for i,f in enumerate(self.categorical_features):
                x_categorical=x_eq[:,f]
                x_categorical+=5 #charge -5 is word 0, charge -4 is word 1, etc
                if i==0: x_embedding=self.embedding_layers[i](x_categorical)
                else: x_embedding=tf.concat(x_embedding,self.embedding_layers[i](x_categorical))

        #reduce dimensions of eq features combining those that are correlated with a single neurone 
        if len(self.eq_correlated_groups)>0:          
            #group the features according to self.eq_correlated_groups list of lists, so features in the same group are combined 
            #with one of the eq_dim_red_layers 
            groups_of_corr_features=[]
            for gr in self.eq_correlated_groups:
                x_new=x_eq[:,gr[0],tf.newaxis]
                for j in range(1,len(gr)): 
                    x_new=tf.concat((x_new,x_eq[:,gr[j],tf.newaxis]),axis=1)       
                groups_of_corr_features.append(x_new)
            x_eq_red=self.eq_dim_red_layers[0](groups_of_corr_features[0])
            for i in range(1,len(groups_of_corr_features)):
                x_new=self.eq_dim_red_layers[i](groups_of_corr_features[i])
                x_eq_red=tf.concat(( x_eq_red,x_new),axis=1)
        
        #concatenate embedding features
        x_eq=tf.concat((x_eq_red,x_embedding),axis=1)

        #reduce dimensions of atomic features combining those that are correlated with a single neurone 
        if len(self.atomic_correlated_groups)>0:
            #group the atomic features according to self.atomic_correlated_groups list of lists, so features in the same group are combined 
            #with aone of the atomic_dim_red_layers 
            groups_of_corr_features=[]
            #x_atm_t=tf.transpose(x_atm,[0,1,2]) #x_atm dimensions are batch_size x n_features x n_atoms, so first transpose it to batch_size x n_atoms x n_features 
            for gr in self.atomic_correlated_groups:
                x_atm_new=x_atm[:,:,gr[0],tf.newaxis]
                for j in range(1,len(gr)): 
                    x_atm_new=tf.concat((x_atm_new,x_atm[:,:,gr[j],tf.newaxis]),axis=2)   
                groups_of_corr_features.append(x_atm_new)
            x_atm_red=self.atomic_dim_red_layers[0](groups_of_corr_features[0])
            for i in range(1,len(groups_of_corr_features)):
                x_atm_new=self.atomic_dim_red_layers[i](groups_of_corr_features[i])
                x_atm_red=tf.concat(( x_atm_red,x_atm_new),axis=2)
            x_atm=x_atm_red

        #reduce dimensions of edge features combining those that are correlated with a single neurone 
        if len(self.edge_correlated_groups)>0:
            #group the edge features according to self.edge_correlated_groups list of lists, so features in the same group are combined 
            #with aone of the atomic_dim_red_layers 
            groups_of_corr_features=[]
            #x_atm_t=tf.transpose(x_atm,[0,1,2]) #x_atm dimensions are batch_size x n_features x n_atoms, so first transpose it to batch_size x n_atoms x n_features 
            for gr in self.edge_correlated_groups:
                e_new=e[:,:,:,gr[0],tf.newaxis]
                for j in range(1,len(gr)): 
                    e_new=tf.concat((e_new,e[:,:,:,gr[j],tf.newaxis]),axis=3)   
                groups_of_corr_features.append(e_new)
            e_red=self.edge_dim_red_layers[0](groups_of_corr_features[0])
            for i in range(1,len(groups_of_corr_features)):
                e_new=self.edge_dim_red_layers[i](groups_of_corr_features[i])
                e_red=tf.concat(( e_red,e_new),axis=3)
            e=e_red
 
        #edge matrices contain: 
        # -identical matrix, adjacency matrix, or analogue of adjacency matrix defining nodes separated by 2 (or 3) bonds
        # -information of how bond order (or inverse distance or force constant) between pair of atoms changes during protonation
        #these matrices will be used as the adjacency matrix in a GIN:
            
        #unroll edge matrix from batch_size x n_atoms x n_atoms x n_vector_features to: n_vector_features x batch_size x n_atoms x n_atoms 
        e_unrolled=tf.transpose(e,[3,0,1,2]) 
        transf_atom_matrix=None
        j=0
        for i in range (e_unrolled.shape[0]): 
            for pool_layer in self.eConv_modules_params["pooling"]:
                v=self.eConv_layers[j]((e_unrolled[i],x_atm,mask))[:,tf.newaxis,:]
                j+=1
                if  transf_atom_matrix==None: transf_atom_matrix=v 
                #dimensions are: [batch_size  x (edge_matrices x number of pooling layers) x atomic features]
                else: transf_atom_matrix=tf.concat([transf_atom_matrix,v],axis=1)                   
        
        #reduce dimensions of the matrix containing atomic information processed by eGIN modules:
        if self.verbose: print ("reducing dimension of matrices with atomic features processed with eGIN modules:")
        x_atm_procsd=transf_atom_matrix
        for l in self.reduce_processed_atm_feat_matrix: 
            x_atm_procsd=l(x_atm_procsd)

        #flatten: reshape to a tensor of dimensions: [batch_size x reduced(atomic features x edge_matrices x number of pooling layers)]
        x_atm_procsd=tf.reshape(x_atm_procsd,[-1,x_atm_procsd.shape[1]*x_atm_procsd.shape[2]])

        #batch normalize before feeding the final dense layers module
        x_atm_procsd=self.batchnorm_transformed_atom_features(x_atm_procsd)
  

        #process eq features
        for l,p in zip(self.reduce_eq_features,self.reduce_eq_features_params):
            if self.verbose: print ("reducing dimension of eq eq features:")
            if any( k in p.keys() for k in ["key_dim","num_heads","output_shape"]):  x_eq=x_eq[:,tf.newaxis,:]                              
            x_eq=l(x_eq)
            if any( k in p.keys() for k in ["key_dim","num_heads","output_shape"]):  x_eq=tf.reshape(x_eq,[-1,x_eq.shape[-1]])

        

        x_final=keras.layers.Concatenate(axis=1)([x_eq,x_atm_procsd])
        #x_final=x_eq #borrame
        if self.verbose: print("final dense module for processing eq and atomic hidden states")
        pka_non_linear= self.output_module(x_final)

        if "aux_at_pKa" in self.outputs:
            pka_aux=self.aux_output_module(x_atm_procsd)
            pka_aux=pka_linear+pka_aux 
        
        #pka_predicted=keras.layers.Add()([pka_linear,pka_eq,pka_at])  
        #pka_predicted=keras.layers.Add()([pka_linear,pka_non_linear])
        pka_predicted=pka_linear+pka_non_linear
         
        if self.dense_modules_params["output_noise"]!=0.0:
              pka_predicted=keras.layers.GaussianNoise(stddev=self.dense_modules_params["output_noise"])(pka_predicted)
              if "aux_at_pKa" in self.outputs: pka_aux=keras.layers.GaussianNoise(stddev=self.dense_modules_params["output_noise"])(pka_aux)

        if self.outputs==["pKa"]: return pka_predicted
        elif "pKa" in self.outputs and "aux_at_pKa" in self.outputs: return pka_predicted,pka_aux

        else: return pka_predicted

    def strip_pruning(self):
        for l in self.pruned_layers:
            l.strip_pruning()

    def update_grid_from_samples(self,x_batch):
        for l in self.KAN_layers: l.update_grid_from_samples(x_batch)

   
