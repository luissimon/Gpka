#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-


import spektral
import numpy as np
import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tfkan

import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)


#cancel-out layer: https://github.com/unnir/CancelOut
# regularization by eliminating less relevant features
@keras.saving.register_keras_serializable()
class CancelOut(keras.layers.Layer):
    '''
    CancelOut Layer
    '''
    def __init__(self, activation='sigmoid', cancelout_loss=True, lambda_1=0.002, lambda_2=0.001,name="cancelout"):
        super(CancelOut, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cancelout_loss = cancelout_loss
        
        if activation == 'sigmoid': self.activation = tf.sigmoid
        if activation == 'softmax': self.activation = tf.nn.softmax

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(1),
            trainable=True)
        
    def call(self, inputs):
        if self.cancelout_loss:
            self.add_loss( self.lambda_1 * tf.norm(self.w, ord=1) + self.lambda_2 * tf.norm(self.w, ord=2))
        return tf.math.multiply(inputs, self.activation(self.w))
    
    def get_config(self):
        dict= {**super().get_config(),
                "activation": self.activation,
                "cancelout_loss": self.cancelout_loss,
                "lambda_1": self.lambda_1,
                "lambda_2": self.lambda_2}
        print (dict)
        return dict  

# LeakyGate as used: https://arxiv.org/pdf/2108.03214.pdf and implemented: https://github.com/jrfiedler/xynn/blob/main/xynn/mlp.py
@keras.saving.register_keras_serializable()
class LeakyGate(keras.layers.Layer):
    def __init__(self, activation=keras.layers.LeakyReLU(),name="leakygate"):
        super(LeakyGate,self).__init__()
        self.activation=activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(1),
            trainable=True,name="kernel")
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(1),
            trainable=True,name="bias")

    def call(self,inputs):
        return self.activation(tf.math.multiply(inputs,self.w) + self.b )    

    def get_config(self):
        return {**super().get_config(),
                "activation": self.activation} 


# Prunable dense and densekan layers (to use with prune)
import tensorflow_model_optimization as tfmot
class prunable_Dense(tf.keras.layers.Dense, tfmot.sparsity.keras.PrunableLayer):
    def get_prunable_weights(self): return [self.kernel]
    #def strip_pruning_wrapper(self): return self.laye,
class prunable_DenseKAN(tfkan.layers.DenseKAN,tfmot.sparsity.keras.PrunableLayer):
    def get_prunable_weights(self): return [self.spline_kernel]


# Trivial layer returning zeros. Used to delete inputs without affecting the next layer's acrhitecture
@keras.saving.register_keras_serializable()
class zero_layer(keras.layers.Layer):
    def get_config(self): return  {  **super().get_config()}
    def __init__(self,**kwargs): super().__init__()
    def build(self,input_shape): pass
    def call(self,inputs): return tf.zeros_like(inputs)


@keras.saving.register_keras_serializable()
class eGATv2_module (keras.Model):

    def get_config(self):
        return  {  **super().get_config(),
            "wq_is_wk":         self.wq_is_wv,
            "wk_is_wv":         self.wq_is_wv,     
            "key_dim":          self.key_dim,
            "use_key_bias":     self.use_key_bias,
            "query_dim":        self.query_dim,
            "use_query_bias":   self.use_query_bias,
            "value_dim":        self.value_dim,
            "use_value_dim":    self.use_value_bias,
            "num_heads":        self.n_hedads,
            "concat_heads":     self.concat_heads,
            "pooling":          self.pooling,
            "pruning":          self.pruning,
            "pruning_schedule": self.pruning_schedule,
            "pruning_initial_sparsity": self.pruning_initial_sparsity,
            "pruning_final_sparsity":   self.pruning_final_sparsity,
            "pruning_begin_step":       self.pruning_begin_step,
            "pruning_end_step":         self.pruning_end_step,
            "pruning_frequency":        self.pruning_frequency,
            "dropout_rate":     self.dropout,
            "verbose":          self.verbose,

                 
        }
    
    def __init__(self,**kwargs):
        super().__init__()
        #default values
        self.verbose=True
        self.key_dim,self.query_dim,self.value_dim=0,0,0
        self.wq_is_wv,self.wk_is_wv=False,False
        self.use_key_bias,self.use_query_bias,self.use_value_bias=False,False,False
        self.num_heads=1
        self.concat_heads=True
        self.dropout=0.0
        self.pooling=None #use mask by default

        self.pruned_layers=[]

        for key,value in kwargs.items():
            if key=="wq_is_wk":         self.wq_is_wk=value
            if key=="wk_is_wv":         self.wk_is_wv=value
            if key=="key_dim":          self.key_dim=value
            if key=="use_key_bias":     self.use_key_bias=value
            if key=="query_dim":        self.query_dim=value
            if key=="use_query_bias":   self.use_query_bias=value
            if key=="value_dim":        self.value_dim=value
            if key=="use_value_bias":   self.use_value_bias=value
            if key=="num_heads":        self.num_heads=value
            if key=="concat_heads":     self.concat_heads=value
            if key=="dropout_rate":     self.dropout=value

            if key=="pruning":                   self.pruning=value
            if key=="pruning_schedule":          self.pruning_schedule=value
            if key=="pruning_initial_sparsity":  self.pruning_initial_sparsity=value
            if key=="pruning_final_sparsity":    self.pruning_final_sparsity=value
            if key=="pruning_begin_step":        self.pruning_begin_step=value
            if key=="pruning_end_step":          self.pruning_end_step=value
            if key=="pruning_frequency":         self.pruning_frequency=value

            if key=="pooling":          self.pooling=value 
            if key=="verbose":          self.verbose=value
        


    def build(self,input_shape):

        if self.pruning:
            if self.pruning_schedule=="PolynomialDecay": 
                pruning_params={"pruning_schedule":tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.pruning_initial_sparsity,
                                                                      final_sparsity=self.pruning_final_sparsity,
                                                                      begin_step=self.pruning_begin_step,
                                                                      end_step=self.pruning_end_step,
                                                                      frequency=self.pruning_frequency) }
            if self.pruning_schedule=="ConstantSparsity":
                pruning_params={"pruning_schedule":tfmot.sparsity.keras.ConstantSparsity (target_sparsity=self.pruning_initial_sparsity,
                                                                      begin_step=self.pruning_begin_step,
                                                                      end_step=self.pruning_end_step,
                                                                      frequency=self.pruning_frequency) }  

        edge_shape,atm_shape,_=input_shape

        #create trainable weight to mix the diagonal unitary matrix to the edge matrix
        #self.eps = self.add_weight(shape=(1,), initializer="random_normal", name="diagonal_eps_for_eConv_layer"+str(super().name)) # learneable weights used in a diagonal matrix that will be added to the edge matrix 
        self.eps = self.add_weight(shape=(1,), initializer="random_normal", name="diagonal_eps_for_eConv_layer"+str(super().name)) # learneable weights used in a diagonal matrix that will be added to the edge matrix 


        #create dense layers for generating query, key and value matrices (if use_bias==False, dense layer is equivalent to matrix multiplication)
        self.query_layers,self.key_layers,self.value_layers,self.attention_query,self.attention_key,self.global_pools=[],[],[],[],[],[]


        if self.key_dim==0: self.key_dim=atm_shape[-1]
        elif type(self.key_dim)==float: self.key_dim=int(np.ceil(atm_shape[-1]*self.key_dim))
        if self.query_dim==0: self.query_dim=self.key_dim
        elif type(self.query_dim)==float: self.query_dim=int(np.ceil(atm_shape[-1]*self.query_dim))
        if self.value_dim==0: self.value_dim=atm_shape[-1]
        elif type(self.value_dim)==float: self.value_dim=int(np.ceil(atm_shape[-1]*self.value_dim))


        for i in range(self.num_heads):

            if self.pruning:
                l=tfmot.sparsity.keras.prune_low_magnitude(
                                                prunable_Dense(self.value_dim,
                                                    use_bias=self.use_query_bias,
                                                    name=str(super().name)+"head_"+str(i)+"_value_layer"    ),
                                                **pruning_params  )
                self.value_layers.append([l,keras.layers.Dropout(self.dropout)])
                self.pruned_layers.append(l)
            else:
                self.value_layers.append([keras.layers.Dense(self.value_dim,
                                                        use_bias=self.use_query_bias,
                                                        name=str(super().name)+"head_"+str(i)+"_value_layer"    ),
                                    keras.layers.Dropout(self.dropout)])

            if self.wk_is_wv:
                self.key_layers.append([self.value_layers[-1][0],keras.layers.LeakyReLU(alpha=0.2)])
            else: 
                if self.pruning:
                    l=tfmot.sparsity.keras.prune_low_magnitude(
                                                    prunable_Dense(self.key_dim,
                                                        use_bias=self.use_query_bias,
                                                        name=str(super().name)+"head_"+str(i)+"_key_layer"    ),
                                                    **pruning_params  )
                    self.key_layers.append([l,keras.layers.LeakyReLU(alpha=0.2)])
                    self.pruned_layers.append(l)
                else:                               
                    self.key_layers.append([keras.layers.Dense(self.key_dim,
                                                        use_bias=self.use_query_bias,
                                                        name=str(super().name)+"head_"+str(i)+"_key_layer"    ),
                                            keras.layers.LeakyReLU(alpha=0.2)])

            if self.wq_is_wk:
                self.query_layers.append([self.key_layers[-1][0],keras.layers.LeakyReLU(alpha=0.2)])
            else:
                if self.pruning:
                    l=tfmot.sparsity.keras.prune_low_magnitude(
                                                    prunable_Dense( self.query_dim,
                                                        use_bias=self.use_query_bias,
                                                        name=str(super().name)+"head_"+str(i)+"_query_layer"    ), 
                                                    **pruning_params  )
                    self.query_layers.append([l,keras.layers.LeakyReLU(alpha=0.2)  ])
                    self.pruned_layers.append(l)
                else:
                    self.query_layers.append([keras.layers.Dense(self.query_dim,
                                                        use_bias=self.use_query_bias,
                                                        name=str(super().name)+"head_"+str(i)+"_query_layer"    ),
                                      keras.layers.LeakyReLU(alpha=0.2)])

            self.attention_query.append(keras.layers.Dense(1,name=str(super().name)+"head_"+str(i)+"_attention_query_layer" ))
            self.attention_key.append(keras.layers.Dense(1,name=str(super().name)+"head_"+str(i)+"_attention_key_layer"))

            #add global pools ("mask" pool does not have to be added) 
            if self.pooling=="GlobalAttentionPool": self.global_pools.append(spektral.layers.GlobalAttentionPool(channels=self.eGIN_output_shape,
                                                                            name=str(super().name)+"_head_"+str(i)+"_GlobalAttentionPool"))
            elif self.pooling=="GlobalAttnSumPool": self.global_pools.append(spektral.layers.GlobalAttnSumPool(name=str(super().name)+"_head_"+str(i)+"_GlobalAttentionSumPool"))
            elif self.pooling=="GlobalSumPool":     self.global_pools.append(spektral.layers.GlobalSumPool(name=str(super().name)+"_head_"+str(i)+"_GlobalSumPool"))                                                                                         
            elif self.pooling=="GlobalMaxPool":     self.global_pools.append(spektral.layers.GlobalMaxPool(name=str(super().name)+"_head_"+str(i)+"_GlobalMaxPool"))
                                                                                        
    def call(self,inputs):
        e,x_atm,mask=inputs

        diag_chngd=tf.linalg.diag_part(e)+self.eps
        e_mod=tf.linalg.set_diag(e,diag_chngd)

        n_nodes=x_atm.shape[-2]
        head_output=[]
        for i in range(self.num_heads):
            batch_size=tf.shape(x_atm)[0]
            n_nodes=tf.shape(x_atm)[1]

            q_vector= self.attention_query[i]( self.query_layers[i][1] ( self.query_layers[i][0](x_atm)) )
            q_vector=q_vector[:,:,0]
            q_vector=tf.repeat(q_vector,n_nodes,axis=1)
            q_vector=tf.reshape(q_vector,[batch_size,n_nodes,n_nodes])

            k_vector=self.attention_key[i](self.key_layers[i][1] ( self.key_layers[i][0](x_atm)) )[:,:,0]  
            k_vector=tf.repeat(k_vector,n_nodes,axis=1)
            k_vector=tf.reshape(k_vector,[batch_size,n_nodes,n_nodes])
            k_vector=tf.transpose(k_vector,[0,2,1])

            attn_matrix=tf.math.multiply((tf.math.add(q_vector,k_vector)),e_mod)
            attn_matrix=tf.math.tanh(attn_matrix)
            #attn_matrix=tf.linalg.normalize(attn_matrix,axis=-1)[0]
            norm_factor=tf.linalg.normalize(attn_matrix,axis=-1)[1]
            norm_factor=tf.add(0.0000001,norm_factor)
            norm_factor=tf.repeat(norm_factor,n_nodes)
            norm_factor=tf.reshape(norm_factor,[batch_size,n_nodes,n_nodes])
            attn_matrix=tf.divide(attn_matrix,norm_factor)

            v_matrix=self.value_layers[i][0](x_atm)
            v_matrix=self.value_layers[i][1](v_matrix)
            x=tf.linalg.matmul(attn_matrix,v_matrix)
            if self.pooling!="mask":
                x=self.global_pools[i](x)
            else: 
                x=(tf.transpose(x,[0,2,1]) @ mask[...,None])[...,0] #https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data

            head_output.append(x)

        if self.concat_heads: return tf.concat(head_output,axis=-1)
        else: return tf.reduce_sum(head_output,axis=0)

    def strip_pruning(self):
        for l in self.pruned_layers:
            l=l.layer
            #l=l.strip_pruning_wrapper()
            #tfmot.sparsity.keras.strip_pruning(l)



@keras.saving.register_keras_serializable()
#class eGIN_module (keras.layers.Layer):
class eGIN_module (keras.Model):

    def get_config(self):
        return  {  **super().get_config(),
            "output_shape":       self.eGIN_output_shape,
            "num_heads":          self.eGIN_num_heads,
            "use_bias":           self.eGIN_use_bias,
            "key_dim":            self.eGIN_key_dim,
            "KAN_gridsize":       self.KAN_gridsize,

            "mlp_shape":          self.eGIN_mlp_shape,
            "batchnorm":          self.eGIN_batchnorm,
            "activation":         self.eGIN_activation, 
            "dropout_rate":       self.eGIN_dropout_rate,
            "l1":                 self.eGIN_l1,
            "l2":                 self.eGIN_l2,

            "consecutive_units":  self.consecutive_units,
            "pooling":            self.pooling, 

            "pruning":          self.pruning,
            "pruning_schedule": self.pruning_schedule,
            "pruning_initial_sparsity": self.pruning_initial_sparsity,
            "pruning_final_sparsity":   self.pruning_final_sparsity,
            "pruning_begin_step":       self.pruning_begin_step,
            "pruning_end_step":         self.pruning_end_step,
            "pruning_frequency":        self.pruning_frequency,

            "verbose":                  self.verbose
            }

    def __init__(self,**kwargs):
        super().__init__()
        
        #default values:  
        self.KAN=False
        self.verbose=True
        self.consecutive_units=1

        self.global_pools=[]
        self.recombination_layer=[]

        self.eGIN_output_shape=None
        self.eGIN_num_heads=None
        self.eGIN_mlp_shape=None
        self.eGIN_activation=None
        self.eGIN_dropout_rate=0.0

        self.pruned_layers=[]
        self.KAN_layers=[]

        for key,value in kwargs.items():

            if key=="output_shape":       self.eGIN_output_shape=value
            if key=="num_heads":          self.eGIN_num_heads=value
            if key=="use_bias":           self.eGIN_use_bias=value
            if key=="key_dim":            self.eGIN_key_dim=value
            if key=="KAN_gridsize":       self.KAN_gridsize=value

            if key=="mlp_shape":          self.eGIN_mlp_shape=value
            if key=="batchnorm":          self.eGIN_batchnorm=value
            if key=="activation":         self.eGIN_activation=value 
            if key=="l1":                 self.eGIN_l1=value
            if key=="l2":                 self.eGIN_l2=value

            if key=="dropout_rate":       self.eGIN_dropout_rate=value
            if key=="pooling":            self.pooling=value 
            if key=="consecutive_units":  self.consecutive_units=value

            if key=="pruning":                   self.pruning=value
            if key=="pruning_schedule":          self.pruning_schedule=value
            if key=="pruning_initial_sparsity":  self.pruning_initial_sparsity=value
            if key=="pruning_final_sparsity":    self.pruning_final_sparsity=value
            if key=="pruning_begin_step":        self.pruning_begin_step=value
            if key=="pruning_end_step":          self.pruning_end_step=value
            if key=="pruning_frequency":         self.pruning_frequency=value

            if key=="verbose":            self.verbose=value

        
        #transforms monodimensional lists eGIN_mlp_shape and KAN_gridsize in bidimiensional lists if consecutive_units!=1 by repetition
        if type(self.eGIN_mlp_shape[0])==int or type(self.eGIN_mlp_shape[0])==float and isinstance(self.consecutive_units,int):
            self.eGIN_mlp_shape=[self.eGIN_mlp_shape for i in range(self.consecutive_units)]
        #repeat monodimensional KAN_gridsize or dropout_rate lists to match eGIN_mlp_shape dimensions
        if type(self.KAN_gridsize)==int or type(self.KAN_gridsize)==float:
            self.KAN_gridsize=[[self.KAN_gridsize for i in range(len(self.eGIN_mlp_shape[0]))] for j in range(len(self.eGIN_mlp_shape))]
        elif type(self.KAN_gridsize[0])==int or type(self.KAN_gridsize[0])==float and isinstance(self.eGIN_mlp_shape[0],int):
            self.KAN_gridsize=[self.KAN_gridsize for i in range(len(self.eGIN_mlp_shape[0]))]
        if type(self.eGIN_dropout_rate)==int or type(self.eGIN_dropout_rate)==float:
            self.eGIN_dropout_rate=[[self.eGIN_dropout_rate for i in range(len(self.eGIN_mlp_shape[0]))] for j in range(len(self.eGIN_mlp_shape))]
        elif type(self.eGIN_dropout_rate[0])==int or type(self.eGIN_dropout_rate[0])==float and isinstance(self.eGIN_mlp_shape[0],list):
            self.eGIN_dropout_rate=[self.eGIN_dropout_rate for i in range(len(self.eGIN_mlp_shape[0]))]
        
    def build(self,input_shape):

        if self.pruning:
            if self.pruning_schedule=="PolynomialDecay": 
                pruning_params={"pruning_schedule":tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.pruning_initial_sparsity,
                                                                      final_sparsity=self.pruning_final_sparsity,
                                                                      begin_step=self.pruning_begin_step,
                                                                      end_step=self.pruning_end_step,
                                                                      frequency=self.pruning_frequency) }
            if self.pruning_schedule=="ConstantSparsity":
                pruning_params={"pruning_schedule":tfmot.sparsity.keras.ConstantSparsity (target_sparsity=self.pruning_initial_sparsity,
                                                                      begin_step=self.pruning_begin_step,
                                                                      end_step=self.pruning_end_step,
                                                                      frequency=self.pruning_frequency) }  

        edge_shape,atm_shape,_=input_shape
        #create trainable weight to mix the diagonal unitary matrix to the edge matrix
        #self.eps = self.add_weight(shape=(1,), initializer="random_normal", name="diagonal_eps_for_eConv_layer"+str(super().name)) # learneable weights used in a diagonal matrix that will be added to the edge matrix 
        self.eps = self.add_weight(shape=(1,), initializer="random_normal", name="diagonal_eps_for_eConv_layer"+str(super().name)) # learneable weights used in a diagonal matrix that will be added to the edge matrix 


        #for using attention mechanism instead of MLP in GIN modules
        if self.eGIN_output_shape!=None and self.eGIN_num_heads!=None:
            info="creating eGIN layer with attention:"+str(self.eGIN_num_heads)+" heads, key_dim: "+str(self.eGIN_key_dim)+" and drop-out rate: "+str(self.eGIN_dropout_rate)+"pooling with: "+str(self.pooling)
            if isinstance(self.eGIN_key_dim,float): eGIN_key_dim=int(self.eGIN_key_dim*atm_shape[-1])
            if isinstance(self.eGIN_output_shape,float): eGIN_output_shape=int(self.eGIN_output_shape*atm_shape[-1])
            
            self.recombination_layer= [ keras.layers.MultiHeadAttention(output_shape=eGIN_output_shape,num_heads=self.eGIN_num_heads,
                                                        use_bias=self.eGIN_use_bias, dropout=self.eGIN_dropout_rate, 
                                                        key_dim=eGIN_key_dim,
                                                        name=str(super().name)+"_multhd_attn"+str(i) ) for i in range(self.consecutive_units) ]
        

        #for using MLP or KAN in GIN modules:
        elif self.eGIN_mlp_shape!=None and (self.eGIN_activation!=None or self.KAN):
            info="creating eGIN module with mlp size: "+str([list(mlp_size) for mlp_size in self.eGIN_mlp_shape])
            j=0
            for mlp_shape,KAN_gridsize,dropout_rate in zip(self.eGIN_mlp_shape,self.KAN_gridsize,self.eGIN_dropout_rate):
                j+=1
                #if the length of KAN_gridsize list is smaller than the length of MLP_shape, repeat last element of KAN_gridsize
                if len(KAN_gridsize)<len(mlp_shape):
                    for i in range(len(mlp_shape)-len(KAN_gridsize)): KAN_gridsize.append(KAN_gridsize[-1])
                info+="\n module including: "
                mlp_hidden_shape=[]
                #add to mlp_hidded_shape the number of neurons in next layers (the number of hidden layers is determined by the number 
                #of elements in each mlp_shape list  
                for i in range(0,len(mlp_shape)):
                    #if the list contain int numbers, use this number as the number of neurons
                    if type(mlp_shape[i])==int: 
                        mlp_hidden_shape.append(mlp_shape[i])
                    #if the list contain float, calculate the number of neurones multiplying by this number the number of atomic features  (ceil is used to approximate to next integer number)
                    elif type(mlp_shape[i])==float:
                        mlp_hidden_shape.append(int(np.ceil(atm_shape[-1]*mlp_shape[i]))) #referred to the first layer
                    #if the number of neurons is reduced to 1, there is no use in adding more layers:
                    if int(np.ceil(mlp_hidden_shape[-1]))==1: break 

                this_layer=[]
                for i,l in enumerate(mlp_hidden_shape):
                    #add layers
                    if self.pruning:
                        if KAN_gridsize[i]!=0:    #using DenseKAN  
                            layer=tfmot.sparsity.keras.prune_low_magnitude(
                                    prunable_DenseKAN(l,
                                                    grid_size=KAN_gridsize[i],
                                                    #kernel_regularizer=keras.regularizers.L1L2(l1=self.eGIN_l1,l2=self.eGIN_l2),
                                                    name=str(super().name)+"_denseKAN_layer_"+str(i)+"-"+str(j)    ),
                                                **pruning_params  )
                            self.KAN_layers.append(layer)
                            info+="\n  denseKAN layer with "+str(l)  +" units and gridsize:"+str(KAN_gridsize[i])
                        else:    #using conventional dense layer         
                            layer=tfmot.sparsity.keras.prune_low_magnitude(
                                                prunable_Dense(l,
                                                        activation=self.eGIN_activation,
                                                        kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                                                        kernel_regularizer=keras.regularizers.L1L2(l1=self.eGIN_l1,l2=self.eGIN_l2),
                                                        name=str(super().name)+"_dense_layer_"+str(i)+"-"+str(j)    ),
                                                    **pruning_params  )
                            info+="\n  dense layer with "+str(l)+" units"
                            
                        this_layer.append(layer)
                        self.pruned_layers.append(layer)
                    else:
                        if KAN_gridsize[i]!=0:   #using DenseKAN  
                            this_layer.append(tfkan.layers.DenseKAN(l,
                                                        grid_size=KAN_gridsize[i],
                                                        #activation=self.eGIN_activation,
                                                        #kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                                                        #kernel_regularizer=keras.regularizers.L1L2(l1=self.eGIN_l1,l2=self.eGIN_l2),
                                                        name=str(super().name)+"_denseKAN_layer_"+str(i)+"-"+str(j)    ))
                            self.KAN_layers.append(this_layer.append[-1])
                            info+="\n  denseKAN layer with "+str(l)  +" units and gridsize:"+str(KAN_gridsize[i])

                        else:     #using conventional dense layer 
                            this_layer.append(keras.layers.Dense(l,
                                                        activation=self.eGIN_activation,
                                                        kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                                                        kernel_regularizer=keras.regularizers.L1L2(l1=self.eGIN_l1,l2=self.eGIN_l2),
                                                        name=str(super().name)+"_dense_layer_"+str(i)+"-"+str(j)    ))
                            info+="\n  dense layer with "+str(l)+" units"
                    #add batchnorm
                    if self.eGIN_batchnorm: 
                        this_layer.append(keras.layers.BatchNormalization(name=str(super().name)+"_bacth_norm_"+str(i)+"-"+str(j)))
                        info+=" followed by batch-normalization"
                    #add dropout 
                    if KAN_gridsize[i]==0: #(not with KAN)
                        if dropout_rate[i]!=0:
                            if i<len(dropout_rate): 
                                this_layer.append( keras.layers.Dropout(dropout_rate[i],name=str(super().name)+"_dropout_"+str(i)+"-"+str(j))) 
                                info+=" followed by "+str(dropout_rate[i])+ " dropout"   
                            else: 
                                this_layer.append(keras.layers.Dropout(dropout_rate[-1],name=str(super().name)+"_dropout_"+str(i)+"-"+str(j)))
                                info+=" followed by "+str(dropout_rate[-1])+ " dropout"   


                self.recombination_layer.append(this_layer)
                
                info+="\n recombination: "+self.pooling
                #add global pools "mask" pool does not have to be added 
                if self.pooling=="GlobalAttentionPool": this_pool=spektral.layers.GlobalAttentionPool(channels=self.eGIN_output_shape,
                                                                name=str(super().name)+"_GlobalAttentionPool_"+str(i))
                elif self.pooling=="GlobalAttnSumPool": this_pool=spektral.layers.GlobalAttnSumPool(name=str(super().name)+"_GlobalAttentionSumPool_"+str(i)+"-"+str(j))
                elif self.pooling=="GlobalSumPool":     this_pool=spektral.layers.GlobalSumPool(name=str(super().name)+"_GlobalSumPool_"+str(i)+"-"+str(j))
                elif self.pooling=="GlobalMaxPool":     this_pool=spektral.layers.GlobalMaxPool(name=str(super().name)+"_GlobalMaxPool_"+str(i)+"-"+str(j))
                if self.pooling!="mask": self.global_pools.append(this_pool)


            #print what has been done  
            if self.verbose:  print (info)
                        
            #print ("len(self.recombination_layer):"+str(len(self.recombination_layer)))
            #for i,l in enumerate(self.recombination_layer): print ("len(self.recombination_layer["+str(i)+"]: "+str(len(self.recombination_layer[i])))

    def call(self,inputs):
        e,x_atm,mask=inputs

        transf_atom_matrix=None #list of matrices with the output of every pooling layer used
        #e_mod= tf.add(e, self.eps * tf.linalg.diag(tf.ones(e.shape[-1]))  )
        diag_chngd=tf.linalg.diag_part(e)+self.eps
        e_mod=tf.linalg.set_diag(e,diag_chngd)
        #e_mod=e

        #for the first unit the matrix that will be operated on is x_atm; next, the operations will be made on v (the resulting matrix after the first unit)
        v=x_atm
        for j in range(len(self.eGIN_mlp_shape) ):

            #transform the x_atm matrix with e_mod; x_atm is batch_size x n_atoms x n_features;  x_atm_mod is batch_size x n_atoms x n_features
            x_atm_mod=tf.matmul(e_mod,v)
            #apply the reduction layer:
            if self.eGIN_output_shape!=None and self.eGIN_num_heads!=None: #using attn
                v= self.recombination_layer[j](query=x_atm, key=x_atm_mod, value=x_atm_mod)   #change query, key and value ???

            elif self.eGIN_mlp_shape!=None and self.eGIN_activation!=None: #using Dense MLP or DenseKan MLP
                v=x_atm_mod
                for rl in self.recombination_layer[j]:
                    v= rl(v)

            #output from e_conv layer that is pooled, transformed to an scalar, and concatenated to x_eq. It consists on two parts:
            #global-pool the vector according to self.pooling:
            if self.pooling=="mask": 
                v_pooled=( tf.transpose(v,[0,2,1]) @ mask[...,None])[...,0]   #https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
            else:
                v_pooled=self.global_pools[j](v)
            if transf_atom_matrix==None: transf_atom_matrix=v_pooled
            else: transf_atom_matrix=keras.layers.Concatenate()([transf_atom_matrix,v_pooled])
        return transf_atom_matrix

    def strip_pruning(self):
        for l in self.pruned_layers:
            l=l.layer
            #l=l.strip_pruning_wrapper()
            #tfmot.sparsity.keras.strip_pruning(l)

    def update_grid_from_samples(self,x_batch):
        for l in self.KAN_layers: l.update_grid_from_samples(x_batch)


@keras.saving.register_keras_serializable()
class DenseModule (keras.Model):


    def get_config(self):
        
        return  {  **super().get_config(),
            "input_noise":              self.input_noise,            
            "output_noise":             self.output_noise,
            "layer_noise":              self.layer_noise,
            "dense_shape":              self.dense_shape,
            "batchnorm":                self.dense_batchnorm,
            "dropout_factor":           self.dropout_factor,  
            "activation":               self.activation,
            "feature_dropout_rate":     self.feature_dropout_rate,
            "dense_l1":                 self.l1,
            "dense_l2":                 self.l2,
            "use_skip_layer":           self.use_skip_layer,
            "skip_filter":              self.skip_filter,
            "skip_insert_point":        self.skip_insert_point,
            "return_one_value":         self.return_one_value,   #NEW
            "KAN_gridsize":             self.KAN_gridsize,

            "cancelout_activation":     self.cancelout_activation,
            "cancelout":                self.cancelout,
            "cancelout_l1":             self.cancelout_l1,
            "cancelout_l2":             self.cancelout_l2, 
            "leakygate":                self.leakygate,
            "leakygate_alpha":          self.leakygate_alpha,

            "pruning":                  self.pruning,
            "pruning_schedule":         self.pruning_schedule,
            "pruning_initial_sparsity": self.pruning_initial_sparsity,
            "pruning_final_sparsity":   self.pruning_final_sparsity,
            "pruning_begin_step":       self.pruning_begin_step,
            "pruning_end_step":         self.pruning_end_step,
            "pruning_frequency":        self.pruning_frequency,

            "reduction_axis":           self.reduction_axis,

            "verbose":                       self.verbose
            }


    def __init__(self,**kwargs):
        super().__init__()

        #default values
        self.input_noise=0.0
        self.output_noise=0.0
        self.layer_output_noise=0.0
        self.dense_shape=[1.0,0.5,0.25]
        self.dense_batchnorm=True
        self.dense_dropout_factor=0.2
        self.activation="relu"
        self.feature_dropout_rate=0.0
        self.dense_reg=keras.regularizers.L1L2(l1=0.0,l2=0.0)
        self.use_skip_layer=True
        self.skip_filter="leakygate" 
        self.skip_insert_point=3
        self.TANGOS=False
        self.cancelout=False
        self.cancelout_l1=0.0
        self.cancelout_l2=0.0
        self.leakygate=False
        self.pruning=False
        self.return_one_value=False
        self.verbose=True
        self.KAN_gridsize=[0,0,0,0,0,0,0]

        self.dlayers=[]
        self.skip_layers=[]
        self.return_one_value=True

        self.reduction_axis=-1

        self.pruned_layers=[]

        for key,value in kwargs.items():
            if key=="input_noise":              self.input_noise=value            
            if key=="output_noise":             self.output_noise=value
            if key=="layer_noise":              self.layer_noise=value
            if key=="dense_shape":              self.dense_shape=value
            if key=="batchnorm":                self.dense_batchnorm=True
            if key=="dropout_factor":           self.dropout_factor=value  
            if key=="activation":               self.activation=value
            if key=="feature_dropout_rate":     self.feature_dropout_rate=value
            if key=="dense_l1":                 self.dense_l1=value
            if key=="dense_l2":                 self.dense_l2=value
            if key=="use_skip_layer":           self.use_skip_layer=value
            if key=="skip_filter":              self.skip_filter=value
            if key=="skip_insert_point":        self.skip_insert_point=value
            if key=="return_one_value":         self.return_one_value=value
            if key=="KAN_gridsize":             self.KAN_gridsize=value

            if key=="TANGOS":                   self.TANGOS=value
            if key=="TANGOS_subsample":         self.TANGOS_subsample=value
            if key=="TANGOS_sp":                self.TANGOS_sp=value
            if key=="TANGOS_ort":               self.TANGOS_ort=value

            if key=="cancelout_activation":     self.cancelout_activation=value
            if key=="cancelout":                self.cancelout=value
            if key=="cancelout_l1":             self.cancelout_l1=value
            if key=="cancelout_l2":             self.cancelout_l2=value 

            if key=="leakygate":                self.leakygate=value
            if key=="leakygate_alpha":          self.leakygate_alpha=value

            if key=="pruning":                   self.pruning=value
            if key=="pruning_schedule":          self.pruning_schedule=value
            if key=="pruning_initial_sparsity":  self.pruning_initial_sparsity=value
            if key=="pruning_final_sparsity":    self.pruning_final_sparsity=value
            if key=="pruning_begin_step":        self.pruning_begin_step=value
            if key=="pruning_end_step":          self.pruning_end_step=value
            if key=="pruning_frequency":         self.pruning_frequency=value

            if key=="reduction_axis":            self.reduction_axis=value

            if key=="verbose":                   self.verbose=value

        #cancelout can add a penalty (optional) if self.cancelout_loss is True
        if self.cancelout_l1==0 and self.cancelout_l2==0: self.cancelout_loss=False  
        else:    self.cancelout_loss=True


        if self.TANGOS:
            if type(self.TANGOS_sp)==int:  self.TANGOS_sp=[self.TANGOS_sp]*80 #a list large enough so there is a value for each layer
            if type(self.TANGOS_ort)==int: self.TANGOS_ort=[self.TANGOS_ort]*80

    def build(self,input_shape):

        if  self.reduction_axis==2 or self.reduction_axis==-2: input_shape=input_shape[-2]
        else: input_shape=input_shape[-1]

        if self.cancelout: self.dlayers.append(CancelOut(activation=self.cancelout_activation,cancelout_loss=self.cancelout_loss,
                                                                lambda_1=self.cancelout_l1,lambda_2=self.cancelout_l2,name=super().name+"_cancelout_layer"))

        if self.leakygate: self.dlayers.append(keras.layers.LeakyReLU(self.leakygate_alpha,name=super().name+"_leakygate_activation"))

        if self.feature_dropout_rate!=0: self.dlayers.append(keras.layers.Dropout(self.feature_dropout_rate,name=super().name+"_feature_selection_dropout_layer"))

        if self.pruning:
            if self.pruning_schedule=="PolynomialDecay": 
                pruning_params={"pruning_schedule":tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.pruning_initial_sparsity,
                                                                      final_sparsity=self.pruning_final_sparsity,
                                                                      begin_step=self.pruning_begin_step,
                                                                      end_step=self.pruning_end_step,
                                                                      frequency=self.pruning_frequency) }
            if self.pruning_schedule=="ConstantSparsity":
                pruning_params={"pruning_schedule":tfmot.sparsity.keras.ConstantSparsity (target_sparsity=self.pruning_initial_sparsity,
                                                                      begin_step=self.pruning_begin_step,
                                                                      end_step=self.pruning_end_step,
                                                                      frequency=self.pruning_frequency) }                



        for i in range(0,len(self.dense_shape)):
 
            info,info_skip,info_batchnorm,info_noise,info_dropout="","","","",""
            if self.verbose: print("LAYER #:"+str(i)) 
            #regularizer:
            dense_reg=keras.regularizers.L1L2(l1=self.dense_l1,l2=self.dense_l1)

            #calculate the number of units of each dense layer
            if type(self.dense_shape[i])==int:   current_layer_shape=self.dense_shape[i]
            elif type(self.dense_shape[i])==float: current_layer_shape=int(np.ceil(input_shape*self.dense_shape[i]))      

            #create the dense layer for the skip connection
            if self.use_skip_layer and i+1==self.skip_insert_point:
                self.skip_layers=[]
                info_skip= "creating skip layer with shape: "+str(current_layer_shape)

                if self.skip_filter=="leakygate":
                    self.skip_layers.append(keras.layers.LeakyReLU(self.leakygate_alpha,name=super().name+"_skip_filter_leakygate"))
                    info_skip+=" and leakygate filter"

                elif self.skip_filter=="cancelout":
                    self.skip_layers.append(CancelOut(activation=self.cancelout_activation,cancelout_loss=self.cancelout_loss,
                                                                lambda_1=self.cancelout_l1,lambda_2=self.cancelout_l2,name=super().name+"_skip_filter_cancelout"))
                    info_skip+=" and cancelout filter"
                if self.pruning:
                    l=tfmot.sparsity.keras.prune_low_magnitude( 
                                                        prunable_Dense(current_layer_shape,activation=None,name=super().name+"_skip_layer_dense"),**pruning_params )
                    self.skip_layers.append(l)
                    self.pruned_layers.append(l)

                else: 
                    self.skip_layers.append(  keras.layers.Dense(current_layer_shape,use_bias=False,activation=None,name=super().name+"_skip_layer_dense"))

                    
                self.weight_skip_layer=self.add_weight(name=super().name+"_skip_layer_weight")

            #if the calculated size of the layer is larger than 1 and it is not the last layer, add dense layers
            if current_layer_shape>1 and i<len(self.dense_shape): 


                if self.pruning:
                    if self.KAN_gridsize[i]!=0: 
                        l=tfmot.sparsity.keras.prune_low_magnitude(
                                                    prunable_DenseKAN(current_layer_shape,grid_size=self.KAN_gridsize[i],
                                                    #kernel_regularizer=dense_reg,
                                                    name=super().name+"_KANdense_layer_"+str(i) ),**pruning_params  )
                        self.KAN_layers.append(l)
                    else:   l=tfmot.sparsity.keras.prune_low_magnitude(
                                                prunable_Dense(current_layer_shape,kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                                                kernel_regularizer=dense_reg,
                                                name=super().name+"_dense_layer_"+str(i) ),**pruning_params  )
                    self.dlayers.append(l)
                    self.pruned_layers.append(l)
                else:
                    if self.KAN_gridsize[i]!=0: 
                        self.dlayers.append(prunable_DenseKAN(current_layer_shape,grid_size=self.KAN_gridsize[i],
                                                    #kernel_regularizer=dense_reg,
                                                    name=super().name+"_KANdense_layer_"+str(i) ))
                        self.KAN_layers.append(self.dlayers[-1])

                    else: self.dlayers.append( prunable_Dense(current_layer_shape,kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                                                kernel_regularizer=dense_reg,name=super().name+"_dense_layer_"+str(i) ))                    
                if self.KAN_gridsize[i]!=0:
                    info+="creating KAN layer with shape: "+str(current_layer_shape)+" and grid size: "+str(self.KAN_gridsize[i])
                else: info+="creating dense layer with shape: "+str(current_layer_shape)+" and activation: "+self.activation

                #add noise layers

                if len(self.layer_noise)>i and self.layer_noise[i]!=0.0: 
                    self.dlayers.append(keras.layers.GaussianNoise(stddev=self.layer_noise[i],name=super().name+"_gaussian_noise_"+str(i)))
                    info_noise=" output noise ("+str(self.layer_noise[i])+")"
                #add activation layer:
                if self.activation=="relu": self.dlayers.append(keras.layers.ReLU(name=super().name+"_ReLU_"+str(i)))
                elif self.activation=="prelu": self.dlayers.append(keras.layers.PReLU(name=super().name+"_PReLU_"+str(i)))
                elif self.activation=="leaky-relu": self.dlayers.append(keras.layers.LeakyReLU(name=super().name+"_LeakyReLU_"+str(i)))                
                #add batch normalization; it should be after activation and before dropout, or before activation:  https://forums.fast.ai/t/where-should-i-place-the-batch-normalization-layer-s/56825/4
                if self.dense_batchnorm: 
                    self.dlayers.append(keras.layers.BatchNormalization(name=super().name+"_batchnorm_"+str(i)   ))
                    info_batchnorm=" batch normalization"
                #add dropout
                if self.KAN_gridsize[i]==0 or self.KAN_gridsize==[0,0,0]: #KAN does not support dropout
                    if self.dropout_factor!=0 and isinstance(self.dropout_factor,float):
                        self.dlayers.append(keras.layers.Dropout(self.dropout_factor,name=super().name+"_dropout_"+str(i))) 
                        info_dropout=" dropout ("+str(self.dropout_factor)+")"
                    elif isinstance(self.dropout_factor,list):
                        if len(self.dropout_factor)<i:
                            self.dlayers.append(keras.layers.Dropout(self.dropout_factor[i],name=super().name+"_dropout_"+str(i)))
                            info_dropout=" dropout ("+str(self.dropout_factor[i])+")"
                        else:
                            self.dlayers.append(keras.layers.Dropout(self.dropout_factor[-1],name=super().name+"_dropout_"+str(i)))
                            info_dropout=" dropout ("+str(self.dropout_factor[-1])+")"
                if self.verbose: 
                    print(info)
                    info2=[m for m in [info_dropout,info_batchnorm,info_noise ] if m!=""]
                    if len(info2)>0: print("layer includes: "+", ".join(info2)) 
                    if info_skip!="": print (info_skip)    
            #if the calculated size of the layer is smaller than 1 or it is the last layer, end up with a single neuron and no activation
            else:
                break

        #add the last layer with a single unit and no activation only if return_one_value is set to True
        if self.return_one_value:  self.dlayers.append(keras.layers.Dense(1,name=super().name+"_output_layer"))

    def call(self,inputs):
        
        #(values of eq. linear features,values of eq. features, atom features matrix ,adjacency matrix, edge matrices, mask,name)
        x = inputs

        #if the reduction is not on the last dimension, transpose input so the dimension to reduce is placed at the end
        if  self.reduction_axis==2 or self.reduction_axis==-2: x=tf.transpose(x,[0,2,1])

        #add noise to the inputs
        if self.input_noise!=0.0: x=keras.layers.GaussianNoise(stddev=self.input_noise)(x)
        
        #calculate the output of the leaky layer 
        if self.use_skip_layer: 
            x_skip=self.skip_layers[0](x)
            for l in self.skip_layers[1:]:x_skip=l(x_skip)

        if self.TANGOS==False:
            counter=0    
            for i in range(0,len(self.dlayers)-1):
                x=self.dlayers[i](x)
                if isinstance(self.dlayers[i],keras.layers.Dense) or str(type(self.dlayers[i]))=="<class 'tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper.PruneLowMagnitude'>": 
                    counter+=1
                if self.use_skip_layer and counter==self.skip_insert_point:
                    counter=10000000#to prevent further insertions 
                    x= x+x_skip*self.weight_skip_layer
        
        else: #calculate TANGOS regularization (see: https://openreview.net/pdf?id=n6H86gW8u0d)
            x_tensor=tf.convert_to_tensor(x, dtype=tf.float32)
            layers_results=[x]
            with tf.GradientTape(persistent=True) as my_tape:
                my_tape.watch(x_tensor)
                counter=0
                for i in range(0,len(self.dlayers)-1):
                    layers_results.append( self.dlayers[i](layers_results[-1])   )  
                    if isinstance(self.dlayers[i],keras.layers.Dense) or str(type(self.dlayers[i]))=="<class 'tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper.PruneLowMagnitude'>": 
                        counter+=1    
                    if self.use_skip_layer and counter==self.skip_insert_point: 
                        layers_results[-1]= layers_results[-1]+x_skip   
                        counter=1000000 #to prevent further insertions 

            x=layers_results[-1]
            gradients={}
            for i in range(0,len(self.dlayers)-1):
                if isinstance(self.dlayers[i],keras.layers.Dense):
                    jacobian=my_tape.batch_jacobian(layers_results[i+1],x_tensor,unconnected_gradients=tf.UnconnectedGradients.NONE)
                    #alternative (sometimes works better)
                    #jacobian=my_tape.jacobian(layers_results[i+1],x_tensor,unconnected_gradients=tf.UnconnectedGradients.NONE)
                    #jacobian=tf.linalg.diag_part ( tf.transpose(jacobian,[1,3,2,0]) )
                    #jacobian=tf.transpose(jacobian,[2,0,1]) #dimensions: batch, hidden state, input 
                    gradients[self.dlayers[i].name]=(jacobian)
            del my_tape

            L_specs,L_orths=[],[]  #one loss penalty per layer     
            for i,gk in enumerate(gradients.keys()):
                g=gradients[gk]
                L_specs.append(tf.reduce_mean(tf.math.abs(g))) 
                if any(self.TANGOS_ort):  #this is costly, so it will only be calculated if any of the weights is different from zero
                    if g.shape[-1]>self.TANGOS_subsample and self.TANGOS_subsample>0:
                        subsample_indexes=np.sort(np.random.choice( range(g.shape[-1]), self.TANGOS_subsample,replace=False ))
                        g=tf.gather(g,subsample_indexes,axis=2)

                    g_norm=tf.math.l2_normalize(g,axis=2)         #normalize vectors
                    g_cos_sim=g_norm@tf.transpose(g_norm,[0,2,1]) #get all scalar product using matrix multiplication; the values of the cos similarities are contained in the triangle blocks, excluding diagonal
                    g_cos_sim=tf.linalg.band_part(g_cos_sim,0,-1)-tf.linalg.band_part(g_cos_sim,0,0) #we are onlly interested in the upper triangular part (excluding diagonal) of that matrix
                    g_cos_sim=tf.math.reduce_mean(tf.math.abs(g_cos_sim)) #calculate absolute value and sum
                    L_orths.append(g_cos_sim)
                else: L_orths.append(0.0)


            if len(TANGOS_sp)<len(L_orths):
                for _ in range(len(TANGOS_sp),len(L_orths)): TANGOS_sp.append(TANGOS_sp[-1])
            if len(TANGOS_ort)<len(L_orths):
                for _ in range(len(TANGOS_ort),len(L_orths)): TANGOS_ort.append(TANGOS_ort[-1])
            loss=tf.reduce_sum(   tf.convert_to_tensor([TANGOS_sp[i]*L_specs[i]+TANGOS_ort[i]*L_orths[i] for i in range(len(L_orths))])   )
            self.add_loss(loss)
        
        #the final layer (it might be a single neuron with no activation if self.return_one_value==True or the output of the last activation layer)
        x=self.dlayers[-1](x)

        #if the reduction was not on the last dimension, transpose it back to the original form 
        if  self.reduction_axis==2 or self.reduction_axis==-2: x=tf.transpose(x,[0,2,1])
        return x

    def strip_pruning(self):
        for l in self.pruned_layers:
            l=l.layer
            #l=l.strip_pruning_wrapper()
            #tfmot.sparsity.keras.strip_pruning(l)




#from: https://gist.github.com/aeftimia/a5249168c84bc541ace2fc4e1d22a13e
# https://arxiv.org/pdf/2303.05506
#ortogonal constrain to force the input or output weights in a layer to be ortogonal
#useful for changing the order of vectors in multihead attention (is it needed?):
from keras.constraints import Constraint
from tensorflow.linalg import expm
class Orthogonal(Constraint):
    """Orthogonal weight constraint.
    Constrains the weights incident to each hidden unit
    to be orthogonal when there are more inputs than hidden units.
    When there are more hidden units than there are inputs,
    the rows of the layer's weight matrix are constrainted
    to be orthogonal.
    # Arguments
        axis: Axis or axes along which to calculate weight norms.
            `None` to use all but the last (output) axis.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
        orthonormal: If `True`, the weight matrix is further
            constrained to be orthonormal along the appropriate axis.
    """

    def __init__(self, axis=None, orthonormal=False):
        self.axis = axis
        self.orthonormal = orthonormal

    def __call__(self, w):
        if self.axis is None:
            self.axis = list(range(len(w.shape) - 1))
        elif type(self.axis) == int:
            self.axis = [self.axis]
        else:
            self.axis = np.asarray(self.axis, dtype='uint8')
        self.axis = list(self.axis)

        axis_shape = [w.shape[a] for a in self.axis]
        perm = [i for i in range(len(w.shape) - 1) if i not in self.axis]
        perm.extend(self.axis)
        #perm.append(len(w.shape) - 1)
        w = tf.transpose(w, perm=perm)
        shape = w.shape
        w = tf.reshape(w, [-1] + axis_shape + [shape[-1]])
        w = tf.map_fn(self.orthogonalize, w)
        w = tf.reshape(w, shape)
        w = tf.transpose(w, perm=np.argsort(perm))
        return w

    def orthogonalize(self, w):
        shape = w.shape 
        output_shape = int(shape[-1])
        input_shape = int(np.prod(shape[:-1]))
        final_shape = int(max(input_shape, output_shape))
        w_matrix = tf.reshape(w, (output_shape, input_shape))
        w_matrix = tf.pad(w_matrix,
                           tf.constant([
                               [0, final_shape - output_shape],
                               [0, final_shape - input_shape]
                           ]))
        upper_triangular = tf.linalg.band_part(w_matrix, 1, -1)
        antisymmetric = upper_triangular - tf.transpose(upper_triangular)
        rotation = expm(antisymmetric)
        w_final = tf.slice(rotation, [0,] * 2, [output_shape, input_shape])
        if not self.orthonormal:
            if input_shape >= output_shape:
                w_final = tf.matmul(w_final,
                                            tf.matrix_band_part(
                                                tf.slice(w_matrix,
                                                                 [0, 0],
                                                                 [input_shape, input_shape]),
                                                0, 0))
            else:
                w_final = tf.matmul(tf.matrix_band_part(
                                                tf.slice(w_matrix,
                                                                 [0, 0],
                                                                 [output_shape, output_shape]),
                                                0, 0), w_final)
        return tf.reshape(w_final, w.shape)

    def get_config(self):
        return {'axis': self.axis,
                'orthonormal': self.orthonormal}


#apply multihead self-attention mechanism to change the size of the last dimension of a matrix
@keras.saving.register_keras_serializable()
class skipped_attention(keras.layers.Layer):

    def get_config(self):
        return  {  **super().get_config(),
                "key_dim":               self.key_dim,  
                "num_heads":             self.num_heads, 
                "output_shape":          self.attn_output_shape,
                "use_skip":              self.use_skip,
                "self_attn_dropout":     self.self_atnn_layer_dropout,
                "batchnorm":             self.batchnorm,
                "non_ortogonal_penalty": self.non_ortogonal_penalty,
                "reduction_axis":        self.reduction_axis,
                "name":                  self.name,
                "verbose":               self.verbose  
            }

    def __init__(self,**kwargs):
        super().__init__()
        #default values:
        self.key_dim=1.0
        self.num_heads=1
        self.attn_output_shape=1.0
        self.use_skip=True
        self.self_atnn_layer_dropout=0.0
        self.batchnorm=False
        self.non_ortogonal_penalty=0.0
        self.reduction_axis=1
        self.verbose=True
        
        for key,value in kwargs.items():

            if key=="key_dim":               self.key_dim=value
            if key=="num_heads":             self.num_heads=value
            if key=="output_shape":          self.attn_output_shape=value
            if key=="use_skip":              self.use_skip=value
            if key=="self_attn_dropout":     self.self_atnn_layer_dropout=value
            if key=="batchnorm":             self.batchnorm=value
            if key=="non_ortogonal_penalty": self.non_ortogonal_penalty=value
            if key=="reduction_axis":        self.reduction_axis=value
            if key=="verbose":               self.verbose=value

    def calculate_non_ortogonal_penalty(self,penalty,x):
        x_norm=tf.math.l2_normalize(x,axis=2) #normalize vectors
        x_cos_sim=x_norm@tf.transpose(x_norm,[0,2,1]) #get all scalar product using matrix multiplication; the values of the cos similarities are contained in the triangle blocks, excluding diagonal
        x_cos_sim=tf.linalg.band_part(x_cos_sim,0,-1)-tf.linalg.band_part(x_cos_sim,0,0) #we are onlly interested in the upper triangular part (excluding diagonal) of that matrix
        x_cos_sim=tf.math.reduce_sum(tf.math.abs(x_cos_sim)) #calculate absolute value and sum 
        return tf.math.abs(x_cos_sim*penalty)       


    def build(self,input_shape):


        #key_dim of attention layer  is set with respect to the max dimension of the matrix that will be passed
        if type(self.key_dim)==int: key_dim=self.key_dim
        elif type(self.key_dim)==float: key_dim=int(np.ceil(self.key_dim*  max([input_shape[-1], input_shape[-2]]) ))

        #output_shape of the last dimension is also set with respect to the initial shape:
        if type(self.attn_output_shape)==int: output_shape=self.attn_output_shape
        if type(self.attn_output_shape)==float: 
            if self.reduction_axis==-2 or self.reduction_axis==2: output_shape=int(np.ceil(self.attn_output_shape*input_shape[-2]))
            elif self.reduction_axis==-1 or self.reduction_axis==1: output_shape=int(np.ceil(self.attn_output_shape*input_shape[-1]))
        #add self attention layer
        self.self_attn_layer=keras.layers.MultiHeadAttention( num_heads=self.num_heads,
                                                              key_dim=key_dim,
                                                              dropout=self.self_atnn_layer_dropout,
                                                              output_shape=output_shape,
                                                              name=super().name+"_self-attention_layer")
        if self.verbose:             
            print ("creating self-multihead-attention layer with "+str(self.num_heads)+
               " heads, dropout: "+str( self.self_atnn_layer_dropout)+
               " key_dim: "+str(key_dim)+
               " and output shape: "+str(output_shape)+
               " acting on axis: "+str(self.reduction_axis)) 
        
        #add skip layer (it reduces last dimension, just as self attention layer does but more directly to allow gradients to travel back)
        if self.use_skip:
            self.skip_layer=keras.layers.Dense(units=output_shape,use_bias=False, name=super().name+"_skip")
            if self.verbose:
                print ("creating skip layer for multihead attention with "+str(output_shape)+" units")

        #add batch normalization
        if self.batchnorm:  self.batchnorm_layer=keras.layers.BatchNormalization(name=super().name+"_batchnorm")
    
    def call(self,input):

        if self.reduction_axis==2 or self.reduction_axis==-2: 
            input=tf.transpose(input,[0,2,1])

        #process input through the skip layer
        if self.use_skip: skip_v=self.skip_layer(input)
        
        #process input through the self attention layer:
        attn_v=self.self_attn_layer(input,input)

        #add penalty if the vectors are not ortogonal: 
        if self.non_ortogonal_penalty!=0.0: 
            self.add_loss(   self.calculate_non_ortogonal_penalty(self.non_ortogonal_penalty,attn_v) )

        if self.batchnorm: attn_v=self.batchnorm_layer(attn_v)

        if self.use_skip: attn_v= attn_v+skip_v
        if self.reduction_axis==-2 or self.reduction_axis==2: attn_v=tf.transpose(attn_v,[0,2,1])
        return attn_v

          
   
