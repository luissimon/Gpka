#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-



import spektral
import json
import numpy as np
import pandas as pd
import joblib
import sklearn 
import os
import types
import copy
import keras 
import tensorflow as tf

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

np.random.seed(16)
tf.random.set_seed(16)


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
        if isinstance(obj, dict):
            new_dict={}
            for k in obj.keys(): new_dict[k]=obj[k]
            return new_dict
        return super(NpEncoder, self).default(obj)


class gpka_spektral_dataset(spektral.data.Dataset):
    from scipy.linalg import block_diag
    #file: the json file that contains the data
    #csv_file: file from which equilibrium properties are read.
    #atom_feature_keys, atom_vector_keys, equilibrium_keys, etc... the name of the properties that will be extracted from the json fila
    #if not given, all will be extracted from the json files.
    def __init__(self, file,
                 atom_feature_keys="",atom_vector_keys="", equilibrium_keys="", linear_equilibrium_keys="",label_key="pKa",
                 csv_file="",**kwargs):
        #path = "/Users/luissimon/Desktop/finalcamgrid" #os.getcwd()
        #print (self.path)
        self.file=file
        self.csv_file=csv_file
        with open (self.file,"r") as f: d=json.loads( f.readline()  )

        if atom_feature_keys!="": self.atom_feature_keys=atom_feature_keys
        elif atom_feature_keys=="" and   type(d)==dict and "feature_keys" in d.keys():
            self.atom_feature_keys=d["feature_keys"]
        else: atom_feature_keys=[]

        if atom_vector_keys!="": self.atom_vector_keys=atom_vector_keys
        elif atom_vector_keys=="" and type(d)==dict and "vector_keys" in d.keys():
            self.atom_vector_keys=d["vector_keys"]
        else: atom_vector_keys=[]

        # if a key is in both equilibrium keys and linear_equilibrium_keys, only put it in "linear_equilibrium_keys
        if equilibrium_keys !="" and linear_equilibrium_keys!="":           
            self.equilibrium_keys=[eq for eq in equilibrium_keys if eq not in linear_equilibrium_keys]
            self.linear_equilibrium_keys= linear_equilibrium_keys
        elif equilibrium_keys!="" and linear_equilibrium_keys=="": 
            self.equilibrium_keys=equilibrium_keys
            self.linear_equilibrium_keys=[]
        elif equilibrium_keys=="" and linear_equilibrium_keys!="":
            self.equilibrium_keys=[]
            self.linear_equilibrium_keys= linear_equilibrium_keys
        else: 
            self.equilibrium_keys=[]
            self.linear_equilibrium_keys=[]

        if label_key!="": self.label_key=label_key
 
        

        super().__init__(**kwargs)

    #this method must be implemented as my_spektral_dataset inherits from epekectra.data.Dataset, to instruct how to fill the dataset with data.
    def read(self):
        output = []
        masks = []
        with open (self.file,"r") as f: lines=f.readlines()[1:]
        for l in lines:
            data=json.loads(l)
            #in case that no csv file with pka values is given, "y" is set to the name of the compound
            output.append(spektral.data.graph.Graph(x=np.array(data['x']).transpose(1,0), #x is n_atoms x n_features
                                                    a=np.array(data['a']), 
                                                    y=np.array(data['y']),
                                                    #transpose is needed because in the json file is n_features x n_atoms x n_atoms
                                                    #and for spektral graphs n_atoms x n_atoms x n_features is needed 
                                                    e=np.array(data['e']).transpose(2,1,0)
                                                    ))
            output[-1].mask=np.array(data["mask"])
            output[-1].weighted_mask=np.array(data["weighted_mask"])
            
        if self.csv_file!="" and (self.equilibrium_keys!="" or self.linear_equilibrium_keys!=""):
            updated_output=[]
            eq_data=pd.read_csv(self.csv_file,low_memory=False,encoding='latin-1')
            eq_data.dropna(axis=0)
            eq_data.dropna()
            for g in output:
                try:
                    index=eq_data.index[eq_data["compn"]==g.y][0]
                except IndexError: index="na"
                if index!="na": 
                    #name of the graph is set to the "compn" in csv file
                    g.name=eq_data.loc[index]["compn"]
                    if self.equilibrium_keys!="": g.z=np.array([eq_data.loc[index][k] for k in self.equilibrium_keys if k in eq_data.columns])
                    if self.linear_equilibrium_keys!="": g.linear_z=np.array([eq_data.loc[index][k] for k in self.linear_equilibrium_keys if k in eq_data.columns])
                    #if it can be read from the csv file, "y" is substitued by label_key (defaults to "pKa") value
                    if eq_data.loc[index][self.label_key]!=None: g.y=eq_data.loc[index][self.label_key]
 
                    #g.mask=np.transpose(g.x)[self.mask_key_index]
                    #g.mask=masks[i]

                    updated_output.append(g)
                output=updated_output
        return output    

    def duplicate_graph(self,index):
        import copy
        new_graph=copy.deepcopy(self.graphs[index])
        self.graphs.append(new_graph)



    def train_test_split(self,test_size=0.1,stratify="",with_replacement=False,remove_stratify_feature=True,data_augmentation=1.0,train_indexes=[],test_indexes=[]):

        repeated_train_indexes=[]
        if len(train_indexes)==0 and len(test_indexes)==0:
            indexes=list(range(len(self.graphs)))
            if stratify!="":
                strates_dict={} 
                for i,g in enumerate(self.graphs):
                    stratify_value=str(g.z[self.equilibrium_keys.index(stratify)])

                    if stratify_value not in strates_dict.keys(): strates_dict[  stratify_value]=[i]
                    else: strates_dict[ stratify_value ].append(i)
            else: strates_dict={"one":indexes}
            test_indexes=[]
            train_indexes=[]
            repeated_train_indexes=[]
            if data_augmentation<=1.0: data_augmentation+=1.0
            for k in strates_dict.keys():
                size=int(test_size*len(strates_dict[k]))
                this_k_test_indexes=list(np.random.choice(strates_dict[k],size,replace=False))
                this_k_train_indexes=[i for i in strates_dict[k] if i not in this_k_test_indexes ]
                if with_replacement: this_k_repeated_train_indexes=list(np.random.choice(this_k_train_indexes,int(size*data_augmentation),replace=False))
                else: this_k_repeated_train_indexes=[]
                test_indexes+=this_k_test_indexes
                train_indexes+=this_k_train_indexes
                repeated_train_indexes+=this_k_repeated_train_indexes
            test_indexes=sorted(test_indexes,reverse=True)
            train_indexes=sorted(train_indexes,reverse=True)
            repeated_train_indexes=sorted(repeated_train_indexes,reverse=True)

        test_dataset=copy.deepcopy(self)
        train_dataset=copy.deepcopy(self)
        test_dataset.graphs,train_dataset.graphs=[],[]
        for i in test_indexes: test_dataset.graphs.append(copy.deepcopy(self.graphs[i]))
        for i in train_indexes: train_dataset.graphs.append(copy.deepcopy(self.graphs[i]))
        for i in repeated_train_indexes: 
            train_dataset.graphs.append(copy.deepcopy(self.graphs[i]))
            train_dataset.graphs[-1].name=train_dataset.graphs[-1].name+"rep" #name must be unique

        if remove_stratify_feature and stratify!="":
            train_dataset.drop_features(stratify)
            test_dataset.drop_features(stratify)

        return train_dataset,test_dataset

    
    def eliminate_equilibrium_features(self,features):
        if type(features)==str:
            remove_indexes=[self.equilibrium_keys.index(features)]
        elif type(features)==list and type(features[0])==str: 
            remove_indexes=[self.equilibrium_keys.index(f) for f in features  ]
        elif type(feature)==list and type(feature[0])==int: 
            remove_indexes=feature
        elif type(feature)==int: 
            remove_indexes=[feature]  
        for G in self.graphs: 
            G.z= np.array([zz for i,zz in enumerate(G.z) if i not in remove_indexes ])
            #g.linear_z=np.array([zz for i,zz in enumerate(G.linear_z) if i not in remove_indexes ])
        self.equilibrium_keys=[xx for i,xx in enumerate(self.equilibrium_keys) if i not in remove_indexes ]
    
    def keep_equilibrium_features(self,features):
        if type(features)==str:
            keep_indexes=[self.equilibrium_keys.index(features)]
        elif type(features)==list and type(features[0])==str: 
            keep_indexes=[self.equilibrium_keys.index(f) for f in features  ]
        elif type(features)==list and type(features[0])==int: 
            keep_indexes=features
        elif type(features)==int: 
            keep_indexes=[features]
        for G in self.graphs: 
            G.z= np.array([zz for i,zz in enumerate(G.z) if i in keep_indexes ])
            #G.linear_z= [zz for i,zz in enumerate(G.linear_z) if i in keep_indexes ]
        self.equilibrium_keys=[xx for i,xx in enumerate(self.equilibrium_keys) if i in keep_indexes ]

    def eliminate_linear_equilibrium_features(self,features):
        if type(features)==str:
            remove_indexes=[self.linear_equilibrium_keys.index(features)]
        elif type(features)==list and type(features[0])==str: 
            remove_indexes=[self.linear_equilibrium_keys.index(f) for f in features  ]
        elif type(features)==list and type(features[0])==int: 
            remove_indexes=features
        elif type(features)==int: 
            remove_indexes=[features]  
        for G in self.graphs: 
            g.linear_z=np.array([zz for i,zz in enumerate(G.linear_z) if i not in remove_indexes ])
        self.linear_equilibrium_keys=[xx for i,xx in enumerate(self.linear_equilibrium_keys) if i not in remove_indexes ]
    
    def keep_linear_equilibrium_features(self,features):
        if type(features)==str:
            keep_indexes=[self.linear_equilibrium_keys.index(features)]
        elif type(features)==list and type(features[0])==str: 
            keep_indexes=[self.linear_equilibrium_keys.index(f) for f in features  ]
        elif type(features)==list and type(features[0])==int: 
            keep_indexes=features
        elif type(features)==int: 
            keep_indexes=[features] 
        for G in self.graphs: 
            G.linear_z= [zz for i,zz in enumerate(G.linear_z) if i in keep_indexes ]
        self.linear_equilibrium_keys=[xx for i,xx in enumerate(self.linear_equilibrium_keys) if i in keep_indexes ]

    def eliminate_atom_features(self,features):
        if type(features)==str:
            remove_indexes=[self.atom_feature_keys.index(features)]
        elif type(features)==list and type(features[0])==str: 
            remove_indexes=[self.atom_feature_keys.index(f) for f in features  ]
        elif type(features)==list and type(features[0])==int: 
            remove_indexes=features
        elif type(features)==int: 
            remove_indexes=[features]
        for G in self.graphs: 
            GxT=G.x.transpose(1,0)
            GxT_selected=np.array([xx for i,xx in enumerate(GxT) if i not in remove_indexes ])
            G.x=GxT_selected.transpose(1,0) 
            #G.x= np.array([xx for i,xx in enumerate(Gx) if i not in remove_indexes ])
        self.atom_feature_keys=[xx for i,xx in enumerate(self.atom_feature_keys) if i not in remove_indexes ]

    def keep_atom_features(self,features):
        if type(features)==str:
            keep_indexes=[self.atom_feature_keys.index(features)]
        elif type(features)==list and type(features[0])==str: 
            keep_indexes=[self.atom_feature_keys.index(f) for f in features  ]
        elif type(features)==list and type(features[0])==int: 
            keep_indexes=features
        elif type(features)==int: 
            keep_indexes=[features]
        for G in self.graphs: 
            GxT=G.x.transpose(1,0)
            GxT_selected=np.array([xx for i,xx in enumerate(GxT) if i in keep_indexes])
            G.x=GxT_selected.transpose(1,0)   
        self.atom_feature_keys=[f for i,f in enumerate(self.atom_feature_keys) if i in keep_indexes]
        
    def keep_atom_vectors(self,vectors):
        if type(vectors)==str:
            keep_indexes=[self.atom_vector_keys.index(vectors)]
        elif type(vectors)==list and type(vectors[0])==str: 
            keep_indexes=[self.atom_vector_keys.index(f) for f in vectors  ]
        elif type(vectors)==list and type(vectors[0])==int: 
            keep_indexes=vectors
        elif type(vectors)==int: 
            keep_indexes=[vectors]
        for G in self.graphs: 
            #GeT is n_features x n_atoms x n_atoms
            GeT=G.e.transpose(2,1,0)
            GeT_selected=np.array([ee for i,ee in enumerate(GeT) if i in keep_indexes ])
            G.e=GeT_selected.transpose(2,1,0)
        self.atom_vector_keys=[f for i,f in enumerate(self.atom_vector_keys) if i in keep_indexes]

    def eliminate_atom_vectors(self,vectors):
        if type(vectors)==str:
            remove_indexes=[self.atom_vector_keys.index(vectors)]
        elif type(vectors)==list and type(vectors[0])==str: 
            remove_indexes=[self.atom_vector_keys.index(f) for f in vectors  ]
        elif type(vectors)==list and type(vectors[0])==int: 
            remove_indexes=vectors
        elif type(vectors)==int: 
            remove_indexes=[vectors]
        for G in self.graphs:
            GeT=G.e.transpose(2,1,0)
            GeT_selected=np.array([ee for i,ee in enumerate(GeT) if i not in remove_indexes ])
            G.e=GeT_selected.transpose(2,1,0)
        self.atom_vector_keys=[f for i,f in enumerate(self.atom_vector_keys) if i not in remove_indexes]

    #add a new atom_vector feature. vectors must have the same dimensions that the number of graphs in the dataset
    def add_atom_vector(self,vectors,name):
        self.atom_vector_keys.append(name)
        for G,v in zip(self.graphs,vectors):
            new_Ge=np.dstack((G.e,np.array(v)))
            G.e=new_Ge

    def add_atom_feature(self,features,name):
        self.atom_feature_keys.append(name)
        for G,f in zip(self.graphs,features):
            #x is n_atoms x n_features
            new_x=np.column_stack((G.x,np.array(f)))
            G.x=new_x
            
    def add_equilibrium_feature(self,features,name):
        self.equilibrium_keys.append(name)
        for G,f in zip(self.graphs,features): G.z.append(f)
            
    
    def drop_features(self,features):
        if type(features)==str: features=[features]
        atom_features=[f for f in features if f in self.atom_feature_keys]
        atom_vectors=[f for f in features if f in self.atom_vector_keys]
        equilibrium_features=[f for f in features if f in self.equilibrium_keys or f in self.linear_equilibrium_keys]
        linear_equilibrium_features=[f for f in features if f in self.linear_equilibrium_keys]
        if len(atom_features)>0: self.eliminate_atom_features(atom_features)
        if len(atom_vectors)>0:  self.eliminate_atom_vectors(atom_vectors)
        if len(equilibrium_features)>0: self.eliminate_equilibrium_features(equilibrium_features)
        if len(linear_equilibrium_features)>0: self.eliminate_linear_equilibrium_features(linear_equilibrium_features)
            
    def keep_features(self,features):
        atom_features=[f for f in features if f in self.atom_feature_keys]
        atom_vectors=[f for f in features if f in self.atom_vector_keys]
        equilibrium_features=[f for f in features if f in self.equilibrium_keys]
        linear_equilibrium_features=[f for f in features if f in self.linear_equilibrium_keys]
        if len(atom_features)>0: self.keep_atom_features(atom_features)
        if len(atom_vectors)>0:  self.keep_atom_vectors(atom_vectors)
        if len(equilibrium_features)>0: self.keep_equilibrium_features(equilibrium_features)
        if len(linear_equilibrium_features)>0:self.keep_linear_equilibrium_features(linear_equilibrium_features)

    
    #retrieve a graph from the dataset using the name
    def get_graph_from_name(self,name):
        for g in self.graphs:
            if g.name==name: return g

    def drop_graph_from_name(self,name): #it can also be done using filter function of the dataset
        if type(name)==int: del(self.graphs[name]); return  #it is not safe to let "nam" be a list of ints, but it can be an int
        elif type(name)==str: name=[name]
        for n in name:
            for g in self.graphs:
                if g.name==n: self.graphs.remove(g)

    def get_values_of_feature(self,feature):
        values=[]
        if feature in self.label_key:
            values=[G.y for G in self.graphs]
        if feature in self.equilibrium_keys:
            index=self.equilibrium_keys.index(feature)
            values=[ G.z[index] for G in self.graphs]
        if feature in self.linear_equilibrium_keys:
            index=self.linear_equilibrium_keys.index(feature)
            values=[ G.linear_z[index] for G in self.graphs]
        if feature in self.atom_feature_keys:
            index=self.atom_feature_keys.index(feature)
            values=[ G.x.T[index] for G in self.graphs]
        if feature in self.atom_vector_keys:
            index=self.atom_vector_keys.index(feature)
            values=[ G.e.transpose(2,1,0)[index] for G in self.graphs]
        if feature=="name":
            values=[G.name for G in self.graphs]
        return values

    def get_index_of_eq_feature(self,features):
        if type(features)==str: features=[features]
        return [self.equilibrium_keys.index(f) for f in features  ]

    def get_index_of_atom_feature(self,features):
        if type(features)==str: features=[features]
        return [self.atom_feature_keys.index(f) for f in features  ]
 
    def get_index_of_atom_vector(self,features):
        if type(features)==str: features=[features]
        return [self.atom_vector_keys.index(f) for f in features  ]    
    
    def from_atom_vector_to_atom_feature(self,vector_feature_name,atom_feature_name="",mode="diag"):
        index=self.atom_vector_keys.index(vector_feature_name)
        if atom_feature_name=="": atom_feature_name=vector_feature_name+"_new"
        if atom_feature_name not in self.atom_feature_keys:
            self.atom_feature_keys.append(atom_feature_name)
            for G in self.graphs:
                v=G.e.transpose(2,1,0)[index]
                if mode=="diag": vv=np.diag(v)
                if mode=="sum": vv=np.sum(v,axis=1)
                if mode=="max": vv=np.max(v,axis=1)
                if mode=="min": vv=np.min(v,axis=1)
                if mode=="abs_max": vv=np.max(np.abs(v),axis=1)
                if mode=="abs_min": vv=np.min(np.abs(v),axis=1)
                G.x=np.column_stack((G.x,vv.T))

    #multiply the atom features by a mask 
    #if the mask contains info about the "alpha" atom (for example, because it is build from the difference of H atoms bound to each
    #atom, that is different from 0 only in atoms that looses a proton after deprotonation), so the list of values for each atom
    #is transformed to a single value corresponding to the alpha atom
    def aply_mask_to_atom_features(self,atom_feature_names,weighting=True,replace=True,suffix=""):
        if type(atom_feature_names)==str: atom_feature_names=[atom_feature_names]
        if type(atom_feature_names[0])==str: 
            atom_features_masked=[]
            atom_vectors_masked=[]
            for f in atom_feature_names:
                if f in self.atom_feature_keys:
                    index=self.atom_feature_keys.index(f)
                    atom_features_masked.append(f)
                    for G in self.graphs:
                        #G.x is n_nodes x n_features, GxT is n_features x n_nodes
                        GxT=G.x.transpose(1,0)
                        if weighting: GxT_masked=np.array(GxT[index]).dot(G.weighted_mask)
                        else: GxT_masked=np.array(GxT[index]).dot(G.mask)
                        G.z=np.append(G.z,GxT_masked)
                    self.equilibrium_keys.append(f+suffix)
                        
                elif f in self.atom_vector_keys:
                    index=self.atom_vector_keys.index(f)
                    atom_vectors_masked.append(f)
                    for G in self.graphs:
                        #e is n_atoms x n_atoms x n_features, GeT is n_features x n_atoms x n_atoms
                        GeT=G.e.transpose(2,1,0)
                        GeT_diag=np.diagonal(GeT[index])                     
                        if weighting: Ge_masked=np.diagonal(GeT[index]).dot(G.weighted_mask)
                        else: Ge_masked=np.diagonal(GeT[index]).dot(G.mask)
                        G.z=np.append(G.z,Ge_masked)
                    self.equilibrium_keys.append(f+suffix)                        
            if replace:
                if len(atom_features_masked)>0: self.eliminate_atom_features(atom_features_masked)
                
                if len(atom_vectors_masked)>0: self.eliminate_atom_vectors(atom_vectors_masked)


    #calculate the analogous of the adjacency matrix but with 1 only in atoms separated n bonds away 
    #unlike powers of adjacency matrix, elements are 0 for atoms connected with few of n bonds, or if the paths involves
    #"returning" from one bond already used
    def n_bonds_away_matrix_of_graph(self,graph,n):
        import networkx as nx
        #find paths in Graph G from node u of length n; uses networkx graphs
        #https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
        def findPaths(G,u,n):
            if n==0: return [[u]]
            paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
            return paths 
        #build a new nx.graph from adjacency matrix
        nxG = nx.from_numpy_array(graph.a)
        connections=[]
        for node in nxG: connections.append([p[-1] for p in findPaths(nxG,node,n)]) 
        #build the matrix from the connection list
        n_bonds_away_matrix_bonds=np.zeros_like(graph.a)
        for i,a in enumerate(connections): 
            for b in a: n_bonds_away_matrix_bonds[i,b]=1
        return n_bonds_away_matrix_bonds
            


    def add_n_bonds_away_matrix(self,n,name):
        n_bonds_away_matrices=[self.n_bonds_away_matrix_of_graph(G,n) for G in self.graphs]
        self.add_atom_vector(n_bonds_away_matrices,name)
        


    
    #determine the features of atoms separated n bonds away of each atom, and apply mask if required so the atoms in
    #alpha, beta, etc atoms are converted to eq. features
    def features_n_bonds_away(self,suffixes=[""],ns=[],apply_mask=True,weighting=False,exclude_ending=""):
        
        if type(ns)==int: ns=[n]
        if type(suffixes)==str: suffixes=[suffixes]
        if ns==[]: 
            for s in suffixes:
                if s=="alpha" or s=="A" or s=="a": ns.append(0)
                elif s=="beta" or s=="B" or s=="b": ns.append(1)
                elif s=="gamma" or s=="C" or s=="c" or s=="g" or s=="G": ns.append(2)
                elif s=="delta" or s=="D" or s=="d": ns.append(3)
                elif s=="epsilon" or s=="E" or s=="e": ns.append(4)
                elif s=="zeta" or s=="F" or s=="f" or s=="Z" or s=="z": ns.append(5)
                elif s=="Eta" or s=="H" or s=="h": ns.append(6)
        if len(ns)!=len(suffixes): suffixes=suffixes*len(ns)                   

        #exceptions 
        exclude_features=[]
        exclude_features_indexes=[]
        if type(exclude_ending)==str: exclude_ending=[exclude_ending]
        if exclude_ending!=[""]:
            exclude_features=[f for f in self.atom_feature_keys if any( [f.endswith(e) for e in exclude_ending])]
            exclude_features_indexes=[i for i,f in enumerate(self.atom_feature_keys) if any( [f.endswith(e) for e in exclude_ending])]

        #add the new features to the dataset
        for G in self.graphs:
            new_x=[G.x]
            pruned_x=np.delete(G.x,exclude_features_indexes,1)
            for n in ns:
                a_n_bonds=self.n_bonds_away_matrix_of_graph(G,n)
                new_x.append(a_n_bonds @ pruned_x)
                #new_x=np.hstack((new_x,additional_x))             
            G.x=np.hstack(new_x)
        #add the new keys to the dataset
        new_keys=[]  
        for suffix in suffixes:
            if suffix=="": suffix="_"+str(n)+"bonds"
            new_keys+=[key+"_"+suffix for key in self.atom_feature_keys if key not in exclude_features]
        self.atom_feature_keys=self.atom_feature_keys+new_keys 

        #if required, apply mask to transorm atom features to equilibrium features.
        if apply_mask: self.aply_mask_to_atom_features(new_keys,weighting=weighting)

    #given a matrix x, substitutes each element xij by sign(xij)*sparsemax(abs(xij))
    def sparsemax_vector_matrix(self,edge_matrix,sign=""):

        #first, zeroes all elements of the original matrix depending of "sign" (or leave it unmodified)
        if (type(sign)=="str" and (sign=="+" or sign=="pos")) or (type(sign)==int and sign>0): sign="pos"            
        if (type(sign)=="str" and (sign=="-" or sign=="neg")) or (type(sign)==int and sign<0): sign="neg"
        #if sign=="pos" :  edge_matrix=np.where(edge_matrix>0,edge_matrix,0)
        #elif sign=="neg":  edge_matrix=np.where(edge_matrix<0,edge_matrix,0)
            
        #z=np.abs([zz for zz in edge_matrix.flatten() if zz!=0.0])
        #print (z)
        z=edge_matrix.flatten()
        # step 1
        z_sorted = z[np.argsort(-z)]
        #print (z_sorted)        
        # step 2
        col_range = (np.arange(len(z))+1)
        ks = (1 + col_range * z_sorted) > np.cumsum(z_sorted)
        k_z = np.max(col_range * ks)    
        # step 3
        tau_z = (z_sorted[:k_z].sum()-1)/k_z
        #print (tau_z)
        # step 4
        np.maximum(z-tau_z, 0)
        #print ("sign")
        #print (np.sign(x))
        #print ("max")
        #print (np.maximum(x-tau_z,0))
        sparse_matrix=np.sign(edge_matrix)*np.maximum(np.abs(edge_matrix)-tau_z,0)
        if sign=="pos"   : return np.where(sparse_matrix>0,sparse_matrix,0.0)
        elif sign=="neg" : return np.where(sparse_matrix<0,-sparse_matrix,0.0)
        else: return sparse_matrix

    #find out the number of microeqs analyzing the bigest block in an edge matrix with
    def number_of_microeqs(self,matrix):
        #possible block sizes can only be exact divisors of the matrix dimensions
        possible_block_sizes=[  int(matrix[0].shape[-1]/i)  for i in  range(matrix[0].shape[-1],0,-1)  if matrix[0].shape[-1]%i==0   ] 
        for i,_ in enumerate(possible_block_sizes):
            j=possible_block_sizes[i]
            if np.any(matrix[:j,j:])==False and np.any(matrix[:-j,-j:])==False: return possible_block_sizes[i-1]
        return 1

    
    def add_sparsemax_matrix(self,vector_feature_key,name="",sign=""):
        from scipy.linalg import block_diag
        
        if (type(sign)=="str" and (sign=="+" or sign=="pos")) or (type(sign)==int and sign>0): sign="pos"
        if (type(sign)=="str" and (sign=="-" or sign=="neg")) or (type(sign)==int and sign<0): sign="neg"
            
        if name=="" and sign=="": name= vector_feature_key+"_sparsemax"
        elif  name=="" and sign=="pos": name= vector_feature_key+"_sparsemax_pos"
        elif  name=="" and sign=="neg": name= vector_feature_key+"_sparsemax_neg"
            
        if type(vector_feature_key)==str: feature_index=self.atom_vector_keys.index(vector_feature_key)
        elif type(vector_feature_key)==int: feature_index=vector_feature_key
            
        sparse_max_matrices=[]
        for G in self.graphs:
            matrix=G.e.transpose(2,1,0)[feature_index]
            #for disjoint graphs, each block will be treated separately and later will be concatenated
            block_size=self.number_of_microeqs(matrix)
            n_blocks=int(matrix.shape[-1]/block_size)
            blocks=[ matrix[b*block_size:(b+1)*block_size,b*block_size:(b+1)*block_size] for b in range(n_blocks)]
            sparse_max_matrix=self.sparsemax_vector_matrix(blocks[0],sign=sign)
            for block in blocks[1:]:
                sparse_max_block=self.sparsemax_vector_matrix(block,sign=sign)
                sparse_max_matrix= block_diag(sparse_max_matrix,sparse_max_block)
            sparse_max_matrices.append(sparse_max_matrix)
        self.add_atom_vector(sparse_max_matrices,name)
        
    def minmax_vector_matrix(self,edge_matrix,sign=""):                    
        if (type(sign)=="str" and (sign=="+" or sign=="pos")) or (type(sign)==int and sign>0): sign="pos"            
        if (type(sign)=="str" and (sign=="-" or sign=="neg")) or (type(sign)==int and sign<0): sign="neg"
        if sign=="pos" :  edge_matrix=np.where(edge_matrix>0,edge_matrix,0)
        elif sign=="neg":  edge_matrix=-1*np.where(edge_matrix<0,edge_matrix,0)  
        max,min=np.abs(np.max(edge_matrix.flatten())),np.abs(np.min(edge_matrix.flatten()))
        if (min!=0.0 or max!=0.0) and max!=min: 
            return (edge_matrix-min)/(max-min)
        else: 
            return edge_matrix

    def add_minmax_matrix(self,vector_feature_key,name="",sign=""):
        from scipy.linalg import block_diag

        if (type(sign)=="str" and (sign=="+" or sign=="pos")) or (type(sign)==int and sign>0): sign="pos"
        if (type(sign)=="str" and (sign=="-" or sign=="neg")) or (type(sign)==int and sign<0): sign="neg"       

        if name=="" and sign=="": name= vector_feature_key+"_minmax"
        elif  name=="" and sign=="pos": name= vector_feature_key+"_minmax_pos"
        elif  name=="" and sign=="neg": name= vector_feature_key+"_minmax_neg"
        if type(vector_feature_key)==str: feature_index=self.atom_vector_keys.index(vector_feature_key)
        elif type(vector_feature_key)==int: feature_index=vector_feature_key
        minmax_matrices=[]
        for G in self.graphs:
            matrix=G.e.transpose(2,1,0)[feature_index]
            #for disjoint graphs, each block will be treated separately and later will be concatenated
            block_size=self.number_of_microeqs(matrix)
            n_blocks=int(matrix.shape[-1]/block_size)
            blocks=[ matrix[b*block_size:(b+1)*block_size,b*block_size:(b+1)*block_size] for b in range(n_blocks)]
            minmax_matrix=self.minmax_vector_matrix(blocks[0],sign=sign)
            for block in blocks[1:]:
                max_min_block=self.minmax_vector_matrix(block,sign=sign)
                minmax_matrix=block_diag(minmax_matrix,max_min_block)
            minmax_matrices.append(minmax_matrix)
        self.add_atom_vector(minmax_matrices,name)


    #OLD OLD OLD OLD OLD OLD OLD OLD
    def sparsemax_atom_vector(self,vector_feature_names):
        from scipy.linalg import block_diag
        #given a matrix x, substitutes each element xij by sign(xij)*sparsemax(abs(xij)) 
        #using code from: https://medium.com/deeplearningmadeeasy/sparsemax-from-paper-to-code-351e9b26647b
        
        def my_sparsemax(x):
        
            z=np.abs(x.flatten())
            # step 1
            z_sorted = z[np.argsort(-z)]
            #print (z_sorted)
            
            # step 2
            col_range = (np.arange(len(z))+1)
            ks = (1 + col_range * z_sorted) > np.cumsum(z_sorted)
            k_z = np.max(col_range * ks)
            
            # step 3
            tau_z = (z_sorted[:k_z].sum()-1)/k_z
            #print (tau_z)
            
            # step 4
            np.maximum(z-tau_z, 0)
            #print ("sign")
            #print (np.sign(x))
            #print ("max")
            #print (np.maximum(x-tau_z,0))
            return (np.sign(x)*np.maximum(np.abs(x)-tau_z,0))

        if type(vector_feature_names)==str: vector_feature_names=[vector_feature_names]
        for vector_feature_name in vector_feature_names:
            index=self.atom_vector_keys.index(vector_feature_name)
            for G in self.graphs:
                ee=G.e.transpose(2,1,0) #unroll the e vector so a particular vector feature can be retrieved 
                v=ee[index] #retrieve it
                vv=my_sparsemax(v) #transform the matrix
                ee[index]=vv #substitute only the matrix corresponding to the vector feature
                G.e=ee.transpose(2,1,0) #roll the e vector again and substitute the current method.
                 
    def eq_features_to_pd_series(self,features=[],include_name=True,include_label=True,include_correct_name=True):
        import pandas as pd
        d={}
        if include_name: d["compn"]=self.get_values_of_feature("name")
        if include_label: d["pKa"]=self.get_values_of_feature("pKa")  
        if include_correct_name: d["correct name"]=self.get_values_of_feature("correct name")  
  
        if len(features)==0: features=self.linear_equilibrium_keys+self.equilibrium_keys
        for f in features: d[f]=self.get_values_of_feature(f)
        data=pd.DataFrame.from_dict(d)
        return data

class acbase_BatchLoader(spektral.data.loaders.BatchLoader):

    def __init__(self,dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    #returns: 
    # (values of eq. linear features,values of eq. features, atom features matrix ,adjacency matrix, edge matrices, masks, name of the equilibriums in the bath (for debuggin')),
    # labels (pka)
    def collate(self,batch):
        c=super().collate(batch)
        
        n_atoms=len(c[0][0][0])
        masks=[]
        #zero-pad mask to equalize the number of elements within each batch
        for g in batch:
            m=np.zeros(n_atoms)
            for i,_ in enumerate(g.mask):m[i]=g.mask[i]
            masks.append(m)
        
        #masks=[g.mask for g in batch]
        return tuple( [np.array([g.linear_z for g in batch])]+[np.array([g.z for g in batch])]+
                     list(c[0])+[np.array(masks)]+[np.array([g.name for g in batch])] ),c[1]


