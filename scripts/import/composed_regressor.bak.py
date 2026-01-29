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
import math
import xgboost


class Model_not_fitted_Exception(Exception): 
    """ exception for linear regressor fit not converged"""


#from: https://stackoverflow.com/questions/44333573/feature-importances-bagging-scikit-learn
from sklearn.ensemble import BaggingRegressor
class myBaggingRegressor(BaggingRegressor):

    def fit(self, X, y,sample_weight=None):
        fitd = super().fit(X, y,sample_weight=sample_weight)
        # need to pad features?
        if self.bootstrap_features!=True:
            if self.max_features == 1.0:
                # compute feature importances or coefficients
                if hasattr(fitd.estimators_[0], 'feature_importances_'):
                    self.feature_importances_ =  np.mean([est.feature_importances_ for est in fitd.estimators_], axis=0)
                else:
                    self.coef_ =  np.mean([est.coef_ for est in fitd.estimators_], axis=0)
                    self.intercept_ =  np.mean([est.intercept_ for est in fitd.estimators_], axis=0)
                    self.feature_importances_ =[1/self.n_features_in_]*self.n_features_in_
            else:
                # need to process results into the right shape
                coefsImports = np.empty(shape=(self.n_features_in_, self.n_estimators), dtype=float)
                coefsImports.fill(np.nan)
                if hasattr(fitd.estimators_[0], 'feature_importances_'):
                    # store the feature importances
                    for idx, thisEstim in enumerate(fitd.estimators_):
                        coefsImports[fitd.estimators_features_[idx], idx] = thisEstim.feature_importances_
                    # compute average
                    self.feature_importances_ = np.nanmean(coefsImports, axis=1)
                else:
                    # store the coefficients & intercepts
                    self.intercept_ = 0
                    for idx, thisEstim in enumerate(fitd.estimators_):
                        coefsImports[fitd.estimators_features_[idx], idx] = thisEstim.coefs_
                        self.intercept += thisEstim.intercept_
                    # compute
                    self.intercept /= self.n_estimators
                    # average
                    self.coefs_ = np.mean(coefsImports, axis=1)   
                    self.feature_importances_ =[1/self.n_features_in_]*self.n_features_in_             
        else:
            self.feature_importances_ =[1/self.n_features_in_]*self.n_features_in_
        return fitd


class composed_regressor(sklearn.base.BaseEstimator):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    
    def set_params(self,**kwargs):

        for key,value in kwargs.items():
            if key=="linear_regressor":          self.linear_regressor_name=value
            if key=="dimensionality_reduction":  self.dimensionality_reduction_name=value
            if key=="non_linear_regressor":      self.non_linear_regressor_name=value               
            if key=="linear_attributes":         self.linear_attributes=value; 
            if key=="non_linear_attributes":     self.non_linear_attributes=value; 
            if key=="combination":               self.combination=value;
        
        #remove from non_linear_attributes any attribute that was already in linear_attributes (for safety)
        if len(self.linear_attributes)!=0:
            for lin_attr in self.linear_attributes:
                if lin_attr in self.non_linear_attributes: self.non_linear_attributes.remove(lin_attr)
        #provide default linear and non linear attributes:   #remove???
        if len(self.linear_attributes)==0: self.linear_attributes="delta gibbs free energy"
        if len(self.non_linear_attributes)==0: self.non_linear_attributes="nlf_atomic_surface"


        linear_params,dim_red_params,non_linear_params={},{},{}

        # using the current (probably the default) values of the parameters of the linear regressor, 
        # filter which of the arguments correspond to parameters for it.
        default_linear_params=self.lin_reg.get_params()       
        for default_param,default_value in default_linear_params.items():
            for param,value in kwargs.items():
                #arguments for the linear parameteres must start with "l_" or "lx"
                if param[:2]=="l_" and param[2:]==default_param:
                    if value!=None:
                        if type(default_value)==type("hola"):    linear_params[param[2:]]=str(value)                        
                        elif type(default_value)==type(1):       linear_params[param[2:]]=int(value)                        
                        elif type(default_value)==type(3.14159): linear_params[param[2:]]=float(value)
                    else:linear_params[param[2:]]=None
                #with the notation "lxparameter", the current (default) value is multiplied by the factor given in the arguments
                elif param[:2]=="lx" and param[2:]==default_param:
                    if type(default_value)==type(1):         linear_params[param[2:]]=int(value*default_value)
                    if type(default_value)==type(3.14159):   linear_params[param[2:]]=float(value*default_value)

        # using the current (probably the default) values of the parameters of the dimensionality reduction algorithm, 
        # filter which of the arguments correspond to parameters for it. 
        default_dimensionality_reduction_params=self.dim_red.get_params()
        for default_param,default_value in  default_dimensionality_reduction_params.items():
            for param,value in kwargs.items(): 
                #n_components can be an integer or a fractional number. If it is a fractional number, it will be interpreted as variance recovered by PCA or
                #as the fraction of features by other algorithms:
                if param=="dr_n_components" and type(value)==type(3.14159) and value<1.0 and value>0.0: #if n_components is a float in the range [0,1] 
                    if self.dimensionality_reduction_name=="PCA" and  self.dimensionality_reduction_name=="None":
                        dim_red_params["n_components"]=value
                    else: 
                        dim_red_params["n_components"]= round(len(self.non_linear_attributes)*value)

                else: #the rest of parameters will be treated without modification
                    #arguments for the non linear parameteres must start with "dr_" or "drx"
                    if param[:3]=="dr_" and param[3:]==default_param: 
                        if value!=None:
                            if type(default_value)==type("hola"):     dim_red_params[param[3:]]=str(value)                        
                            elif type(default_value)==type(1):        dim_red_params[param[3:]]=int(value)                        
                            elif type(default_value)==type(3.14159):  dim_red_params[param[3:]]=float(value)
                            elif default_value==None:                 dim_red_params[param[3:]]=value   #in this case, trust that the parameter is passed correctly
                        else: dim_red_params[param[3:]]=None
                    #with the notation "drxparameter", the current (default) value is multiplied by the factor given in the arguments
                    elif param[:3]=="drx" and param[3:]==default_param:
                        if type(default_value)==type(1):         dim_red_params[param[3:]]=int(value*default_value)
                        if type(default_value)==type(3.14159):   dim_red_params[param[3:]]=float(value*default_value)

        # using the current (probably the default) values of the parameters of the non linear regressor, 
        # filter which of the arguments correspond to parameters for it.                
        default_non_linear_params=self.non_lin_reg.get_params()
        for default_param,default_value in default_non_linear_params.items():
            for param,value in kwargs.items():
                #arguments for the non linear parameteres must start with "nl_" or "nlx"
                if param[:3]=="nl_" and param[3:]==default_param:
                    if value!=None:
                        if param=="nl_estimator": pass #?
                        elif type(default_value)==type("hola"):   non_linear_params[param[3:]]=str(value)                        
                        elif type(default_value)==type(1):        non_linear_params[param[3:]]=int(value)                        
                        elif type(default_value)==type(3.14159):  non_linear_params[param[3:]]=float(value)
                    else: non_linear_params[param[3:]]=None
                #with the notation "lxparameter", the current (default) value is multiplied by the factor given in the arguments
                elif param[:3]=="nlx" and param[3:]==default_param:
                    if type(default_value)==type(1):         non_linear_params[param[3:]]=int(value*default_value)
                    elif type(default_value)==type(3.14159): non_linear_params[param[3:]]=float(value*default_value)
                        
        #set the parameters for the linear regressor and the non linear regressor

        self.lin_reg.set_params(**linear_params)
        #if self.dimensionality_reduction_name!="None":  self.dim_red.set_params(**dim_red_params)
        self.dim_red.set_params(**dim_red_params)
        self.non_lin_reg.set_params(**non_linear_params)
        super().set_params()
    
    def __init__(self,**kwargs):
        #default values, in case they are needed
        self.linear_regressor_name="ElasticNet"
        self.non_linear_regressor_name="GradientBoostingRegressor"
        self.dimensionality_reduction_name="None"
        self.linear_attributes=[]
        self.non_linear_attributes=[]
        self.use_shap=True
        self.fit_failure=False
        self.combination="sum"

        #the data set during fitting, so it can be reused:
        self.linear_X_fitted=[]
        self.non_linear_X_fitted=[]
        self.reduced_non_linear_X_fitted=[]
        self.nonlin_y_train_fitted=[]

        self.normalized_components_fitted=[]
        self.nl_feature_importances=[]
        self.shap_nl_values=[]
        self.shap_nl_feature_importances=[]


        
        #need this here, cannot wait to call set_params() because linear and non linear regressor objects are needed before
        for key,value in kwargs.items():
            if key=="linear_regressor":         self.linear_regressor_name=value
            if key=="dimensionality_reduction": self.dimensionality_reduction_name=value
            if key=="non_linear_regressor":     self.non_linear_regressor_name=value  
            if key=="use_shap":                 self.use_shap=value        

        #create the linear, dimensionality reduction algorithm, and non linear regressors depending on the arguments, 
        # using default values for the parameters (they will be changedset later)
        if self.linear_regressor_name==  "LinearRegression": self.lin_reg=sklearn.linear_model.LinearRegression()
        elif self.linear_regressor_name=="Lasso":            self.lin_reg=sklearn.linear_model.Lasso()   
        elif self.linear_regressor_name=="Ridge":            self.lin_reg=sklearn.linear_model.Ridge()  
        elif self.linear_regressor_name=="ElasticNet":       self.lin_reg=sklearn.linear_model.ElasticNet()
        elif self.linear_regressor_name=="HuberRegressor":   self.lin_reg=sklearn.linear_model.HuberRegressor()
        elif self.linear_regressor_name=="LinearSVR":        self.lin_reg=sklearn.svm.LinearSVR()
        elif self.linear_regressor_name=="RANSACRegressor":  self.lin_reg=sklearn.linear_model.RANSACRegressor()
        elif self.linear_regressor_name=="XGB-gblinear":     self.lin_reg=xgboost.XGBRegressor(booster="gblinear")
        #etc

        if self.dimensionality_reduction_name==        "PCA":               self.dim_red=sklearn.decomposition.PCA()
        elif  self.dimensionality_reduction_name==     "KernelPCA":         self.dim_red=sklearn.decomposition.KernelPCA()
        elif  self.dimensionality_reduction_name==     "FactorAnalysis":    self.dim_red=sklearn.decomposition.FactorAnalysis()
        elif  self.dimensionality_reduction_name==     "FastICA":           self.dim_red=sklearn.decomposition.FastICA()
        elif  self.dimensionality_reduction_name==     "NMF":               self.dim_red=sklearn.decomposition.NMF()
        elif  self.dimensionality_reduction_name==     "None":              self.dim_red=sklearn.decomposition.PCA() #set to PCA so it can return default values, but it will not be trained or executed
        #etc
        
        if self.non_linear_regressor_name==    "AdaBoostRegressor":         self.non_lin_reg=sklearn.ensemble.AdaBoostRegressor()
        #elif self.non_linear_regressor_name==  "BaggingRegressor":          self.non_lin_reg=sklearn.ensemble.BaggingRegressor()
        elif self.non_linear_regressor_name==  "BaggingRegressor":          self.non_lin_reg=myBaggingRegressor()
        #elif self.non_linear_regressor_name==  "BaggingRegressor-XGB":      self.non_lin_reg=sklearn.ensemble.BaggingRegressor(xgboost.XGBRegressor())
        elif self.non_linear_regressor_name==  "BaggingRegressor-XGB":      self.non_lin_reg=myBaggingRegressor(xgboost.XGBRegressor(tree_method="exact"))
        elif self.non_linear_regressor_name==  "BaggingRegressor-XGB-hist": self.non_lin_reg=myBaggingRegressor(xgboost.XGBRegressor(tree_method="hist"))
        elif self.non_linear_regressor_name==  "ExtraTreesRegressor":       self.non_lin_reg=sklearn.ensemble.ExtraTreesRegressor()
        elif self.non_linear_regressor_name==  "GradientBoostingRegressor": self.non_lin_reg=sklearn.ensemble.GradientBoostingRegressor()
        elif self.non_linear_regressor_name==  "RandomForestRegressor":     self.non_lin_reg=sklearn.ensemble.RandomForestRegressor()
        elif self.non_linear_regressor_name==  "XGBRegressor-hist":         self.non_lin_reg=xgboost.XGBRegressor(tree_method="exact")
        elif self.non_linear_regressor_name==  "XGBRegressor":              self.non_lin_reg=xgboost.XGBRegressor(tree_method="hist")
        elif self.non_linear_regressor_name==  "XGBRFRegressor":            self.non_lin_reg=xgboost.XGBRFRegressor(tree_method="exact")  #substitutes scikit learn RandomForestRegressor(), but there are some differences
        elif self.non_linear_regressor_name==  "XGBRFRegressor-hist":       self.non_lin_reg=xgboost.XGBRFRegressor(tree_method="hist")
        elif self.non_linear_regressor_name==  "SVR":                       self.non_lin_reg=sklearn.svm.SVR()
        elif self.non_linear_regressor_name==  "None":                      self.non_lin_reg=sklearn.ensemble.RandomForestRegressor() #set to RFR for avoiding errors, but it will not be used.
        #etc

        self.set_params(**kwargs)

    def do_shap(self, X_train="None",y_train="None",fraction=1):
        import shap
        print ("doing shap")
        #train (or retrain) the model using X_train and y_train
        #if X_train!="None" and y_train!="None":  
        #if True:
        self.fit(X_train,y_train)
        print ("fitting for shap")
        n_samples=int(len(y_train)*fraction)
           
        #shap is not available for "AdaBoostRegressor"
        #if self.non_linear_regressor_name not in ["BaggingRegressor","AdaBoostRegressor","SVR","BaggingRegressor-XGB"]: 
        if self.non_linear_regressor_name not in ["AdaBoostRegressor"]: 
            print(self.non_linear_regressor_name)#borrame
            if self.non_linear_regressor_name not in ["AdaBoostRegressor","BaggingRegressor","SVR","BaggingRegressor-XGB","BaggingRegressor-XGB-hist"]:
                non_lin_explainer = shap.explainers.Tree(self.non_lin_reg,shap.sample(self.reduced_non_linear_X_fitted,n_samples))
                self.shap_nl_values = non_lin_explainer.shap_values(self.reduced_non_linear_X_fitted,self.nonlin_y_train_fitted,check_additivity=False)
                print("here"); print(self.shap_nl_values)#borrame    
            elif self.non_linear_regressor_name in ["BaggingRegressor","BaggingRegressor-XGB","BaggingRegressor-XGB-hist"]:
                non_lin_explainers = [ shap.explainers.Tree(estimator, shap.sample(self.reduced_non_linear_X_fitted,n_samples))   for estimator in self.non_lin_reg.estimators_]
                self.shap_nl_values = np.mean( [explainer.shap_values(self.reduced_non_linear_X_fitted,self.nonlin_y_train_fitted,check_additivity=False) for explainer in non_lin_explainers],axis=0)
            else: 
                print ("kernel_explainer")
                non_lin_explainer = shap.KernelExplainer(self.non_lin_reg.predict,shap.sample(self.reduced_non_linear_X_fitted,10))
                self.shap_nl_values = non_lin_explainer.shap_values(self.reduced_non_linear_X_fitted)

            self.shap_nl_feature_importances = np.average(np.abs(self.shap_nl_values), axis=0)
            
            if self.dimensionality_reduction_name!= "None": 
                self.shap_nl_feature_importances = shap_nl_feature_importances.dot(self.normalized_components_fitted)
                self.shap_nl_values = np.array([v.dot(self.normalized_components_fitted) for v in self.shap_nl_values])                     

    def fit(self,X_train,y_train,sample_weight=None,early_stopping_rounds=10):

        #prepare the data
        self.linear_X_fitted=np.c_[X_train[self.linear_attributes]]
        self.non_linear_X_fitted=np.c_[X_train[self.non_linear_attributes]]
        #fit the linear regressor and predict values 
        #from sklearn.exceptions import ConvergenceWarning
        
        try: self.lin_reg.fit(self.linear_X_fitted,y_train,sample_weight=sample_weight)
        except ValueError as error: 
             self.fit_failure=True
             print (error)
             print (type(error).__name__)
             print("y_train length:"+str(len(y_train)))
             print ("x_train_dims:"+str(np.shape(self.linear_X_fitted)))
        if self.linear_regressor_name not in ["LinearRegression","RANSACRegressor"] and self.lin_reg.n_iter_==self.lin_reg.get_params()["max_iter"]:
            #print ("LINEAR PREDICTOD DID NOT CONVERGE DURING FITTING")
            self.fit_failure=True

        self.lin_predicted_Y_fitted=self.lin_reg.predict(self.linear_X_fitted)

        if self.combination=="sum":
            self.nonlin_y_train_fitted=y_train-self.lin_predicted_Y_fitted # Prediction_linear + Prediction_non_linear = Prediction
        elif self.combination=="average":
            self.nonlin_y_train_fitted=2*y_train-self.lin_predicted_Y_fitted # (Prediction_linear + Prediction_non_linear)/2 = Prediction
        elif self.combination=="product":
            self.nonlin_y_train_fitted=y_train/self.lin_predicted_Y_fitted # Prediction_linear * Prediction_non_linear = Prediction

        
        if self.non_linear_regressor_name!="None":
            
            if self.dimensionality_reduction_name!= "None":  #aply dimmensionality reduction, upgrading first the n_components if it was given as a fraction of the n_features
                self.reduced_non_linear_X_fitted=self.dim_red.fit_transform(self.non_linear_X_fitted) 
                #fit non linear regressor to the nl_y_train
                try: self.non_lin_reg.fit(self.reduced_non_linear_X_fitted,self.nonlin_y_train_fitted,sample_weight=sample_weight)
                except ValueError: self.fit_failure=True

                sq_components=self.dim_red.components_**2 #matrix with the squares of the components, one row per PCA dimension, one column per feature
                sum_sq_components=np.sum(sq_components,axis=1) #vector with the sum of the squares of the components, one element per PCA dimension
                normalized_components=[]
                for i in range(0,len(sum_sq_components)): normalized_components.append(sq_components[i]/sum_sq_components[i]) 
                self.normalized_components_fitted=np.array(normalized_components) #matrix with the normalized sq components, one row per PCA dim, one column per feature

                if self.non_linear_regressor_name not in ["BaggingRegressor","SVR","BaggingRegressor-XGB","XGBRegressor","XGBRFRegressor","XGBRegressor-hist","XGBRFRegressor-hist","BaggingRegressor-XGB-hist"]:
                    self.nl_feature_importances= self.non_lin_reg.feature_importances_.dot(self.normalized_components_fitted)
                else: self.nl_feature_importances=np.ones( len(self.non_linear_attributes) )/len(self.non_linear_attributes)

            else:
                #if self.non_linear_regressor_name in ["XGBRegressor","XGBRegressor-hist","BagginRegressor-XGB"]: #there is a problem with XGB early stopping; the model is sometimes not fitted and creates other erors in other methods.
                if self.non_linear_regressor_name in ["borrame"]: #["XGBRegressor","XGBRegressor-hist","BagginRegressor-XGB","bagginRegressor-XGB-hist"]:
                        try: self.non_lin_reg.fit(self.non_linear_X_fitted,self.nonlin_y_train_fitted,sample_weight=sample_weight,early_stopping_rounds=early_stopping_rounds)
                        except ValueError: self.fit_failure=True
                else:
                        try: self.non_lin_reg.fit(self.non_linear_X_fitted,self.nonlin_y_train_fitted,sample_weight=sample_weight)
                        except ValueError: self.fit_failure=True
                if self.non_linear_regressor_name not in  ["BaggingRegressor","SVR","BaggingRegressor-XGB","XGBRegressor","XGBRFRegressor","XGBRegressor-hist","XGBRFRegressor-hist","BaggingRegressor-XGB-hist"] and not self.fit_failure:
                    self.nl_feature_importances=self.non_lin_reg.feature_importances_
                else:
                    self.nl_feature_importances=np.ones(len(self.non_linear_attributes))/len(self.non_linear_attributes)
            self.reduced_non_linear_X_fitted=self.non_linear_X_fitted




    # predictions made by the linear regressor
    def linear_predict(self,data_X):
        linear_data_X=np.c_[data_X[self.linear_attributes]]
        return self.lin_reg.predict(linear_data_X)

    # predictions of the non linear model
    def nonlin_predict(self,data_X):
        non_linear_data_X=np.c_[data_X[self.non_linear_attributes]]
        if self.dimensionality_reduction_name== "None":
            return self.non_lin_reg.predict(non_linear_data_X)
        else:
            reduced_non_linear_data=self.dim_red.transform(non_linear_data_X)
            return self.non_lin_reg.predict(reduced_non_linear_data)


    # final Y predicted
    def predict(self,data_X):            
        linear_predicted_Y=self.linear_predict(data_X)
        
        if self.non_linear_regressor_name!="None":
            nonlin_predicted=self.nonlin_predict(data_X)
            if self.combination=="sum": 
                return linear_predicted_Y+nonlin_predicted  # Prediction= Linear_prediction + non_linear_prediction
            elif self.combination=="average":
                return (linear_predicted_Y+nonlin_predicted)*0.5  # Prediction= (Linear_prediction + non_linear_prediction)/2
            elif self.combination=="product":
                return (linear_predicted_Y*nonlin_predicted)  # Prediction= (Linear_prediction * non_linear_prediction)
        else:
            return linear_predicted_Y

    # returns the parameters (required for cross validation and cross prediction)         
    def get_params(self,deep=False):        
        linear_params,dim_red_params,non_linear_params={},{},{}
        for key,value in self.lin_reg.get_params(deep).items():     linear_params["l_"+key]=value
        for key,value in self.dim_red.get_params(deep).items():     dim_red_params["dr_"+key]=value
        for key,value in self.non_lin_reg.get_params(deep).items(): non_linear_params["nl_"+key]=value 
        if "nl_estimator"  in   non_linear_params.keys(): non_linear_params.pop("nl_estimator")

        return {**{'linear_regressor': self.linear_regressor_name, 
                   'dimensionality_reduction':self.dimensionality_reduction_name,
                   'non_linear_regressor': self.non_linear_regressor_name,
                   'linear_attributes': self.linear_attributes,
                   'non_linear_attributes': self.non_linear_attributes},
                   **linear_params,**dim_red_params,**non_linear_params,
                   **super().get_params(deep)
                   }



