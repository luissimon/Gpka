#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
# for snapshot-ensembles, from: https://github.com/titu1994/Snapshot-Ensembles/blob/master/snapshot.py

import numpy as np
import os
import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)





import keras.callbacks as callbacks
from keras.callbacks import Callback

class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.

    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).

    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.

    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            #print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))

class SnapshotBestModelCheckpoint(callbacks.ModelCheckpoint):
    
    def __init__(self,nb_epochs, nb_snapshots, save_best_only =True,**kwargs):

        self.check = nb_epochs // nb_snapshots      
        super().__init__(**kwargs)
        self.initial_best=self.best
        self.counter=0
        self.original_filepath=self.filepath
    
    def on_train_begin(self,logs):  #set the name of the filepath for the first snapshot (otherwise it will appear as "***")
        self.counter+=1
        self.filepath= self.original_filepath.replace("***",str(self.counter))

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:  #on every annealing cycle...
            self.counter+=1                                                         #update counter
            self.filepath= self.original_filepath.replace("***",str(self.counter))  #change the name of the ModelCheckpoint filepath 
            self.best=self.initial_best                                             #reset best value
        if (epoch + 1) % self.check> self.check/4 and (epoch + 1) % self.check< 4*self.check/4:  #black-out period to not save structures in the peaks even if they are apparently better
            super().on_epoch_end(epoch,logs)                                                    #...let ModelCheckpoint parent class to do the rest...

class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.

    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """
    #modified to include two initial values of the learning rates, so that the first is used only in the first annealing cycle
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.01):
        """
        Initialize a snapshot callback builder.

        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model',monitor="val_mae"):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.

        Args:
            model_prefix: prefix for the filename of the weights.

        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        if not os.path.exists('weights/'):
            os.makedirs('weights/')

        callback_list = [SnapshotBestModelCheckpoint(filepath= "weights/%s-***.h5" % model_prefix,  #will replace "***" with the number of the annealing cycle
                                                     nb_epochs=self.T, nb_snapshots=self.M,         #specific parameters of SnapshotBestModelCheckpoint -the rest are parameters for ModelCheckpoint
                                                     monitor="val_mae", save_weights_only=True, save_best_only=True ),#no point in not setting save_best_only to True...                       
                         callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)                         ]

        return callback_list


    def _exp_trig_anneal_schedule(self,t):
        inner=np.pi*(2*t/(self.T/self.M)+1/4)
        factor_trig=np.exp( np.cos(inner)+np.sin(inner)  )-np.exp(-2**0.5)
        factor_trig=factor_trig/(np.exp(2**0.5))  #normalize, cos(pi/4)+sin(pi/4)=2**0.5 and add a minimal value to learning rate (3% of initial learning rate)
        factor_decay=max([(2**0.25)**(-t/(self.T/self.M)),0.2]) #divide by 2 the max. value in every cycle
        return self.alpha_zero*max([factor_decay*factor_trig , 0.01])  #the minimal value is at least 1% of the max value (we do not want learning rate to be 0)

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        factor_decay=max(  [(2**(-t/(self.T/self.M))),1/2])
        return float((self.alpha_zero / 2 * cos_out)*factor_decay)


