#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
import copy
import numpy as np
import math
import itertools
#import quaternion
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.spatial.transform import Rotation 

angs_to_bohr=1.88973
bohr_to_angs=0.52917720859
hartrees_to_kcal=627.5095

class Property:

    def __init__ (self,name,text_before="",text_after="",separators=["\n",""],drop_words=[],length=1000,format="float"):
        self.name=name               #the name of the property
        self.text_before=text_before #the text that is before the property to analyze
        self.text_after=text_after   #the text that is after the property to analyze (optional; if not given, only one line will be read)
        self.separators=separators   #a list of the strings that will be used to split data in the property
        self.length=length           #the number of lines that the text will be read after finding text_before searching of text_after (default value should be enough large)
        self.drop_words=drop_words   #a list of strings that will be removed (for example, units, etc)
        self.format=format           #the format (float, string, np.array) of the output properties

    #a function for transforming strings in floats (returns a string if the cast not possible)
    def __cond_float(self,number):
        try: 
            a= float(number)
            return a
        except ValueError:
            return number

    #a function to read text in search of the properties, using the text, separators, etc, of the object
    #a list of strings is passed, along with the index of the first of the elements in which to begin the search
    def read (self,lines,offset=0):
        p=None
        text_to_analyze="" 
        if self.text_before!="" and self.text_before in lines[offset]:
            #prepare the text that will be analyzed 
            text_to_analyze=lines[offset]
            #if self.text_after is not in the analyzed line, add lines until it is found:
            if self.text_after!="" and self.text_after not in lines[offset]:
                more_text_to_analyze=""
                for j in range(offset+1,offset+self.length):
                    more_text_to_analyze+=lines[j]
                    if self.text_after in lines[j]:
                        #only when "text_after" is found, the new lines are added to text_to_analyze 
                        text_to_analyze+="\n"+more_text_to_analyze
                        break
                    #do something if the text_after was not found:
                    #print ("warning: 'text after' not found")
            
        #make sure both text_after and text_before are in the text_to_analyze
        if text_to_analyze!="" and (self.text_before in text_to_analyze or self.text_before=="") and (self.text_after in text_to_analyze or self.text_after==""):
                #remove everything that is not between text_before and text_after
                if self.text_before!="": text_to_analyze=text_to_analyze.split(self.text_before)[1]
                if self.text_after!="" : text_to_analyze=text_to_analyze.split(self.text_after)[0]
                text_to_analyze=text_to_analyze.strip()
                
                #split text using separators
                if len(self.separators)>0:
                    if self.separators[0]!="":p=text_to_analyze.split(self.separators[0])
                    else: p=text_to_analyze.split()
                    #if there are more than one separator, split recursively (only 2 supported)
                    if len(self.separators)==2:
                        if self.separators[1]!="": p=[q.split(self.separators[1]) for q in p]
                        else: p=[q.split() for q in p]
                        
                else: p=text_to_analyze
                #drop words that should not be incluede (for example, units, etc) and format
                if len(self.separators)==0:
                    for d_w in self.drop_words: p=p.replace(d_w) 
                    if format=="float": p=self.__cond_float(p)
                    else: p=p
                if len(self.separators)==1:
                    for d_w in self.drop_words: p.remove(d_w)
                    if format=="float" or format=="np_array":p=[self.__cond_float(e) for e in p]
                if len(self.separators)==2:
                    for d_w in self.drop_words:
                        for pp in p: pp.remove(d_w)       
                    if format=="float" or format=="np_array":p=[[self.__cond_float(e) for e in pp]for pp in p]
                if format=="np_array": p= np.array(p)
        if type(p)==list and len(p)==1:p=p[0]
        return p


#each output file is an object from which properties can be read. 
# generic class from which G16 and ORCA output files inherit:               
class QM_output:

    # Dictionary of all elements matched with their atomic masses. Thanks to
    # https://gist.github.com/lukasrichters14/c862644d4cbcf2d67252a484b7c6049c
    atomic_weights = {
            'h' : 1.008,'he' : 4.003, 'li' : 6.941, 'be' : 9.012,
            'b' : 10.811, 'c' : 12.011, 'n' : 14.007, 'o' : 15.999,
            'f' : 18.998, 'ne' : 20.180, 'na' : 22.990, 'mg' : 24.305,
            'al' : 26.982, 'si' : 28.086, 'p' : 30.974, 's' : 32.066,
            'cl' : 35.453, 'ar' : 39.948, 'k' : 39.098, 'ca' : 40.078,
            'sc' : 44.956, 'ti' : 47.867, 'v' : 50.942, 'cr' : 51.996,
            'mn' : 54.938, 'fe' : 55.845, 'co' : 58.933, 'ni' : 58.693,
            'cu' : 63.546, 'zn' : 65.38, 'ga' : 69.723, 'ge' : 72.631,
            'as' : 74.922, 'se' : 78.971, 'br' : 79.904, 'kr' : 84.798,
            'rb' : 84.468, 'sr' : 87.62, 'y' : 88.906, 'zr' : 91.224,
            'nb' : 92.906, 'mo' : 95.95, 'tc' : 98.907, 'ru' : 101.07,
            'rh' : 102.906, 'pd' : 106.42, 'ag' : 107.868, 'cd' : 112.414,
            'in' : 114.818, 'sn' : 118.711, 'sb' : 121.760, 'te' : 126.7,
            'i' : 126.904, 'xe' : 131.294, 'cs' : 132.905, 'ba' : 137.328,
            'la' : 138.905, 'ce' : 140.116, 'pr' : 140.908, 'nd' : 144.243,
            'pm' : 144.913, 'sm' : 150.36, 'eu' : 151.964, 'gd' : 157.25,
            'tb' : 158.925, 'dy': 162.500, 'ho' : 164.930, 'er' : 167.259,
            'tm' : 168.934, 'yb' : 173.055, 'lu' : 174.967, 'hf' : 178.49,
            'ta' : 180.948, 'w' : 183.84, 're' : 186.207, 'os' : 190.23,
            'ir' : 192.217, 'pt' : 195.085, 'au' : 196.967, 'hg' : 200.592,
            'tl' : 204.383, 'pb' : 207.2, 'bi' : 208.980, 'po' : 208.982,
            'at' : 209.987, 'rn' : 222.081, 'fr' : 223.020, 'ra' : 226.025,
            'ac' : 227.028, 'th' : 232.038, 'pa' : 231.036, 'u' : 238.029,
            'np' : 237, 'pu' : 244, 'am' : 243, 'cm' : 247, 'bk' : 247,
            'ct' : 251, 'es' : 252, 'fm' : 257, 'md' : 258, 'no' : 259,
            'lr' : 262, 'rf' : 261, 'db' : 262, 'sg' : 266, 'bh' : 264,
            'hs' : 269, 'mt' : 268, 'ds' : 271, 'rg' : 272, 'cn' : 285,
            'nh' : 284, 'fl' : 289, 'mc' : 288, 'lv' : 292, 'ts' : 294,
            'og' : 294}

    # Dictionary of all elements matched with their atomic numbers. 
    atomic_numbers = {
            'h' : 1,'he' : 2, 'li' : 3, 'be' : 4, 'b' : 5, 'c' : 6, 'n' : 7, 'o' : 8,
            'f' : 9, 'ne' : 10, 'na' : 11, 'mg' : 12, 'al' : 13, 'si' : 14, 'p' : 15, 's' : 16,
            'cl' : 17, 'ar' : 18, 'k' : 19, 'ca' : 20, 'sc' : 21, 'ti' : 22, 'v' : 23, 'cr' : 24,
            'mn' : 25, 'fe' : 26, 'co' : 27, 'ni' : 28, 'cu' : 29, 'zn' : 30, 'ga' : 31, 'ge' : 32,
            'as' : 33, 'se' : 34, 'br' : 35, 'kr' : 36, 'rb' : 37, 'sr' : 38, 'y' : 39, 'zr' : 40,
            'nb' : 41, 'mo' : 42, 'tc' : 43, 'ru' : 44, 'rh' : 45, 'pd' : 46, 'ag' : 47, 'cd' : 48,
            'in' : 49, 'sn' : 50, 'sb' : 51, 'te' : 52, 'i' : 53, 'xe' : 54, 'cs' : 55, 'ba' : 56,
            'la' : 57, 'ce' : 58, 'pr' : 59, 'nd' : 60, 'pm' : 61, 'sm' : 62, 'eu' : 63, 'gd' : 64,
            'tb' : 65, 'dy': 66, 'ho' : 67, 'er' : 68, 'tm' : 69, 'yb' : 70, 'lu' : 71, 'hf' : 72,
            'ta' : 73, 'w' : 74, 're' : 75, 'os' : 76, 'ir' : 777, 'pt' : 78, 'au' : 79, 'hg' : 80,
            'tl' : 81, 'pb' : 82, 'bi' : 83, 'po' :84, 'at' : 85, 'rn' : 86, 'fr' : 87, 'ra' : 88,
            'ac' : 89, 'th' : 90, 'pa' : 91, 'u' : 92, 'np' : 93, 'pu' : 94, 'am' : 95, 'cm' : 96, 'bk' : 97,
            'ct' : 98, 'es' : 99, 'fm' : 100, 'md' : 101, 'no' : 102,  'lr' : 103, 'rf' : 104, 'db' : 105,
            'sg' : 106, 'bh' : 107, 'hs' : 108, 'mt' : 109, 'ds' : 110, 'rg' : 111, 'cn' : 112,
            'nh' : 113, 'fl' : 114, 'mc' : 115, 'lv' : 116, 'ts' : 117,  'og' : 118}

    def reset_local_properties(self):
        self.temp_dipole_moment=[np.nan,np.nan,np.nan]
        self.temp_polarizability=0.0
        self.temp_cartesians=[]   #does this work for ORCA outputs?
        self.temp_energy=np.nan
        self.temp_chelpg_charges=[]
        self.temp_ESP_charges=[]
        self.temp_mulliken_charges=[]
        self.temp_loewdin_charges=[]
        self.temp_hirshfeld_charges=[]
        self.temp_mayer_pop_analysis=[]
        self.temp_mayer_bond_orders=[]
        self.temp_chemical_isotropic_shields=[]
        self.temp_structure_number=0
        self.temp_rms_gradient=np.nan
        self.temp_max_cart_force=np.nan
        self.temp_property={}
        for prop in self.list_of_properties: self.temp_property[prop.name]=[]
        self.temp_gibbs_free_energy=np.nan
        self.temp_zero_point_energy=np.nan
        self.temp_scaling_factor_for_freq=np.nan
        self.temp_frequencies=[]
        self.temp_red_masses=[]
        self.temp_frc_const=[] 
        self.temp_model_syst_percent=[]
        self.temp_real_syst_percent=[]
        self.temp_ir_intensities=[]
        self.temp_normal_coordinates=[]
    
    def flush_local_properties(self):
        self.cart_coordinates.append(self.temp_cartesians)
        self.dipole_moments.append(self.temp_dipole_moment)
        self.polarizabilities.append(self.temp_polarizability)
        self.energies=np.hstack([self.energies, float(self.temp_energy)])
        self.RMS_cart_forces=np.hstack([self.RMS_cart_forces,self.temp_rms_gradient])
        self.MAX_cart_forces=np.hstack([self.MAX_cart_forces,self.temp_max_cart_force])
        self.ESP_charges.append(self.temp_ESP_charges)
        self.chelpg_charges.append(self.temp_chelpg_charges)
        self.mulliken_charges.append(self.temp_mulliken_charges)
        self.loewdin_charges.append(self.temp_loewdin_charges)
        self.hirshfeld_charges.append(self.temp_hirshfeld_charges)
        self.mayer_pop_analysis.append(self.temp_mayer_pop_analysis)
        self.mayer_bond_orders.append(self.temp_mayer_bond_orders)
        self.chemical_isotropic_shields.append(self.temp_chemical_isotropic_shields)
        self.zero_point_energies=np.hstack([self.zero_point_energies,self.temp_zero_point_energy])
        self.gibbs_free_energies=np.hstack([self.gibbs_free_energies,self.temp_gibbs_free_energy])
        self.frequencies.append(self.temp_frequencies)
        self.scaling_factor_for_freq.append(self.temp_scaling_factor_for_freq)
        self.red_masses.append(self.temp_red_masses)  
        self.frc_const.append(self.temp_frc_const)  
        self.model_syst_percent.append(self.temp_model_syst_percent)
        self.real_syst_percent.append(self.temp_real_syst_percent) 
        self.ir_intensities.append(self.temp_ir_intensities) 
        self.normal_coordinates.append(self.temp_normal_coordinates) 
        for prop in self.list_of_properties:
            self.properties[prop.name].append(self.temp_property[prop.name])

    def __init__(self,outfile,properties=[]):

        self.outfile=outfile
        #file=open(self.outfile,"r")
        #lines_in_file=file.readlines()

        self.list_of_properties=properties
        #initialize a dictionary with the properties, elaborated from the list of properties passed to the constructor
        self.properties={}
        for prop in self.list_of_properties: self.properties[prop.name]=[]

        self.cart_coordinates=[]
        #internal coordinates definitions (not values!) with format: list of strings with comma-separated atom numbers
        self.int_coord_definitions=[]
        self.RMS_cart_forces=np.array([],dtype=float)
        self.MAX_cart_forces=np.array([],dtype=float)
        self.energies=np.array([],dtype=float)
        #self.dipole_moments=np.empty_like([1,2,3],dtype=float)
        self.dipole_moments=[]
        self.polarizabilities=[]

                
        self.stationary_points=[]
        self.thermochemistry_calculations=[]

        self.frequencies=[]
        self.scaling_factor_for_freq=[]
        self.red_masses=[]
        self.atomic_symbols=[]
        self.masses=[]
        self.frc_const=[]
        self.model_syst_percent=[] #this is not read in ORCA calcns. but is included for coherence with G16_output
        self.real_syst_percent=[]
        self.ir_intensities=[]
        self.normal_coordinates=[]
        self.gibbs_free_energies=np.array([],dtype=float)
        self.zero_point_energies=np.array([],dtype=float)

        self.ESP_charges=[]
        self.chelpg_charges=[]
        self.mulliken_charges=[]
        self.loewdin_charges=[]
        self.hirshfeld_charges=[]
        self.mayer_pop_analysis=[]
        self.mayer_bond_orders=[]
        self.chemical_isotropic_shields=[]


        #local temporary variables that will be read and flushed after each cycle:
        self.temp_cartesians=[]
        self.temp_dipole_moment=[np.nan,np.nan,np.nan]
        self.temp_polarizability=0.0
        self.temp_energy=np.nan
        self.temp_ESP_charges=[]
        self.temp_chelpg_charges=[]
        self.temp_mulliken_charges=[]
        self.temp_loewdin_charges=[]
        self.temp_hirshfeld_charges=[]
        self.temp_mayer_pop_analysis=[]
        self.temp_mayer_bond_orders=[]
        self.temp_chemical_isotropic_shields=[]
        self.structure_number=0
        self.temp_rms_gradient=np.nan
        self.temp_max_cart_force=np.nan
        self.temp_property={}
        for prop in self.list_of_properties: self.temp_property[prop.name]=[]
        self.temp_gibbs_free_energy=np.nan
        self.temp_zero_point_energy=np.nan
        self.temp_scaling_factor_for_freq=np.nan
        self.temp_frequencies=[]
        self.temp_normal_coordinates=[]
        self.temp_red_masses=[]
        self.temp_frc_const=[]
        self.temp_model_syst_percent=[]
        self.temp_real_syst_percent=[]
        self.temp_ir_intensities=[]

        self.first_structure_with_different_level_of_theory=False

    #determine if the first structure was calculated with a different level of theory
    def determine_first_str_with_different_level_of_theory(self):
        if len(self.energies)>1:
            self.first_structure_with_different_level_of_theory=((self.energies[0]-np.mean(self.energies[1:]))/np.std(self.energies[1:]))>100

    def get_smaller_rms_point(self):
        if self.first_structure_with_different_level_of_theory:
            return np.argmin(self.RMS_cart_forces[1:])+1
        else:
            return np.argmin(self.RMS_cart_forces)  
    
    def get_smaller_energy_point(self):
        if self.first_structure_with_different_level_of_theory:
            return np.argmin(self.energies[1:])+1
        else:
            return np.argmin(self.energies)

    def get_higher_energy_point(self):
        if self.first_structure_with_different_level_of_theory:
            return np.argmax(self.energies[1:])+1
        else:
            return np.argmax(self.energies)

    def get_stationary_point_with_higher_energy(self):
        st_points_energies=self.energies[self.stationary_points] 
        i=np.argmax(st_points_energies)
        return self.stationary_points[i] 

    def get_stationary_point_with_lower_energy(self):
        st_points_energies=self.energies[self.stationary_points] 
        i=np.argmin(st_points_energies)
        return self.stationary_points[i]

    #a deep copy of current object
    def copy(self):
        new_object=copy.deepcopy(self)
        new_object.outfile=self.outfile
        new_object.cartesian_coordinates=copy.deepcopy(self.cartesian_coordinates)
        new_object.int_coord_definitions=copy.deepcopy(self.int_coord_definitions)
        new_object.RMS_cart_forces=copy.deepcopy(self.RMS_cart_forces)
        new_object.Max_cart_forces=copy.deepcopy(self.Max_cart_forces)
        new_object.energies=copy.deepcopy(self.energies)
        new_object.stationary_points=copy.deepcopy(self.stationary_points)
        new_object.dipole_moments=copy.deepcopy(self.dipole_moments)
        new_object.polarizabilities=copy.deepcopy(self.polarizabilities)
        new_object.first_freq_movements=copy.deepcopy(self.first_freq_movements)
        new_object.freq_movements=copy.deepcopy(self.freq_movements)
        new_object.frequencies=copy.deepcopy(self.frequencies)
        new_object.red_masses=copy.deepcopy(self.red_masses)
        new_object.frc_consts=copy.deepcopy(self.frc_consts)
        new_object.model_syst_percent=copy.deepcopy(self.model_syst_percent)
        new_object.real_syst_percent=copy.deepcopy(self.real_syst_percent)
        new_object.ir_intensities=copy.deepcopy(self.ir_intensities)
        new_object.normal_coordinates=copy.deepcopy(self.normal_coordinates)
        return new_object


# if the molecular structure is read from an output file, it can contain a G16_output object
# with information about structures, energies, convergence, frequencies, etc.
# to do: read polarizabilities 
class G16_output(QM_output):

    # for a freq calculation, read the frequencies, reduced masses, force constants... and normal_coordinates
    # if there are more than one freq calculation in the file, it only reads the last.
    def read_frequencies(self,outfile=""):
        if outfile=="": outfile=self.outfile
        file=open(self.outfile,"r")
        lines_in_file=file.readlines()
        for i in range(0,len(lines_in_file)):
            #reset everything if a link1 hessian job is at the end
            if ("Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering" in lines_in_file[i] ):
                self.frequencies=[]
                self.red_masses=[]
                self.frc_const=[]
                self.model_syst_percent=[]
                self.real_syst_percent=[]
                self.ir_intensities=[]
                normal_coordinates=[]
            if (lines_in_file[i]).startswith(" Frequencies -- "):
                w=lines_in_file[i].split() 
                for ww in w[2:]: self.frequencies.append(float(ww))
            if (lines_in_file[i]).startswith(" Red. masses -- "):
                w=lines_in_file[i].split() 
                for ww in w[3:]: self.red_masses.append(float(ww))                                    
            if (lines_in_file[i]).startswith(" Frc consts  -- "):
                w=lines_in_file[i].split() 
                for ww in w[3:]: self.frc_const.append(float(ww))  
            if (lines_in_file[i]).startswith(" %ModelSys   -- "):
                w=lines_in_file[i].split() 
                for ww in w[2:]: self.model_syst_percent.append(float(ww))
            if (lines_in_file[i]).startswith(" %RealSys    -- "):
                w=lines_in_file[i].split() 
                for ww in w[2:]: self.real_syst_percent.append(float(ww))
            if (lines_in_file[i]).startswith(" IR Inten    -- "):
                w=lines_in_file[i].split() 
                for ww in w[3:]: self.ir_intensities.append(float(ww))
            if "Atom  AN      X      Y      Z" in lines_in_file[i]>-1:
                normal_coordinate=[]
                while (str.isdigit(lines_in_file[i][5])):
                    w=lines_in_file[i].split()
                    number_of_normal_modes_in_line=int(len(w)/3)
                    for j in range(0,number_of_normal_modes_in_line):
                        normal_coordinate.append([float(w[2+j]),float(w[3+j]),float(w[4+j])])
                    self.normal_coordinates.append(normal_coordinate)


    def __init__(self,outfile,properties=[]):
        super().__init__(outfile,properties)
        file=open(self.outfile,"r")
        lines_in_file=file.readlines()
        i=0
        structure_number=0  #counting structures, useful for creating a list with the structures that are optimized
        final_freq=False # a flag to know whether it is a conventional new step in the optimizatio or is the final freq calculation
        
        #read initial geometry; layers, atom symbols and atom types are read here as well
        while i<len(lines_in_file):

            #read geometry for every step of an optimization and add the cartesian coordinates to self.cart_coordinates_
            #some conditions:
            beguin_of_file="Entering Gaussian System," in lines_in_file[i]
            end_of_opt_step=( "GradGradGradGradGrad" in lines_in_file[i]) and ("Predicted" in lines_in_file[i-1])
            freq_calcn_after_conv=("Redundant internal coordinates found in file.  (old form)." in lines_in_file[i])
            if end_of_opt_step:
                #2) look forward to find out if there is a new step:
                for j in range(0,20):
                    if "orientation" in lines_in_file[i+j] and "Coordinates" in lines_in_file[i+j+2]:
                        new_step=True; break

            #1) when the begining of the file, a new step is found or when a final hess calculation is found:
            if beguin_of_file or new_step or freq_calcn_after_conv:
                
                #3) look forward to check if it is an ONIOM and/or external calculation, and set energy_id and dipole_id accordingly
                #if energy_id=="" or dipole_id=="":
                ONIOM_calculation=False
                external_calculation=False
                for j in range(1,1000):
                    if "ONIOM:" in lines_in_file[i+j]:
                        ONIOM_calculation=True;break
                for j in range(1,1000):
                    if "External" in lines_in_file[i+j]  and "Running external command" in lines_in_file[i+j+1]:
                        external_calculation=True;break
                
                if ONIOM_calculation: 
                    energy_id="ONIOM: extrapolated energy ="
                    dipole_id="ONIOM: Dipole moment (Debye):"
                else:
                    energy_id="SCF Done:"
                    dipole_id="Dipole moment (field-independent basis, Debye):"
                if external_calculation and not ONIOM_calculation:
                    energy_id="Recovered energy="
                    dipole_id="Dipole moment= "

                structure_number=structure_number+1
                self.reset_local_properties()

                #4) look forward to read the coordinates, energies, dipole moment, cartesian forces
                j=0
                while j<99999:

                        #read atomic symbols (only once):
                        if len(self.atomic_symbols)==0 and "Center     Atomic" in lines_in_file[i+j] and "Number     Number" in lines_in_file[i+j+1]:
                            j+=3
                            while "----------------------" in  lines_in_file[i+j]:
                                a_number=int(lines_in_file[i+j].split()[1])
                                a_symbol=list(Molecular_structure.atomic_numbers.keys())[list(Molecular_structure.atomic_numbers.values() ).index(a_number) ]
                                a_weight=Molecular_structure.atomic_weights[a_symbol]
                                self.atomic_symbols.append(a_symbol)
                                self.masses.append(a_weight)
                                j+=1

                        #read the redundant internal coordinates definitions 
                        # (only last in file will be kept)                    
                        if "! (Angstroms and Degrees)  !" in lines_in_file[i+j]:
                            j+=4
                            self.int_coord_definitions=[]
                            while "!" in lines_in_file[i+j][1]:
                                #a string containing the atom numbers involved in the redundant internal coordinate separated by commas
                                s=lines_in_file[i+j][10:30].strip().strip("(").strip(")")
                                int_coord=[int(x) for x in s.split(",")]
                                #borrame
                                #self.int_coord_definitions.append(lines_in_file[i][10:27].strip().strip("(").strip(")"))
                                self.int_coord_definitions.append(int_coord)
                                j+=1

                        #read coordinates:
                        if (i+j+1)<len(lines_in_file) and "orientation" in lines_in_file[i+j] and "Coordinates" in lines_in_file[i+j+2]:
                            j=j+5  
                            while not (lines_in_file[i+j].strip(' ').startswith("-------------------------------------------------------")):
                                if len(lines_in_file[i+j].split())>1:
                                    atom_x=float(lines_in_file[i+j].split()[3])
                                    atom_y=float(lines_in_file[i+j].split()[4])
                                    atom_z=float(lines_in_file[i+j].split()[5])
                                    self.temp_cartesians.append([atom_x,atom_y,atom_z])
                                j=j+1  
                            #5) if there is a distance matrix, skip lines containing it  
                            if (lines_in_file[i+j+1].strip(' ').startswith("Distance matrix")):
                                while "Stoichiometry" in lines_in_file[i]: j+=1

                        #read energy:
                        if (i+j+1)<len(lines_in_file) and energy_id in lines_in_file[i+j]:
                            if energy_id=="SCF Done:": self.temp_energy= lines_in_file[i+j].split()[4]    
                            elif energy_id=="Recovered energy=": self.temp_energy=lines_in_file[i+j].split()[2]
                            elif energy_id=="ONIOM: extrapolated energy =": self.temp_energy=lines_in_file[i+j].split()[4]
                            else:  self.temp_energy=lines_in_file[i+j].split(energy_id)[1]  

                        #read dipole moment:
                        if (i+j+1)<len(lines_in_file) and dipole_id in lines_in_file[i+j]:
                            l=lines_in_file[i+j+1].split()
                            if dipole_id=="Dipole moment= " :
                                l=lines_in_file[i+j].split()
                                self.temp_dipole_moment=np.array([float(l[2]),float(l[3]),float(l[4])])
                            elif dipole_id=="ONIOM: Dipole moment (Debye):" :
                                self.temp_dipole_moment=np.array([float(l[1]),float(l[3]),float(l[5])])
                            elif dipole_id=="Dipole moment (field-independent basis, Debye):" :
                                self.temp_dipole_moment=np.array([float(l[1]),float(l[3]),float(l[5])])                                 

                        #read cartesian forces:
                        if (i+j+1)<len(lines_in_file) and "Cartesian Forces:" in lines_in_file[i+j] and "----------------------" in lines_in_file[i+j-1]:
                            if lines_in_file[i+j].split()[0]=="Integrated": shft=1
                            else: shft=0
                            self.temp_rms_gradient=lines_in_file[i+j].split()[5+shft] 
                            self.temp_max_cart_force=lines_in_file[i+j].split()[3+shft]


                        #read mulliken charges:
                        if (i+j+1)<len(lines_in_file) and lines_in_file[i+j].startswith(" Mulliken charges:"): 
                            j+=2
                            while not lines_in_file[i+j].startswith(" Sum of Mulliken charges ="):
                                self.temp_mulliken_charges.append(float(lines_in_file[i+j].split()[2]))
                                j+=1

                        #read ESP charges:
                        if (i+j+1)<len(lines_in_file) and lines_in_file[i+j].startswith(" ESP charges:"):
                            j+=2
                            while not lines_in_file[i+j].startswith(" Sum of ESP charges ="):
                                self.temp_ESP_charges.append(float(lines_in_file[i+j].split()[2]))
                                j+=1

                        #read any of the properties in the list passed to the constructor:
                        #it is first added to property dictionary, and later appended to self.properties
                        for prop in properties:
                            read_property=prop.read(lines_in_file,i+j)
                            if read_property!=None:
                                self.temp_property[prop.name]=read_property

                        
                        #read the frequencies, reduced masses, force constants.... etc from a hessian calculation
                        #(only the last in file will be kept).
                        if  "Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering" in lines_in_file[i+j]:
                            """
                            self.frequencies=[]
                            self.red_masses=[]
                            self.frc_consts=[]
                            self.model_syst_percent=[]
                            self.real_syst_percent=[]
                            self.ir_intensities=[]
                            self.normal_coordinates=[]
                            """
                            pass
                        if (lines_in_file[i+j]).startswith(" Frequencies -- "):
                            w=lines_in_file[i+j].split() 
                            for ww in w[2:]: self.temp_frequencies.append(float(ww))
                        if (lines_in_file[i+j]).startswith(" Red. masses -- "):
                            w=lines_in_file[i+j].split() 
                            for ww in w[3:]: self.temp_red_masses.append(float(ww))                                    
                        if (lines_in_file[i+j]).startswith(" Frc consts  -- "):
                            w=lines_in_file[i+j].split() 
                            for ww in w[3:]: self.temp_frc_const.append(float(ww))  
                        if (lines_in_file[i+j]).startswith(" %ModelSys   -- "):
                            w=lines_in_file[i+j].split() 
                            for ww in w[2:]: self.temp_model_syst_percent.append(float(ww))
                        if (lines_in_file[i+j]).startswith(" %RealSys    -- "):
                            w=lines_in_file[i+j].split() 
                            for ww in w[2:]: self.temp_real_syst_percent.append(float(ww))
                        if (lines_in_file[i+j]).startswith(" IR Inten    -- "):
                            w=lines_in_file[i+j].split() 
                            for ww in w[3:]: self.temp_ir_intensities.append(float(ww))
                        if ("Atom  AN      X      Y      Z" in lines_in_file[i+j]):
                            normal_coordinate1=[]
                            normal_coordinate2=[]
                            mormal_coordinate3=[]
                            w=lines_in_file[i+j].split()
                            number_of_normal_modes_in_line=int(len(w)/3)
                            i+=1
                            while (len (lines_in_file[i+j])>6 and str.isdigit(lines_in_file[i+j][5])):
                                w=lines_in_file[i+j].split()
                                if number_of_normal_modes_in_line>0:
                                    normal_coordinate1.append([float(w[2]),float(w[3]),float(w[4])])
                                if number_of_normal_modes_in_line>1:
                                    normal_coordinate2.append([float(w[5]),float(w[6]),float(w[7])])
                                if number_of_normal_modes_in_line>2:
                                    mormal_coordinate3.append([float(w[5]),float(w[6]),float(w[7])])
                                i+=1
                            if len(normal_coordinate1)>0:
                                self.temp_normal_coordinates.append(normal_coordinate1)
                            if len(normal_coordinate2)>0:
                                self.temp_normal_coordinates.append(normal_coordinate2)
                            if len(mormal_coordinate3)>0:
                                self.temp_normal_coordinates.append(mormal_coordinate3)  
                        if "Sum of electronic and thermal Free Energies=" in lines_in_file[i+j]:
                            self.thermochemistry_calculations.append(structure_number)
                            self.temp_gibbs_free_energy=float(lines_in_file[i+j].split()[7])
                        if "Sum of electronic and zero-point Energies=" in lines_in_file[i+j]:
                            self.temp_zero_point_energy=float(lines_in_file[i+j].split()[6])                        
                        #print (str(j)+" out of:"+str(len(lines_in_file)))
                        #print (lines_in_file[i+j])

                        #break the search of properties when reaching the end of the file or a new "converged?" table
                        if (i+j+2)>len(lines_in_file) or " Predicted change in Energy=" in lines_in_file[i+j] and  (("GradGradGradGradGradGradGradGrad" in lines_in_file[i+j+1]) or ("Optimization completed." in  lines_in_file[i+j+1])):

                            if ((i+j+2)>len(lines_in_file)): break

                            if "Optimization completed." in  lines_in_file[i+j+1]:
                                self.stationary_points.append(structure_number-1)
                            new_step=False #set to false so this will not be executed until a new geometry is found.
                            self.flush_local_properties()
                            self.reset_local_properties()
                            break

                        """
                        #OLD: if an stationary point has been found, take a note of the structure number.
                        if any (["-- Stationary point found." in l for l in lines_in_file[i+j:i+j+15] ]) :
                            self.stationary_points.append(structure_number-1)
                            j+=7
                            print (structure_number-1)
                            break 
                        """                       
                        j+=1
                        
                #move the position in the file to the end of this geometry optimization step (needed?)
                i+=j
            
            i+=1


        #things to do at the end:
        #transform self.dipole_moments to np.array
        self.dipole_moments=np.array(self.dipole_moments)
        #close file
        file.close()
        #print ("number of stationary points in file: "+str(len(self.stationary_points)))
        #determine if the first structure was calculated with a different level of theory
        self.determine_first_str_with_different_level_of_theory()
        #self.first_structure_with_different_level_of_theory=False
        #if len(self.energies)>1:
        #    self.first_structure_with_different_level_of_theory=((self.energies[1]-np.mean(self.energies[1:]))/np.std(self.energies[1:]))>100
        

class ORCA_output(QM_output):

    def __init__(self,outfile,properties=[]):

        super().__init__(outfile,properties)

        self.outfile=outfile
        file=open(self.outfile,"r")
        lines_in_file=file.readlines()

        #look forward to find out which kind of calculation (no combined calculations supoorted yet)

        for j in range(1,1000):
            if "* Single Point Calculation *" in lines_in_file[j]:
                clcn_type="single_point"
                break
            if "* Geometry Optimization Run *"  in lines_in_file[j]:
                clcn_type="geom_opt"
                break
            if "* Energy+Gradient Calculation *" in lines_in_file[j]:
                clcn_type="e_grad" #pure freq calculations are also of this type
                break

        i=0
        

        structure_number=0  #counting structures, useful for creating a list with the structures that are optimized
        while i<len(lines_in_file):

            #read the atomic masses and symbols (only the first time):
            if "CARTESIAN COORDINATES (A.U.)" in lines_in_file[i] and self.atomic_symbols==[]:
                i+=3
                while lines_in_file[i]!="\n" and lines_in_file[i]!="* core charge reduced due to ECP\n":
                    self.masses.append(float(lines_in_file[i].split()[4]))
                    self.atomic_symbols.append(lines_in_file[i].split()[1])
                    i=i+1

            #read internal coordinates definitions (only once)
            if self.int_coord_definitions==[] and "Redundant Internal Coordinates" in lines_in_file[i] and "Definition" in lines_in_file[i+4]:
                i+=6
                self.int_coord_definitions=[]
                while "-----------------------------------------------------------------" not in lines_in_file[i]:
                    pos=lines_in_file[i].find("(")
                    #print (lines_in_file[i]) #borrame
                    int_coord=[int(lines_in_file[i][(pos+3):(pos+6)])+1,int(lines_in_file[i][(pos+9):(pos+12)])+1]
                    #old: int_coord=[int(lines_in_file[i][(pos+3):(pos+6)])+1,int(lines_in_file[i][(pos+8):(pos+12)])+1] does the change work? borrame
                    if lines_in_file[i][(pos+15):(pos+18)]!="   ":int_coord.append(int(lines_in_file[i][(pos+15):(pos+18)])+1)
                    if lines_in_file[i][(pos+21):(pos+24)]!="   " and lines_in_file[i][(pos-1)]=="D":
                        int_coord.append(int(lines_in_file[i][(pos+21):(pos+24)])+1)
                    #if lines_in_file[i][(pos+21):(pos+24)]!="   " and lines_in_file[i][(pos-1)]=="L": #do something in these cases. It appears with linear bonds
                    #    int_coord.append(int(lines_in_file[i][(pos+21):(pos+24)])+1)
                    self.int_coord_definitions.append(int_coord)
                    i+=1

            #read cartesian coordinates
            if "CARTESIAN COORDINATES (ANGSTROEM)" in lines_in_file[i]:
                self.temp_cartesians=[]
                i+=2
                while lines_in_file[i]!="\n":
                    w=lines_in_file[i].split()
                    self.temp_cartesians.append([float(w[1]),float(w[2]),float(w[3])]) 
                    i=i+1
            if "FINAL SINGLE POINT ENERGY" in lines_in_file[i]: self.temp_energy=float(lines_in_file[i].split()[4])
            if "          RMS gradient" in lines_in_file[i]:            self.temp_rms_gradient=float(lines_in_file[i].split()[2])                            
            if "          MAX gradient" in lines_in_file[i]:            self.temp_max_cart_force=float(lines_in_file[i].split()[2])
            if any([a in lines_in_file[i] for a in ["CHELPG Charges","MULLIKEN ATOMIC CHARGES","LOEWDIN ATOMIC CHARGES"]])  and "----------------------" in lines_in_file[i+1]:
                j=2
                #for some reason, chelpg charges are shown twice in a single-point calculation; this deletes the first read and keeps only the latest.
                if clcn_type=="single_point": self.temp_chelpg_charges=[] 

                while "-----------------------" not in lines_in_file[i+j] and "Sum of " not in lines_in_file[i+j] and lines_in_file[i+j].strip()!="":
                    if "CHELPG Charges" in lines_in_file[i]:              
                        self.temp_chelpg_charges.append(float(lines_in_file[i+j].split(":")[1]))
                    elif "MULLIKEN ATOMIC CHARGES" in lines_in_file[i]:    
                        self.temp_mulliken_charges.append(float(lines_in_file[i+j].split(":")[1]))
                    elif "LOEWDIN ATOMIC CHARGES" in lines_in_file[i]:     
                        self.temp_loewdin_charges.append(float(lines_in_file[i+j].split(":")[1])) 
                    j+=1
                i+=j
            if "HIRSHFELD ANALYSIS" in lines_in_file[i] and "------------------" in lines_in_file[i+1]:
                j=7
                while " TOTAL" not in lines_in_file[i+j+1]:
                    self.temp_hirshfeld_charges.append(float(lines_in_file[i+j].split()[2]))
                    j+=1
                i+=j
            if "* MAYER POPULATION ANALYSIS *" in lines_in_file[i] and "***********************" in lines_in_file[i+1]:
                j=11
                while  "Mayer bond orders " not in lines_in_file[i+j+1]:
                    w=lines_in_file[i+j].split()
                    mayer_pop={}
                    mayer_pop["NA"],mayer_pop["ZA"],mayer_pop["QA"],mayer_pop["VA"],mayer_pop["BVA"],mayer_pop["FA"]=float(w[2]),float(w[3]),float(w[4]),float(w[5]),float(w[6]),float(w[7])
                    self.temp_mayer_pop_analysis.append(mayer_pop)
                    j+=1
                i+=j
            if "  Mayer bond orders larger than" in lines_in_file[i]:
                j=1
                while lines_in_file[i+j].strip()!="":
                    w=lines_in_file[i+j].split("B(")[1:]
                    for ww in w:
                        self.temp_mayer_bond_orders.append( [int(ww.split("-")[0])+1,int(ww.split(",")[1].split("-")[0])+1,  float(ww.split(":")[1])] )
                    j+=1
                i+=j


            if "CHEMICAL SHIELDING SUMMARY (ppm)" in lines_in_file[i] and "------------------" in lines_in_file[i+1]:
                j=6
                while lines_in_file[i+j].strip()!="":
                    self.temp_chemical_isotropic_shields.append(float((lines_in_file[i+j].split()[2])))
                    j+=1
                i+=j



            if "Total Dipole Moment    :" in lines_in_file[i]: 
                w=lines_in_file[i].split()
                self.temp_dipole_moment=np.array([float(w[4]),float(w[5]),float(w[6])])
            if "Isotropic polarizability :"  in lines_in_file[i]:
                self.temp_polarizability=float( lines_in_file[i].split()[-1] )
            #read frequencies and thermochemistry (only one for each file... composed jobs?):
            if ("VIBRATIONAL FREQUENCIES" in lines_in_file[i] and "-----------------------" in lines_in_file[i+1]):                
                i+=3
                self.temp_scaling_factor_for_freq=float(lines_in_file[i].split()[5])
                i+=2
                while lines_in_file[i].strip()!="":
                    self.temp_frequencies.append(float(lines_in_file[i].split()[1]))
                    i+=1
            if ("NORMAL MODES" in lines_in_file[i] and len(self.masses)>0):
                #initialize normal_coordinates
                self.temp_normal_coordinates=[ [[0.0,0.0,0.0] for k in range(0,len(self.atomic_symbols))] for kk in range(0,3*len(self.atomic_symbols)) ]
                i+=7
                while all([str.isdigit(n) for n in lines_in_file[i].split()]):
                    number_of_normal_modes_per_line=len(lines_in_file[i].split())
                    normal_mode_numbers=[int(x) for x in lines_in_file[i].split()]
                    i+=1
                    while (lines_in_file[i].strip()!="" and str.isdigit(lines_in_file[i][6])):
                        for j in range(0,number_of_normal_modes_per_line-1):
                            atom_number=int(int(lines_in_file[i].split()[0])/3)
                            X=float(lines_in_file[i].split()[j+1]) * self.masses[atom_number]**0.5
                            Y=float(lines_in_file[i+1].split()[j+1]) * self.masses[atom_number]**0.5
                            Z=float(lines_in_file[i+2].split()[j+1]) * self.masses[atom_number]**0.5
                            self.temp_normal_coordinates[normal_mode_numbers[j]][atom_number]=[X,Y,Z]
                        i+=3
                #remove normal modes with 0,i.e., elements that are equal to normal_coordinate_zero
                normal_coordinate_zero=[ [0.0,0.0,0.0] for k in range(0,len(self.atomic_symbols)) ]
                self.temp_normal_coordinates[:]=[normal_coordinate for normal_coordinate in self.temp_normal_coordinates if normal_coordinate != normal_coordinate_zero]

            if "Final Gibbs free energy         ..." in lines_in_file[i]:
                self.thermochemistry_calculations.append(structure_number)
                self.temp_gibbs_free_energy=float(lines_in_file[i].split()[5])
            if "Zero point energy                ..." in lines_in_file[i]:
                self.temp_zero_point_energy= float(lines_in_file[i].split("...")[1].split()[0]) + float(lines_in_file[i-1].split("...")[1].split()[0]) 
                
                                                    
            #read any of the properties in the list passed to the constructor:
            for prop in properties:
                read_property=prop.read(lines_in_file,i)
                if read_property!=None:
                    self.temp_property[prop.name]=read_property


            #flush everything after each optimization cycle         
            #if clcn_type=="geom_opt":
            if "*                GEOMETRY OPTIMIZATION CYCLE" in lines_in_file[i] and int(lines_in_file[i].split("GEOMETRY OPTIMIZATION CYCLE")[1].split("*")[0])>1: 
                    self.flush_local_properties()
                    structure_number+=1
                    #if lines_in_file[i].find("***        THE OPTIMIZATION HAS CONVERGED     ***")>-1: self.stationary_points=structure_number
                
            if  "***        THE OPTIMIZATION HAS CONVERGED     ***" in lines_in_file[i]: self.stationary_points=structure_number+1

            if "$$$$$$$$$$$$$$$$  JOB NUMBER" in lines_in_file[i] or "****ORCA TERMINATED NORMALLY****" in lines_in_file[i]:
                    self.flush_local_properties()
                    self.reset_local_properties()


            
            #elif clcn_type=="single_point" or clcn_type=="e_grad":
            #if "****ORCA TERMINATED NORMALLY****" in lines_in_file[i]:
            #        self.flush_local_properties()
            #        self.reset_local_properties()
                    
            #        structure_number=1
            i+=1
        
        #things to do at the end:
        #transform self.dipole_moments to np.array
        self.dipole_moments=np.array(self.dipole_moments)
        #close file
        file.close()

    def read_frequencies(self,outfile=""):

        if outfile=="": outfile=self.outfile
        file=open(self.outfile,"r")
        lines_in_file=file.readlines()
        for i in range(0,len(lines_in_file)):
            #reset everything if a link1 hessian job is at the end
            if "ORCA SCF HESSIAN" in lines_in_file[i]:
                self.frequencies=[]
                self.red_masses=[]
                #self.masses=[]   no reset this!
                self.frc_const=[]
                self.model_syst_percent=[]
                self.real_syst_percent=[]
                self.ir_intensities=[]
                
                self.normal_coordinates=[]
            if (lines_in_file[i].startswith("VIBRATIONAL FREQUENCIES")):
                i+=3
                if (lines_in_file[i].startswith("Scaling factor for frequencies =")):
                    scaling_factor_for_freq=float(lines_in_file[i].split()[5])
                i+=2
                while ( len(lines_in_file[i])>3 and  str.isdigit(lines_in_file[i][3]) ):
                    f=float(lines_in_file[i].split()[1])
                    #do not store the first 6(5) 0.00 cm**-1 frequencies 
                    #if f>0.00: self.frequencies.append(float(lines_in_file[i].split()[1]))
                    # remove the frist 6 (5) normal modes (full of zeroes) later
                    self.frequencies.append(float(lines_in_file[i].split()[1]))
                    i+=1
            if (lines_in_file[i].startswith("CARTESIAN COORDINATES (A.U.)")):
                self.masses=[]
                i+=3
                while (lines_in_file[i]!="\n" and str.isdigit(lines_in_file[i].split()[0])): 
                    self.masses.append(float(lines_in_file[i].split()[4]))
                    i+=1

            if (lines_in_file[i].startswith("NORMAL MODES")):
                #initialize normal_coordinates
                self.normal_coordinates=[ [[0.0,0.0,0.0] for k in range(0,len(self.masses))] for kk in range(0,len(self.frequencies)) ]
                i+=7
                while all([str.isdigit(n) for n in lines_in_file[i].split()]):
                    number_of_normal_modes_per_line=len(lines_in_file[i].split())
                    normal_mode_numbers=[int(x) for x in lines_in_file[i].split()]
                    i+=1
                    while (lines_in_file[i].strip()!="" and str.isdigit(lines_in_file[i][6])):
                        for j in range(0,number_of_normal_modes_per_line-1):
                            atom_number=int(lines_in_file[i].split()[0])/3
                            X=float(lines_in_file[i].split()[j+1]) * self.masses[atom_number]**0.5
                            Y=float(lines_in_file[i+1].split()[j+1]) * self.masses[atom_number]**0.5
                            Z=float(lines_in_file[i+2].split()[j+1]) * self.masses[atom_number]**0.5
                            self.normal_coordinates[normal_mode_numbers[j]][atom_number]=[X,Y,Z]
                        i+=3
                #remove normal modes with 0,i.e., elements that are equal to normal_coordinate_zero
                normal_coordinate_zero=[ [0.0,0.0,0.0] for k in range(0,len(self.masses)) ]
                self.normal_coordinates[:]=[normal_coordinate for normal_coordinate in self.normal_coordinates if normal_coordinate != normal_coordinate_zero]



class Molecular_structure:
        


        # each atom is an object    
        class Atom():
            symbol=""
            #a list of the atoms that are connected to this; each element is a list: [atom_number,bond order,atom symbol]
            connection=[]
            back_up_connection=[]
            atom_type=""
            atom_type_warning=""
            #the cartesian coordinate of the atom, in numpy array format
            coord=np.empty(3)
            possible_atom_type=[] # NOT USED
            atom_number=0 #not atomic number!
            atom_layer="" # the layer in oniom calculations
            interface=0 # if the atom links LL layer with HL layer, store the number of the atom of the HL layer that is connected to it
            charge=0.0  # how to read it from the input file?
            mulliken_charge=0.0
            chelpg_charge=0.0
            NBO_charge=0.0
            loewdin_charge=0.0
            hirshfeld_charge=0.0
            mayer_pop_analysis={}
            mayer_bond_orders=[]
            mulliken_charge_standarized=0.0
            chelpg_charge_standarized=0.0
            NBO_charge_standarized=0.0
            hirshfeld_charge_standarized=0.0
            chemical_isotropic_shield=0.0
            properties={} #any other atomic property can be specified in this dictionary


            # an array to include iteratively characteristics of the atom
            # it contains vectors with the format: [nC,nN,nO,nS,nP,nX,stereo] where: nH: number of H; nC: number of C;... nX: number of any other element
            # the first element is the symbol ([1,0,0,0,0,0,0] for a C atom; [0,1,0,0,0,0,0] for a O atom, [0,0,0,0,0,0,0] for H atom...)
            # the second element is the atoms joint to this atom ([2,0,0,0,0,0,0] the atom is joint to 2 C atoms and to 2 H atoms) 
            # the third is the atoms joint to the atoms joint to this atom....
            # the last element of the list, stereo, will be defined later to differenciate atoms that are equivalent by the connectivity alone
            fingerprint=[]


            def __init__(self, symbol,coord,atom_number, atom_type="", atom_type_warning="undefined atom type", atom_layer="H"):
                self.connection=[]
                self.symbol=symbol
                self.coord=coord
                self.atom_number=atom_number
                self.atom_type=atom_type
                self.atom_type_warning=atom_type_warning
                self.atom_layer=atom_layer
                self.mayer_bond_orders=[]
                self.properties={}

            # a method for deep copying the atom
            def copy(self,fast=True):
                new_atom= atom(self.symbol,copy.deepcopy(self.coord),self.atom_number,self.atom_type,self.atom_type_warning,self.atom_layer)
                if not fast: 
                    new_atom.charge=self.charge
                    new_atom.interface=self.interface
                    new_atom.possible_atom_type=self.possible_atom_type
                new_atom.connection=[ [cc for cc in c] for c in self.connection]
                new_atom.fingerprint=[ [ff for ff in f] for f in self.fingerprint()]
                return new_atom

            #for debuggin
            def print_atom(self):
                text="atom object with the following properties:"
                text+="\natom_number: "+str(self.atom_number)
                text+="\nsymbol: "+self.symbol
                for c in self.connection: text+="\nconnected to atom: "+ str(c[0])+"("+str(c[2])+") with order: "+ str(c[1])  
                text+="\n"
                #text+="fingerprint:\n"
                #for c in self.fingerprint: text+=" | "+str(c).strip("[]")
                return text    

        #for ORCA NEB calculations, reading the MEP xyz file:
        class MEP_xyz:

            xyz_file=""
            cart_coordinates=[]
            energies=np.array([],dtype=float)
            rel_energies=[]
            atom_symbol_list=[]
            MEP_distances=[]
            max_forces=[]
            rms_forces=[]

            # covalent radii
            # From: Beatriz Cordero; Verónica Gómez; Ana E. Platero-Prats; Marc Revés; Jorge Echeverría; 
            # Eduard Cremades; Flavia Barragán; Santiago Alvarez (2008). 
            # "Covalent radii revisited". Dalton Trans. (21): 2832–2838. doi:10.1039/b801115j.
            r_cov={
                    "h":0.31,"li":1.28,"na":1.66,"k":2.03,"rb":2.20,"cs":2.44,"fr":2.6,
                    "be":0.96,"mg":1.41,"ca":1.76,"sr":1.95,"ba":2.15,"ra":2.21,
                    "sc":1.70,"y": 1.90,"la":2.07,"ac":2.15,
                    "ti":1.6,"zr":1.75,"hf":1.87,
                    "v":1.53,"nb":1.64,"ta":1.70,
                    "cr":1.39,"mo":1.54,"w":1.62,
                    "mn":1.39,"tc":1.47,"re":1.51,
                    "fe":1.32,"co":1.26,"ni":1.24,"ru":1.46,"rh":1.42,"pd":1.39,"os":144,"ir":77,"pt":136,
                    "cu":1.32,"ag":1.45,"au":1.36,
                    "zn":1.22,"cd":1.44,"hg":1.32,
                    "b":0.84,"al":1.21,"ga":1.22,"in":1.42,"tl":1.45,
                    "c":0.76,"si":1.11,"ge":1.20,"sn":1.39,"pb":1.46,
                    "n":0.71,"p":1.07,"as":1.19,"sb":1.39,"bi":1.48,
                    "o":0.66,"s":1.05,"se":1.20,"te":1.38,"po":1.40,
                    "f":0.57,"cl":1.02,"br":1.20,"i":1.39,"at":1.50,
                    "he":0.28,"ne":0.58,"ar":1.06, "kr":1.16,"xe":1.40,"rn":1.5,
                    "ce":1.04,"pr":2.03,"nd":2.01,"pm":1.99,"sm":1.98, "eu":1.98,"gd":1.96,"tb":1.94,"dy":1.92,"ho":1.92,"er":1.89,"tm":1.90,"yb":1.87,"lu":1.75,
                    "th":2.06,"pa":2.00,"u":1.96,"np":1.90,"pu":1.87,"am":1.8,"cm":1.69}


            def dist(self,atoms,structure):

                if len(atoms)==4:

                    if type(atoms)==list and type(atoms[0])==int:
                        q1=np.array(self.cart_coordinates[structure-1][atoms[1]-1])-np.array(self.cart_coordinates[structure-1][atoms[0]-1])
                        q2=np.array(self.cart_coordinates[structure-1][atoms[2]-1])-np.array(self.cart_coordinates[structure-1][atoms[1]-1])
                        q3=np.array(self.cart_coordinates[structure-1][atoms[3]-1])-np.array(self.cart_coordinates[structure-1][atoms[2]-1])

                    n11=np.cross(q1,q2)
                    n1=n11/(np.sum(n11**2)**0.5)
                    n22=np.cross(q2,q3)
                    n2=n22/(np.sum(n22**2)**0.5)
                    u1=n2.copy()
                    u3=q2/(np.sum(q2**2)**0.5)
                    u2=np.cross(u1,u3)
                    return (180/math.pi)*math.atan2(np.dot(n1,u2),np.dot(n1,n2))

                if len(atoms)==3:
                    if type(atoms)==list and type(atoms[0])==int:                
                        q1=np.array(self.cart_coordinates[structure-1][atoms[1]-1])-np.array(self.cart_coordinates[structure-1][atoms[0]-1])
                        q2=np.array(self.cart_coordinates[structure-1][atoms[1]-1])-np.array(self.cart_coordinates[structure-1][atoms[2]-1])
                                
                    c=np.dot(q1,q2)/((np.sum(q1**2)*np.sum(q2**2)))**0.5
                    s=np.sum( np.cross(q1,q2)**2)**0.5 /((np.sum(q1**2)*np.sum(q2**2)))**0.5
                    return (180/math.pi)*math.atan2(s,c)
                
                if len(atoms)==2:
                    if type(atoms)==list and type(atoms[0])==int:
                        sum_of_cov_radii= self.r_cov[self.atom_symbol_list[atoms[1]-1].lower()]+self.r_cov[self.atom_symbol_list[atoms[0]-1].lower()]
                        distance_between_atoms=np.sum( (np.array(self.cart_coordinates[structure-1][atoms[1]-1])-np.array(self.cart_coordinates[structure-1][atoms[0]-1]))**2)**0.5
                        if distance_between_atoms>2*sum_of_cov_radii: return 0
                        else: return 2-distance_between_atoms/sum_of_cov_radii
                        #return ((np.sum( (np.array(self.cart_coordinates[structure-1][atoms[1]-1])-np.array(self.cart_coordinates[structure-1][atoms[0]-1]))**2)**0.5)/sum_of_cov_radii)**-1


            def __init__(self,xyz_file,out_file=""):

                self.cart_coordinates=[]
                self.rel_energies=[]
                self.inc_rel_energies=[]
                self.atom_symbol_list=[]
                self.MEP_distances=[]
                self.inc_MEP_distances=[]
                self.max_forces=[]
                rms_forces=[]

                with open (xyz_file,"r") as file: lines=file.readlines()
                number_of_atoms=int(lines[0])
                i=0
                while i<len(lines):
                    if lines[i].startswith("Coordinates from ORCA-job"):
                        self.energies=np.hstack([self.energies,float( lines[i].split()[-1] ) ]  )
                        coord=[]
                        atoms=[]
                        for j in range(0,number_of_atoms):
                            i+=1
                            j+=1
                            coord.append([float(lines[i].split()[1]),float(lines[i].split()[2]),float(lines[i].split()[3])])
                            atoms.append(str(lines[i].split()[0]))
                        self.cart_coordinates.append(coord)
                        self.atom_symbol_list=atoms
                    i+=1
                if out_file=="": out_file=xyz_file.split("_MEP_trj.xyz")[0]+".out"
                if os.path.isfile(out_file):
                    with open (out_file,"r") as file: lines=file.readlines()
                    i=0
                    while i<len(lines):
                        i+=1
                        if "PATH SUMMARY " in lines[i]:
                            i+=5
                            while lines[i].strip()!="":
                                w=lines[i].split()
                                if len(self.MEP_distances)>0:  self.inc_MEP_distances.append(float(w[1])-self.MEP_distances[-1])
                                else: self.inc_MEP_distances.append(0.0)
                                self.MEP_distances.append(float(w[1]))
                                if len(self.rel_energies)>0: self.inc_rel_energies.append(float(w[3])-self.rel_energies[-1])
                                else: self.inc_rel_energies.append(0.0)
                                self.rel_energies.append(float(w[3]))
                                self.max_forces.append(float(w[4]))
                                self.rms_forces.append(float(w[5]))
                                i+=1
                            break
                if self.rel_energies==[]:
                    for e in self.energies: 
                        self.rel_energies.append(627.5095*(e-min(self.energies)))
                self.inc_rel_energies.append(0.0)
                if self.MEP_distances!=[]:
                    for i in range(1,len(self.rel_energies)):
                        self.inc_rel_energies.append(self.rel_energies[i]-self.MEP_distances[i-1])

            

            def csv_report(self,csv_file_name="",distances=[],images=[]): 

                if csv_file_name!="":
                    if csv_file_name[-4:]!=".csv": csv_file_name+=".csv"
                
                if images==[]: images=range(1,len(self.rel_energies)+1)

                n_prev_image=0
                X_prev=0.0
                energy_prev=0.0
                csv_lines=[]
                csv_text=""

                if csv_file_name!="":
                    if os.path.isfile(csv_file_name):
                        with open(csv_file_name,"r") as f: csv_lines=f.readlines()
                    if len(csv_lines)>1:
                        #get the definitions of distances from the header in the csv file
                        if distances==[]:
                            w=csv_lines[0].split(",")[3:-1]
                            for d in w: distances.append([int(a) for a in d.split("---")])
                        n_prev_image=int(csv_lines[-1].split(",")[0])
                        X_prev=float(csv_lines[-1].split(",")[1])
                        energy_prev=float(csv_lines[-1].split(",")[2])
                        
                counter=0
                if len(csv_lines)==0:
                    csv_text="structure,dist. along MEP,energy,,"
                    for d in distances:
                        csv_text+="---".join([str(dd) for dd in d])+","
                    csv_text+="\n"
                else: 
                    for csv_line in csv_lines: csv_text+=csv_line

                #for energy,X in zip(self.rel_energies, self.MEP_distances):
                for i in images:
                    energy=self.inc_rel_energies[i-1]+energy_prev
                    energy_prev=energy
                    abs_energy=self.energies[i-1]
                    X=self.inc_MEP_distances[i-1]+X_prev
                    X_prev=X
                    counter+=1
                    csv_text+=str(counter+n_prev_image)+","+str(X)+","+str(energy)+","+str(abs_energy)+","
                    for dist in distances: csv_text+=str(self.dist(dist,counter))+","
                    csv_text+="\n"
                    
                
                if csv_file_name!="":
                    with open(csv_file_name,"w") as f: f.write(csv_text)
                
                return csv_text

        cart_hess=np.empty(0)
        int_hess=np.empty(0)
        normal_modes=[]



        # SOME DICTIONARIES
        # covalent radii
        # From: Beatriz Cordero; Verónica Gómez; Ana E. Platero-Prats; Marc Revés; Jorge Echeverría; 
        # Eduard Cremades; Flavia Barragán; Santiago Alvarez (2008). 
        # "Covalent radii revisited". Dalton Trans. (21): 2832–2838. doi:10.1039/b801115j.
        r_cov={
                "h":0.31,"li":1.28,"na":1.66,"k":2.03,"rb":2.20,"cs":2.44,"fr":2.6,
                "be":0.96,"mg":1.41,"ca":1.76,"sr":1.95,"ba":2.15,"ra":2.21,
                "sc":1.70,"y": 1.90,"la":2.07,"ac":2.15,
                "ti":1.6,"zr":1.75,"hf":1.87,
                "v":1.53,"nb":1.64,"ta":1.70,
                "cr":1.39,"mo":1.54,"w":1.62,
                "mn":1.39,"tc":1.47,"re":1.51,
                "fe":1.32,"co":1.26,"ni":1.24,"ru":1.46,"rh":1.42,"pd":1.39,"os":144,"ir":77,"pt":136,
                "cu":1.32,"ag":1.45,"au":1.36,
                "zn":1.22,"cd":1.44,"hg":1.32,
                "b":0.84,"al":1.21,"ga":1.22,"in":1.42,"tl":1.45,
                "c":0.76,"si":1.11,"ge":1.20,"sn":1.39,"pb":1.46,
                "n":0.71,"p":1.07,"as":1.19,"sb":1.39,"bi":1.48,
                "o":0.66,"s":1.05,"se":1.20,"te":1.38,"po":1.40,
                "f":0.57,"cl":1.02,"br":1.20,"i":1.39,"at":1.50,
                "he":0.28,"ne":0.58,"ar":1.06, "kr":1.16,"xe":1.40,"rn":1.5,
                "ce":1.04,"pr":2.03,"nd":2.01,"pm":1.99,"sm":1.98, "eu":1.98,"gd":1.96,"tb":1.94,"dy":1.92,"ho":1.92,"er":1.89,"tm":1.90,"yb":1.87,"lu":1.75,
                "th":2.06,"pa":2.00,"u":1.96,"np":1.90,"pu":1.87,"am":1.8,"cm":1.69}

        # Dictionary of all elements matched with their atomic masses. Thanks to
        # https://gist.github.com/lukasrichters14/c862644d4cbcf2d67252a484b7c6049c
        atomic_weights = {
                'h' : 1.008,'he' : 4.003, 'li' : 6.941, 'be' : 9.012,
                'b' : 10.811, 'c' : 12.011, 'n' : 14.007, 'o' : 15.999,
                'f' : 18.998, 'ne' : 20.180, 'na' : 22.990, 'mg' : 24.305,
                'al' : 26.982, 'si' : 28.086, 'p' : 30.974, 's' : 32.066,
                'cl' : 35.453, 'ar' : 39.948, 'k' : 39.098, 'ca' : 40.078,
                'sc' : 44.956, 'ti' : 47.867, 'v' : 50.942, 'cr' : 51.996,
                'mn' : 54.938, 'fe' : 55.845, 'co' : 58.933, 'ni' : 58.693,
                'cu' : 63.546, 'zn' : 65.38, 'ga' : 69.723, 'ge' : 72.631,
                'as' : 74.922, 'se' : 78.971, 'br' : 79.904, 'kr' : 84.798,
                'rb' : 84.468, 'sr' : 87.62, 'y' : 88.906, 'zr' : 91.224,
                'nb' : 92.906, 'mo' : 95.95, 'tc' : 98.907, 'ru' : 101.07,
                'rh' : 102.906, 'pd' : 106.42, 'ag' : 107.868, 'cd' : 112.414,
                'in' : 114.818, 'sn' : 118.711, 'sb' : 121.760, 'te' : 126.7,
                'i' : 126.904, 'xe' : 131.294, 'cs' : 132.905, 'ba' : 137.328,
                'la' : 138.905, 'ce' : 140.116, 'pr' : 140.908, 'nd' : 144.243,
                'pm' : 144.913, 'sm' : 150.36, 'eu' : 151.964, 'gd' : 157.25,
                'tb' : 158.925, 'dy': 162.500, 'ho' : 164.930, 'er' : 167.259,
                'tm' : 168.934, 'yb' : 173.055, 'lu' : 174.967, 'hf' : 178.49,
                'ta' : 180.948, 'w' : 183.84, 're' : 186.207, 'os' : 190.23,
                'ir' : 192.217, 'pt' : 195.085, 'au' : 196.967, 'hg' : 200.592,
                'tl' : 204.383, 'pb' : 207.2, 'bi' : 208.980, 'po' : 208.982,
                'at' : 209.987, 'rn' : 222.081, 'fr' : 223.020, 'ra' : 226.025,
                'ac' : 227.028, 'th' : 232.038, 'pa' : 231.036, 'u' : 238.029,
                'np' : 237, 'pu' : 244, 'am' : 243, 'cm' : 247, 'bk' : 247,
                'ct' : 251, 'es' : 252, 'fm' : 257, 'md' : 258, 'no' : 259,
                'lr' : 262, 'rf' : 261, 'db' : 262, 'sg' : 266, 'bh' : 264,
                'hs' : 269, 'mt' : 268, 'ds' : 271, 'rg' : 272, 'cn' : 285,
                'nh' : 284, 'fl' : 289, 'mc' : 288, 'lv' : 292, 'ts' : 294,
                'og' : 294}

        # Dictionary of all elements matched with their atomic numbers. 
        atomic_numbers = {
                'h' : 1,'he' : 2, 'li' : 3, 'be' : 4, 'b' : 5, 'c' : 6, 'n' : 7, 'o' : 8,
                'f' : 9, 'ne' : 10, 'na' : 11, 'mg' : 12, 'al' : 13, 'si' : 14, 'p' : 15, 's' : 16,
                'cl' : 17, 'ar' : 18, 'k' : 19, 'ca' : 20, 'sc' : 21, 'ti' : 22, 'v' : 23, 'cr' : 24,
                'mn' : 25, 'fe' : 26, 'co' : 27, 'ni' : 28, 'cu' : 29, 'zn' : 30, 'ga' : 31, 'ge' : 32,
                'as' : 33, 'se' : 34, 'br' : 35, 'kr' : 36, 'rb' : 37, 'sr' : 38, 'y' : 39, 'zr' : 40,
                'nb' : 41, 'mo' : 42, 'tc' : 43, 'ru' : 44, 'rh' : 45, 'pd' : 46, 'ag' : 47, 'cd' : 48,
                'in' : 49, 'sn' : 50, 'sb' : 51, 'te' : 52, 'i' : 53, 'xe' : 54, 'cs' : 55, 'ba' : 56,
                'la' : 57, 'ce' : 58, 'pr' : 59, 'nd' : 60, 'pm' : 61, 'sm' : 62, 'eu' : 63, 'gd' : 64,
                'tb' : 65, 'dy': 66, 'ho' : 67, 'er' : 68, 'tm' : 69, 'yb' : 70, 'lu' : 71, 'hf' : 72,
                'ta' : 73, 'w' : 74, 're' : 75, 'os' : 76, 'ir' : 777, 'pt' : 78, 'au' : 79, 'hg' : 80,
                'tl' : 81, 'pb' : 82, 'bi' : 83, 'po' :84, 'at' : 85, 'rn' : 86, 'fr' : 87, 'ra' : 88,
                'ac' : 89, 'th' : 90, 'pa' : 91, 'u' : 92, 'np' : 93, 'pu' : 94, 'am' : 95, 'cm' : 96, 'bk' : 97,
                'ct' : 98, 'es' : 99, 'fm' : 100, 'md' : 101, 'no' : 102,  'lr' : 103, 'rf' : 104, 'db' : 105,
                'sg' : 106, 'bh' : 107, 'hs' : 108, 'mt' : 109, 'ds' : 110, 'rg' : 111, 'cn' : 112,
                'nh' : 113, 'fl' : 114, 'mc' : 115, 'lv' : 116, 'ts' : 117,  'og' : 118}

        #Van-Der-Waals parameters in UFF force field, according to Table 1 in JACS 114,(25),10024,1992, in format [distance,potential_well,scale]
        uff_vdw={
                "h":[2.886,0.044,12.0],"li":[2.451,0.025,12.0],"na":[2.983,0.03,12.0],"k":[3.812,0.035,12.0],"rb":[4.114,0.04,12.0],"cs":[4.517,0.045,12.0],"fr":[4.9,0.05,12.0],
                "be":[2.745,0.085,12.0],"mg":[3.021,0.111,12.0],"ca":[3.399,0.238,12.0],"sr":[3.641,0.235,12.0],"ba":[3.703,0.364,12.0],"ra":[3.677,0.404,12.],
                "sc":[3.295,0.019,12.0],"y": [3.345,0.072,12.0],"la":[3.522,0.017,12.0],"ac":[3.478,0.033,12.0],
                "ti":[3.175,0.017,12.0],"zr":[3.124,0.069,12.0],"hf":[3.141,0.072,12.0],
                "v":[3.144,0.016,12.0],"nb":[3.165,0.059,12.0],"ta":[3.170,0.081,12.0],
                "cr":[3.023,0.015,12.0],"mo":[3.052,0.056,12.0],"w":[3.069,0.067,12],
                "mn":[2.961,0.013,12.0],"tc":[2.998,0.048,12.0],"re":[2.954,0.066,12.0],
                "fe":[2.912,0.013,12.0],"co":[2.872,0.014,12.0],"ni":[2.834,0.015,12.0],
                "ru":[2.963,0.056,12.0],"rh":[2.929,0.053,12.0],"pd":[2.899,0.048,12.0],
                "os":[3.120,0.037,12.0],"ir":[2.840,0.073,12.0],"pt":[2.754,0.08,12.0],
                "cu":[3.495,0.005,12.0],"ag":[3.148,0.036,12.0],"au":[3.239,0.039,12.0],
                "zn":[2.763,0.124,12.0],"cd":[2.848,0.228,12.0],"hg":[2.705,0.385,12.0],
                "b":[4.083,0.18,12.052],"al":[4.499,0.505,11.278],"ga":[4.383,0.415,11.0],"in":[4.463,0.599,11.0],"tl":[4.347,0.680,11.0],
                "c":[3.851,0.105,12.73],"si":[4.295,0.402,12.175],"ge":[4.280,0.379,12.0],"sn":[4.392,0.567,12.0],"pb":[4.297,0.663,12.0],
                "n":[3.660,0.069,13.407],"p":[4.147,0.305,13.072],"as":[4.230,0.309,13.0],"sb":[4.420,0.449,13.0],"bi":[4.370,0.518,13.0],
                "o":[3.500,0.060,14.085],"s":[4.035,0.274,13.969],"se":[4.205,0.291,14.0],"te":[4.470,0.398,14.0],"po":[4.709,0.325,14.0],
                "f":[3.364,0.05,14.762],"cl":[3.947,0.227,14.866],"br":[4.189,0.251,15.0],"i":[4.5,0.339,15.0],"at":[4.750,0.284,15.0],
                "he":[2.362,0.056,15.24],"ne":[3.243,0.042,15,44],"ar":[3.868,0.185,15.763],"kr":[4.141,0.220,16.0],"xe":[4.404,0.332,12.0],"rn":[4.765,0.248,15.0],
                "ce":[3.556,0.013,12.0],"pr":[3.606,0.01,12.0],"nd":[3.575,0.01,12.0],"pm":[3.547,0.009,12.0],"sm":[3.520,0.08,12.0], "eu":[3.493,0.008,12.0],"gd":[3.368,0.009,12.0],"tb":[3.451,0.007,12.0],"dy":[3.428,0.007,12.0],"ho":[3.409,0.007,12.0],"er":[3.391,0.007,12.0],"tm":[3.374,0.006,12.0],"yb":[3.355,0.228,12.0],"lu":[3.640,0.041,12.0],
                "th":[3.396,0.026,12.0],"pa":[3.424,0.022,12.0],"u":[3.395,0.022,12.0],"np":[3.424,0.019,12.0],"pu":[3.424,0.016,12.0],"am":[3.381,0.014,12.0],"cm":[3.326,0.012,12.0],"bk":[3.339,0.013,12.0],"cf":[3.313,0.013,12.0],"es":[3.299,0.012,12.0],"fm":[3.286,0.012,12.0],"md":[3.274,0.011,12.0],"no":[3.248,0.011,12.0],"lw":[3.236,0.011,12.0]
                }

        ##Van-Der-Waals parameters in MM3 force field, from: Lii, J. H.; Allinger, N. L.
        # Molecular mechanics. The MM3 force field for hydrocarbons. 3. The van der Waals' potentials and crystal data for aliphatic and aromatic hydrocarbons. 
        # J. Am. Chem. Soc. 1989, 111 (23), 8576-8582.
        mm3_vdw={
            "1":[2.0400,   0.0270],"2":[1.9600,   0.0560],"3":[1.9400,   0.0560],"4":[1.9400,   0.0560],"5":[1.6200,   0.0200, ],
            "6":[1.8200,   0.0590],"7":[1.8200,   0.0590],"8":[1.9300,   0.0430],"9":[1.9300,   0.0430],"10":[1.9300,   0.0430],
            "11":[1.7100,   0.0750],"12":[2.0700,   0.2400],"13":[2.2200,   0.3200],"14":[2.3600,   0.4240],"15":[2.1500,   0.2020],
            "16":[2.1500,   0.2020],"17":[2.1500,   0.2020],"18":[2.1500,   0.2020],"19":[2.2900,   0.1400],"20":[0.0000,   0.0000],
            "21":[1.6000,   0.0160],"22":[1.9400,   0.0450],"23":[1.6000,   0.0180],"24":[0.9000,   0.0150],"25":[2.2200,   0.1680],
            "26":[2.1500,   0.0140],"27":[2.1500,   0.0140],"28":[1.6000,   0.0150],"29":[1.9400,   0.0300],"30":[1.9400,   0.0560],
            "31":[2.4400,   0.2000],"32":[2.5900,   0.2700],"33":[2.7400,   0.3400],"34":[2.2900,   0.2760],"35":[2.4400,   0.3680],
            "36":[1.6170,   0.0200],"37":[1.9300,   0.0430],"38":[1.9600,   0.0560],"39":[1.9300,   0.0430],"40":[1.9300,   0.0430],
            "41":[1.8200,   0.0590],"42":[2.1500,   0.2020],"43":[1.9300,   0.0430],"44":[1.6200,   0.0200],"45":[1.9300,   0.0430],
            "46":[1.9300,   0.0430],"47":[1.8200,   0.0590],"48":[1.6000,   0.0340],"49":[1.8200,   0.0590],"50":[1.9600,   0.0560],
            "51":[1.5300,   0.0260],"52":[1.6000,   0.0900],"53":[1.9900,   0.2680],"54":[2.1500,   0.3580],"55":[2.2800,   0.4950],
            "56":[2.0400,   0.0270],"57":[1.9600,   0.0560],"58":[1.9400,   0.0560],"59":[2.2000,   0.0150],"60":[2.2000,   0.1680],
            "61":[2.2000,   0.0200],"62":[2.2000,   0.0200],"63":[2.2000,   0.0200],"64":[2.2000,   0.0200],"65":[2.2000,   0.0200],
            "66":[2.2000,   0.0200],"67":[1.9400,   0.0560],"68":[1.9400,   0.0560],"69":[1.8200,   0.0590],"70":[1.8200,   0.0590],
            "71":[1.9400,   0.0560],"72":[1.9300,   0.0430],"73":[1.6000,   0.0150],"74":[2.1500,   0.2020],"75":[1.8200,   0.0590],
            "76":[1.8200,   0.0590],"77":[1.8200,   0.0590],"78":[1.8200,   0.0590],"79":[1.8200,   0.0590],"80":[1.8200,   0.0590],
            "81":[1.8200,   0.0590],"82":[1.8200,   0.0590],"83":[1.8200,   0.0590],"84":[1.8200,   0.0590],"85":[1.8200,   0.0590],
            "86":[1.8200,   0.0590],"87":[1.8200,   0.0590],"88":[1.8200,   0.0590],"89":[1.8200,   0.0590],"90":[1.8200,   0.0590],
            "91":[1.8200,   0.0590],"92":[1.8200,   0.0590],"93":[1.8200,   0.0590],"94":[1.8200,   0.0590],"95":[1.8200,   0.0590],
            "96":[1.8200,   0.0590],"97":[1.8200,   0.0590],"98":[1.8200,   0.0590],"99":[1.8200,   0.0590],"100":[1.8200,   0.0590],
            "101":[1.8200,   0.0590],"102":[1.8200,   0.0590],"103":[1.8200,   0.0590],"104":[2.1500,   0.2020],"105":[2.1500,   0.2020],
            "106":[1.9400,   0.0560],"107":[1.9300,   0.0430],"108":[1.9300,   0.0430],"109":[1.9300,   0.0430],"110":[1.9400,   0.0440],
            "111":[1.9300,   0.0430],"112":[0.0000,   0.0000],"113":[1.9600,   0.0560],"114":[1.9600,   0.0560],"115":[1.8200,   0.0590],
            "116":[1.8200,   0.0590],"117":[1.8200,   0.0590],"118":[1.8200,   0.0590],"119":[1.8200,   0.0590],"120":[1.8200,   0.0590],
            "121":[1.8200,   0.0590],"122":[0.0000,   0.0000],"123":[0.0000,   0.0000],"124":[1.6200,   0.0200],"125":[2.8100,   0.1340],
            "126":[3.0000,   0.1850],"127":[3.0700,   0.2330],"128":[2.7800,   0.3120],"129":[2.7400,   0.3400],"130":[2.7300,   0.3490],
            "131":[2.7300,   0.3490],"132":[2.7200,   0.3580],"133":[2.7100,   0.3650],"134":[2.9400,   0.2710],"135":[2.7100,   0.3650],
            "136":[2.7000,   0.3720],"137":[2.6900,   0.3790],"138":[2.6700,   0.3930],"139":[2.6700,   0.3930],"140":[2.6700,   0.3930],
            "141":[2.7900,   0.2980],"142":[2.6500,   0.4070],"143":[1.9300,   0.0430],"144":[1.9300,   0.0430],"145":[1.8200,   0.0590],
            "146":[1.9300,   0.0430],"147":[0.0000,   0.0000],"148":[1.8200,   0.0600],"149":[1.8200,   0.0600],"150":[1.9300,   0.0430],
            "151":[1.9300,   0.0430],"152":[0.0000,   0.0000],"153":[2.2200,   0.1680],"154":[2.1500,   0.2020],"155":[1.9300,   0.0430],
            "159":[1.8200,   0.0590],"160":[1.9600,   0.0560],"161":[1.9600,   0.0560],"162":[1.9400,   0.0560],"163":[2.5500,   0.0070],
            "164":[1.9300,   0.0430]
            }
        mm3_vdw_pairs={
            "1-5":[ 3.5600, 0.0230] ,  "1-36":[ 3.5570, 0.0230] ,  "19-113":[ 2.4500,15.0000] ,  "19-114":[ 2.4500,15.0000] ,  "31-113":[ 2.5000,15.0000] ,  
            "31-114":[ 2.5000,15.0000] ,  "32-113":[ 2.7100,15.0000] ,  "32-114":[ 2.7100,15.0000] ,  "33-113":[ 2.7800,15.0000] ,  
            "33-114":[ 2.7800,15.0000] ,  "61-113":[ 2.0600,15.0000] ,  "61-114":[ 2.0600,15.0000] ,  "63-113":[ 2.2000,15.0000] ,  
            "63-114":[ 2.2000,15.0000] ,  "65-113":[ 2.1200,15.0000] ,  "65-114":[ 2.1200,15.0000] ,  "113-125":[ 2.6100,15.0000] ,  
            "113-126":[ 2.7500,15.0000] ,  "113-127":[ 2.9000,15.0000] ,  "113-133":[ 2.7900,15.0000] ,  "113-134":[ 2.7900,15.0000] ,  
            "113-141":[ 2.6200,15.0000] ,  "114-125":[ 2.6100,15.0000] ,  "114-126":[ 2.7500,15.0000] ,  "114-127":[ 2.9000,15.0000] ,  
            "114-133":[ 2.7900,15.0000] ,  "114-134":[ 2.7900,15.0000] ,  "114-141":[ 2.6200,15.0000] ,
            "2-21":[ 2.6500,0.5500], "2-23":[ 3.0000,0.1000], "2-28":[ 2.6200,0.8600], "2-44":[ 2.8300,0.2000], "2-73":[ 3.0000,0.7500], 
            "2-124":[ 2.9600,0.2500], "4-21":[ 2.6000,0.5100], "4-23":[ 2.8700,0.2600], "4-28":[ 2.5400,0.9300], "4-124":[ 2.8200,0.3400], 
            "6-21":[ 2.1100,3.0000], "6-23":[ 2.3800,1.3000], "6-24":[ 2.1400,3.3000], "6-28":[ 2.0300,5.1400], "6-44":[ 2.6800,0.1500], 
            "6-48":[ 1.7200,13.8000], "6-73":[ 2.2500,3.1500], "6-124":[ 2.3000,1.7000], "7-21":[ 2.0700,2.5500], "7-23":[ 2.3900,1.1000], 
            "7-24":[ 2.0500,3.0000], "7-28":[ 2.0700,4.4400], "7-48":[ 2.0000,6.0000], "7-73":[ 2.0300,5.1000], "7-124":[ 2.3300,1.3900], 
            "8-21":[ 2.1500,4.7000], "8-23":[ 2.4000,2.2800], "8-24":[ 2.2200,2.1000], "8-28":[ 2.1300,7.7800], "8-48":[ 1.6870,22.4220], 
            "8-73":[ 2.2200,1.5000], "8-124":[ 2.3000,3.5300], "11-21":[ 2.0500,0.4500], "11-23":[ 2.0500,0.3000], "11-24":[ 2.0500,0.9000], 
            "11-28":[ 2.0500,0.6000], "11-73":[ 2.0500,0.6000], "12-21":[ 2.8000,1.1000], "12-23":[ 3.1400,0.4400], "12-24":[ 2.4300,0.9000], 
            "12-28":[ 2.7700,1.8200], "12-73":[ 2.4300,0.6000], "12-124":[ 3.0900,0.6700], "13-21":[ 2.5800,1.2000], "13-23":[ 2.5800,0.3000], 
            "13-24":[ 2.5800,0.9000], "13-28":[ 2.5800,0.6000], "13-73":[ 2.5800,0.6000], "14-21":[ 2.7200,0.9000], "14-23":[ 2.7200,0.6000], 
            "14-24":[ 2.7200,0.6000], "14-28":[ 2.7200,0.3000], "14-73":[ 2.7200,0.3000], "15-21":[ 2.6800,1.3500], "15-23":[ 2.5100,0.3000], 
            "15-24":[ 2.5100,0.6000], "15-28":[ 2.5100,0.3000], "15-44":[ 2.9400,0.7000], "15-73":[ 2.5100,0.3000], "17-21":[ 2.5500,0.6000], 
            "17-23":[ 2.5500,0.1500], "17-24":[ 2.5500,0.3000], "17-28":[ 2.5500,0.3000], "17-73":[ 2.5500,0.3000], "21-22":[ 2.4000,1.0600], 
            "21-37":[ 1.7400,3.7200], "21-47":[ 2.0700,5.0000], "21-50":[ 2.6500,0.5500], "21-79":[ 1.9500,3.4600], "22-23":[ 2.6900,0.4270], 
            "22-28":[ 2.3700,1.7560], "22-124":[ 2.6300,0.6580], "23-37":[ 2.2500,0.6750], "23-47":[ 2.3100,2.4300], "23-50":[ 3.0000,0.1000], 
            "23-79":[ 2.1900,1.5900], "23-150":[ 2.4000,2.2800], "24-37":[ 2.2200,2.1000], "24-75":[ 2.1400,1.4500], "24-77":[ 1.8300,4.9500], 
            "28-37":[ 2.2200,1.5000], "28-47":[ 2.0500,8.2800], "28-50":[ 3.0000,0.7500], "28-79":[ 1.9600,5.2400], "37-73":[ 2.2200,1.5000], 
            "44-50":[ 2.8300,0.2000], "47-48":[ 1.7900,32.1000], "47-124":[ 2.2100,3.7600], "50-73":[ 3.0000,0.7500], "79-124":[ 2.1100,2.2800]
            }
        

        #some converting factors
        angstrom_to_bohr=1.8897162
        bohr_to_angstrom=0.52918
        radian_to_degree=180.0/np.pi
        degree_to_radian=np.pi/180.0
        hartree_to_kcalmol=627.509391
        kcalmol_to_hartree=0.00159360164


            
        # read gaussian input file; it is preferred to create objects reading both gaussian input and output files since connectivity is not included in output files
        def read_keywords_in_gaussian_input_file(self):                
            with open(self.input_file) as f:
                atom_number=1
                modred=""
                genecp=""
                for l in f:
                    if "chk" in l: self.chk=l
                    if "%mem=" in l: self.mem=l
                    if "%nprocshared=" in l: self.nprocshared=l

                    if l.startswith("#"):
                            route=""
                            coordinates=[]
                            while l!="\n":
                                route=route+l.strip("\n")
                                if route[-1]==")":route+=" "
                                l=next(f)
                            self.route_sections.append(route)
                            l=next(f)
                            self.titles_sections.append(l)
                            l=next(f);l=next(f)
                            self.chg_and_mult_sections.append(l)
                            l=next(f)

                            # read the coordinates:
                            while l!="\n":
                                atom_type=""
                                atom_symbol=""
                                b= l.split()
                                j=1
                                shift=0
                                if b[1] in ["0","-1","-2","-3"]: shift=1
                                if "-" in b[0]: 
                                    atom_symbol=b[0].split("-")[0]
                                    if len( b[0].split("-")) >0:
                                        if  b[0].split("-")[1].isdigit(): 
                                            atom_type=b[0].split("-")[1]
                                            atom_type_warning="atom type read from gaussian input file"
                                else :
                                    atom_symbol=b[0]
                                    atom_type="0"
                                    atom_type_warning="undefined atom type"
                                if len(b)>4+shift: atom_layer=b[4+shift]
                                else: atom_layer="H"
                                # only read coordinates and create atom objects for the first structure found in the file 
                                if len(self.titles_sections)<2:
                                    new_atom=self.Atom(atom_symbol,np.array([float(b[1+shift]),float(b[2+shift]),float(b[3+shift])]),atom_number,atom_type,atom_type_warning,atom_layer   )  
                                    if len(b)>6+shift: new_atom.interface= int(b[6+shift])
                                    if atom_symbol!="": self.atom_list.append(new_atom)
                                    atom_number=atom_number+1
                                # XYZ coordinates will also be stored on a different list, so it can be changed in two molecule specifications (for example, for qst2 or qst3 methods)
                                coordinates.append("%16.11f %16.11f %16.11f"%(float(b[1+shift]),float(b[2+shift]),float(b[3+shift])))  
                                l=next(f,"\n")

                            # read the connectivity
                            if "geom=connectivity" in route.lower():
                                conn=""
                                connectivity_list=[]
                                l=next(f, "\n")
                                while l!="\n":             
                                    conn=conn+l
                                    b=l.split()
                                    j=1
                                    while j+1<len(b):
                                        connectivity_list.append([int(b[0]),int(b[j]),float(b[j+1])])
                                        j=j+2
                                    l=next(f,"\n")
                                #literal, original gaussian connectivity format... it is simpler for some tasks copying it than rebuilding it from the connections of each atom
                                self.connectivity_lines_sections.append(conn) 
                                # assign connections to each atom, including the number, the connection order and the symbol of each connected atom
                                if len(self.titles_sections)<2: # only do this if Link1 has not been found
                                    for atom in self.atom_list:
                                        atom.connection=[]
                                        for c in connectivity_list:
                                            if atom.atom_number==c[0]: connection_number=c[1]
                                            if atom.atom_number==c[1]: connection_number=c[0]
                                            if atom.atom_number==c[1] or atom.atom_number==c[0]:
                                                connection_order=c[2]
                                                for a in self.atom_list:
                                                    if a.atom_number==connection_number:
                                                        connection_symbol=a.symbol
                                                atom.connection.append([connection_number,connection_order,connection_symbol])
                                self.groups_of_atoms_by_connection()

                            if "modredundant" in route.lower()  or "addgic" in route.lower() :
                                modred=""
                                l=next(f,"\n")
                                while l!="\n":
                                    if l.strip(" ")!="": modred=modred+l                                        
                                    l=next(f,"\n")                                   
                                
                            if "gen " in route.lower() or "genecp" in route.lower():
                                genecp=""
                                if l.strip()=="":l=next(f)
                                while l.strip(" ")!="\n":
                                    if l.strip(" ")!="": genecp=genecp+l
                                    l=next(f,"\n")
                                if "genecp" in route.lower():
                                    genecp=genecp+"\n"
                                    l=next(f,"\n")
                                    while l.strip(" ")!="\n":
                                        if l.strip(" ")!="": genecp=genecp+l
                                        l=next(f,"\n") 

                            self.modred_sections.append(modred)
                            self.genecp_sections.append(genecp)      

        # read gaussian output file trying to deduce keywords; note that the first geometry present is used (the geometry can be updated with update_coords_from_output)
        def read_keywords_in_gaussian_output_file(self):
            with open(self.output_file) as f:
                f.readline();f.readline()
                line=f.readline()
                modred=""
                for l in f:
                    if "Link1:  Proceeding to internal job step number" in l:#do not read the input of freq calculations after an optimization
                        while l.find("Normal termination of Gaussian 16") ==-1: 
                            l=next(f,"eof")
                            if l=="eof":break
                    if "1\\1\\GINC" in l: break #do not keep on reading if you reach the archive entry
                    if "chk=" in l: self.chk=l
                    if l.strip().startswith("%mem="): self.mem=l.strip()+"\n"
                    if l.strip().startswith("%nprocshared="): self.nprocshared=l.strip()+"\n"
                    #read keywords
                    if l.startswith(" #"):
                            route=""
                            coordinates=[]
                            while "----" not in l:
                                #route=route+l.strip("\n")
                                route=route+l.strip()
                                l=next(f)                                     
                            self.route_sections.append(route)
                            #only in output files:
                            l=next(f)
                            while "----" not in l:
                                l=next(f)
                            self.titles_sections.append(next(f))
                            next(f)
                            next(f)
                            l=next(f)
                            c_and_m=""
                            while l.startswith(" Charge"):
                                c_and_m+=l.split()[2]+" "+l.split()[5]+"  "
                                l=next(f)
                            self.chg_and_mult_sections.append(c_and_m+"\n")

                            # read the coordinates:
                            atom_number=1
                            while l.strip()!="":
                                atom_type=""
                                atom_type_warning="undefined atom type"
                                atom_symbol=""
                                b= l.split()
                                j=1
                                shift=0
                                if b[1] in ["0","-1","-2","-3","1","2","3"]: shift=1
                                if "-" in b[0]: 
                                    atom_symbol=b[0].split("-")[0]
                                    if len( b[0].split("-")) >0:
                                        if  b[0].split("-")[1].isdigit(): 
                                            atom_type=b[0].split("-")[1]
                                            atom_type_warning="atom type read from gaussian input file"
                                else :
                                    atom_symbol=b[0]
                                    atom_type="0"
                                    atom_type_warning="undefined atom type"
                                if len(b)>4+shift: atom_layer=b[4+shift]
                                else: atom_layer="H"
                                # only read coordinates and create atom objects for the first structure found in the file
                                # the coordinates could be updated with the update_coords_from_G16output method 
                                if len(self.titles_sections)<2:
                                    new_atom=self.Atom(atom_symbol,np.array([float(b[1+shift]),float(b[2+shift]),float(b[3+shift])]),atom_number,atom_type,atom_type_warning,atom_layer   )  
                                    if len(b)>6+shift: new_atom.interface= int(b[6+shift])
                                    if atom_symbol!="": self.atom_list.append(new_atom)
                                    atom_number+=1
                                # XYZ coordinates will also be stored on a different list, so it can be changed in two molecule specifications (for example, for qst2 or qst3 methods)
                                coordinates.append("%16.11f %16.11f %16.11f"%(float(b[1+shift]),float(b[2+shift]),float(b[3+shift]))) 
                                l=next(f,"\n")

                            #connectivity is not read for output files. It will be calculated:
                            self.generate_connections_from_distance()

                            # read modredundant; addgic can not be read from the ouptput file
                            if "modredundant" in route.lower():
                                modred=""
                                while "The following ModRedundant input section has been read" in l: l=next(f,"\n")
                                l=next(f,"\n")
                                while l.strip()!="":
                                    if l.strip(" ")!="": modred=modred+l                                        
                                    l=next(f,"\n")                                   
                            self.modred_sections.append(modred)

        # read orca output file trying to deduce keywords; note that the first geometry present is used (the geometry can be updated with update_coords_from_output)
        # there is no need to read the input file since ORCA includes the whole file
        def read_keywords_in_orca_output_file(self):
            with open(self.output_file) as f:
                atom_number=1
                for l in f:
                    if ("*xyz" in l or "* xyz" in l) and "> " in l and len(self.atom_list)==0: #will only create atom objects if the atom list is empty 
                        if "xyzfile" not in l: 
                            #if the coordinates are present:
                            #(note that in files with serveral jobs initially atoms will be created with the coordinates of the first job)
                            self.chg_and_mult_sections.append(l.split("xyz")[1])
                            l=next(f)
                            while "end" not in l and "*" not in l:
                                if l.strip()!="":
                                    w=l.split(">")[1].split()
                                    atom_symbol=w[0]
                                    atom_coordinates=np.array([float(w[1]),float(w[2]),float(w[3])])
                                    self.atom_list.append(self.Atom(atom_symbol,atom_coordinates,atom_number  ) ) 
                                    atom_number+=1
                                l=next(f)

                            if len(l)>10 and "****END OF INPUT****" not in l:
                                w=l.split(">")[1].split()
                                atom_symbol=w[0]
                                if l.endswith("*"):atom_coordinates=np.array([float(w[1]),float(w[2]),float(w[3]).replace("*","")])
                                elif l.endswith("end"):atom_coordinates=np.array([float(w[1]),float(w[2]),float(w[3]).replace("end","")])
                                elif l.endswith("endd"):atom_coordinates=np.array([float(w[1]),float(w[2]),float(w[3]).replace("endd","")])
                                self.atom_list.append(self.Atom(atom_symbol,atom_coordinates,atom_number  ) )
                                atom_number+=1
                                l=next(f)

                        else: #if the coordinates were read from an xyz file, use the first cartesian coordinates in the file:
                            for ll in f:
                                if "CARTESIAN COORDINATES (ANGSTROEM)" in ll:
                                    next(f);ll=next(f)
                                    while ll.strip()!="":
                                        w=ll.split()
                                        atom_symbol=w[0]
                                        atom_coordinates=np.array([float(w[1]),float(w[2]),float(w[3])])
                                        self.atom_list.append(self.Atom(atom_symbol,atom_coordinates,atom_number  ) ) 
                                        atom_number+=1
                                        ll=next(f)  
                                    break                                      
                    if ">" in l and "%pal" in l and "nprocs" in l: 
                        self.nprocshared="%nprocshared="+l.split("nprocs")[1]+"\n"
                        #if self.mem!="": self.mem="%mem="+str((int(self.nprocshared.split("=")[1])*int(self.mem.split("%mem=")[1].split("Mb")[0]) )/1024)+"Mb\n" #¿quitar esto?
                    if "%MaxCore" in l: 
                        if self.nprocshared!="": 
                            self.mem="%mem="+str(round(int( self.nprocshared.split("=")[1]) * int(l.split("%MaxCore")[1].split()[0]) /1.024))+"Mb\n"
                        else:
                            self.mem="%mem="+str(int(self.nprocshared.split("="))/1024)+"Mb\n"
                        
                    #DELETE? USE THE CHARGES IN G16_output object instead?
                    #read charges; last appearance of the Mulliken charges will be used for the atoms charges
                    if "MULLIKEN ATOMIC CHARGES" in l:
                        l=next(f)
                        if "-----------------------" in l:
                            i=0
                            l=next(f)
                            while "Sum of atomic charges:" in l :
                                i+=1
                                self.atom(i).mulliken_charge= float(l.split()[3])
                                self.atom(i).charge=self.atom(i).mulliken_charge
                                l=next(f)
                    if "General Settings:" in l:
                        while "Nuclear Repulsion" not in l:  
                            if "Total Charge" in l : chg=l.split("....")[1].strip()
                            if "Multiplicity" in l : multpl=l.split("....")[1].strip()
                            l=next(f)  
                        if self.chg_and_mult_sections==[]:  self.chg_and_mult_sections=[chg+"  "+multpl+"\n"]                   

                    #l=next(f)
                self.generate_connections_from_distance()                    

        # copy the keywords from another molecule
        def copy_keywords(self,other):
                self.mem=other.mem
                self.nprocshared=other.nprocshared
                self.head=other.head
                self.tail=other.tail
                self.geom_sections=copy.deepcopy(other.geom_sections)
                self.lotHL_sections=copy.deepcopy(other.lotHL_sections)
                self.lotLL_sections=copy.deepcopy(other.lotLL_sections)
                self.lot_sections=copy.deepcopy(other.lot_sections)
                self.freq_sections=copy.deepcopy(other.freq_sections)
                self.genecp_sections=copy.deepcopy(other.genecp_sections)
                self.opt_sections=copy.deepcopy(other.opt_sections)
                self.extra_sections=copy.deepcopy(other.extra_sections)
                self.route_sections=copy.deepcopy(other.route_sections)
                self.titles_sections=copy.deepcopy(other.titles_sections)
                self.modred_sections=copy.deepcopy(other.modred_sections)

        #create a deep copy of current object
        def copy(self,fast=True):
            new_molecule=Molecular_structure()  #does this work? without arguments? or I should have include this option in the __init__?
            if not fast:
                new_molecule.chk= self.chk
                new_molecule.mem= self.mem
                new_molecule.nprocshared= self.nprocshared
                new_molecule.head= self.head
                new_molecule.tail= self.tail
                new_molecule.geom_sections= self.geom_sections
                new_molecule.lotHL_sections=self.lotHL_sections
                new_molecule.lotLL_sections=self.lotLL_sections
                new_molecule.lot_sections=self.lot_sections
                new_molecule.freq_sections=self.freq_sections
                new_molecule.opt_sections=self.opt_sections
                new_molecule.extra_sections=self.extra_sections
                new_molecule.route_sections=self.route_sections
                new_molecule.titles_sections=self.titles_sections
                new_molecule.chg_and_mult_sections=self.chg_and_mult_sections 



            return new_molecule



        #the constructor of the class, reading from a gaussian input file and/or gaussian or ORCA output file.
        #*file arguments: name of the input file (with extension), and/or name of the output file (with extension) and
        # "first","last","first-optimized","last-optimized","best-rms","lower-energy","higher-energy"  "step="number for selecting with geometry to use from the output file.
        def __init__(self,*arg):
 
            #initializing class variables:
            self.atom_list=[] # a list of the atoms object: [atom with atom_number=1,atom with atom_number=2...] note that atom_list[n-1] is equivalent to self.atom(n)
            #keywords and things for/from the input files:
            self.atom_layers=[]
            self.connectivity_lines_sections=[]
            self.head=""
            self.tail=""
            self.modred_sections=[]
            self.genecp_sections=[]  
            self.chg_and_mult_sections=[]
            self.chk=""
            self.mem=""
            self.nprocshared=""
            self.geom_sections=[]
            self.lotHL_sections=[]
            self.lotLL_sections=[]
            self.lot_sections=[]
            self.freq_sections=[]
            self.opt_sections=[]
            self.extra_sections=[]
            self.route_sections=[]
            self.titles_sections=[]
            self.coordinates=[]
            self.dipole_moment=np.array([np.nan,np.nan,np.nan])
            self.polarizability=0.0
            self.groups=[]
            #a list of the redundant internal coordinates definitions
            self.int_coord_definitions=[] 
            #for those cases in which the molecule is renumbered taking another molecule as reference:
            #a dictionary containing old atom numbers as keys and new atom numbers as values
            self.old_to_new_numbers={}
            #a dictionary containing atom numbers in reference_molecule as keys and new atom numbers as values
            self.ref_to_new_numbers={}
            #list of atoms that are forced to be connected; if connectivity is re-calculated, these atoms will be connected regarded of their distance
            self.forced_connections=[]

            #some things from the outputs:
            self.energy=np.nan
            self.electronic_energy=np.nan   #IMPORTANT: complete this in the update geometry
            self.gibbs_free_energy=np.nan

            #a dictionary of properties requested by the user (see class Property)
            self.properties={}

            #name of files
            self.input_file="" # the name of the input file used for creating the molecular structure object
            self.output_file="" # the name of the output file used for creating the molecular structure object
            self.NEB_xyz_file="" # the file of a NEB trajectory xyz file
            self.fchk_file=""
            self.orca_hess_file=""
            self.orca_internal_hess_file=""

            self.MM3_pi_systems=[]  #list of lists containing atoms in the same pi-system; required for MM3 calculations


            #a list of the arguments that are strings
            str_args=[f for f in arg if type(f)==str] #only strings in the 
            #mol_arg is another structure_molecule object passed as argument
            mol_arg=None
            #gr_arg is a list passed as arguemnt
            gr_arg=None
            properties=[]
            for a in arg:
                if type(a)==type(self): 
                    mol_arg=a
                if type(a)==list:
                    if type(a[0])==int: gr_arg=a
                    if isinstance(a[0],Property): 
                        for prop in a: 
                            self.properties[prop.name]=[]  #setting the keys of the self.properties dictionary
                            properties.append(prop)        #for passing to QM_outputfile constructor

            #find out which are the input and output files, based on their extensions
            for f in str_args:
                if any(f.endswith(ext) for ext in [".com",".gjf",".inp"]): self.input_file=f
                if any(f.endswith(ext) for ext in [".out",".log"]): self.output_file=f 
                if f.endswith("_MEP_trj.xyz"): self.NEB_xyz_file=f
                elif f.endswith("_MEP_trj"):  self.NEB_xyz_file=f+".xyz"              
            #inpfile=open(self.input_file,"r")
            #inplines=inpfile.readlines()
            #blanklinecount=0
            #atom_number=-2

            #create a new Molecular_structure object from another one and a list of groups
            #inherits the keywords from the old Molecular_structure object
            # replace this functionality with copy.deepcopy()? 
            if mol_arg!=None:
                self.chk=mol_arg.chk
                self.mem=mol_arg.mem
                self.nprocshared=mol_arg.nprocshared
                self.head=mol_arg.head
                self.tail=mol_arg.tail
                self.geom_sections=copy.deepcopy(mol_arg.geom_sections)
                self.lotHL_sections=copy.deepcopy(mol_arg.lotHL_sections)
                self.lotLL_sections=copy.deepcopy(mol_arg.lotLL_sections)
                self.lot_sections=copy.deepcopy(mol_arg.lot_sections)
                self.freq_sections=copy.deepcopy(mol_arg.freq_sections)
                self.genecp_sections=copy.deepcopy(mol_arg.genecp_sections)
                self.opt_sections=copy.deepcopy(mol_arg.opt_sections)
                self.extra_sections=copy.deepcopy(mol_arg.extra_sections)
                self.route_sections=copy.deepcopy(mol_arg.route_sections)
                self.titles_sections=copy.deepcopy(mol_arg.titles_sections)
                self.chg_and_mult_sections=copy.deepcopy(mol_arg.chg_and_mult_sections)
                self.input_file=mol_arg.input_file
                self.int_coord_definitions=copy.deepcopy(mol_arg.int_coord_definitions)
                self.int_hess=copy.deepcopy(mol_arg.int_hess)
                self.MM3_pi_systems=copy.deepcopy(mol_arg.MM3_pi_systems)
                self.modred_sections=copy.deepcopy(mol_arg.modred_sections)
                self.normal_modes=copy.deepcopy(mol_arg.normal_modes)
                self.old_to_new_numbers=copy.deepcopy(mol_arg.old_to_new_numbers)
                self.opt_sections=copy.deepcopy(mol_arg.opt_sections)
                self.orca_hess_file=mol_arg.orca_hess_file
                self.orca_internal_hess_file=mol_arg.orca_internal_hess_file
                self.output_file=mol_arg.output_file
                
                if gr_arg!=None:
                    for gr in gr_arg:
                        for atom_number in mol_arg.groups[gr-1]:
                            #self.atom_list.append(copy.copy( mol_arg.atom_list[atom_number-1] ) )
                            self.atom_list.append(mol_arg.atom_list[atom_number-1].copy())
                reord_dict={self.atom_list[i].atom_number:i+1 for i in range(0,len(self.atom_list))}
                for gr in self.groups: #update numbers in the groups
                    for g in gr: g=reord_dict[g]
                for atom in self.atom_list: # update numbers in atoms
                    atom.atom_number=reord_dict[atom.atom_number]
                    if atom.interface!=0: atom.interface=reord_dict[atom.interface]                    
                    for c in atom.connection:
                        c[0]=reord_dict[c[0]]
                self.connectivity_lines_sections.append(self.__print_connectivity())
                #self.groups_of_atoms_by_connection()
                self.groups=[copy.deepcopy(gr) for gr in mol_arg.groups]
                self.forced_connections=[copy.deepcopy(fc) for fc in mol_arg.forced_connections]


            #if there is a gaussian input file
            if len(str_args)>0 and self.input_file!="": self.read_keywords_in_gaussian_input_file()
           
            #if there is an output file but not an input file:
            if len(str_args)>0 and self.input_file=="" and self.output_file!="":
                with open(self.output_file) as f:
                    firstlines=f.readlines()[0:10]
                    if any( ["Entering Gaussian System," in l for l in firstlines]): self.read_keywords_in_gaussian_output_file()
                    elif any( ["* O   R   C   A *" in l  for l in firstlines]): 
                        self.read_keywords_in_orca_output_file() 
            

            #if there is an output file and also an input file
            if len(str_args)>0 and self.output_file!="":
                with open(self.output_file) as f:
                    firstlines=f.readlines()[0:10]
                    if any( ["Entering Gaussian System," in l for l in firstlines]):
                        self.QM_output=G16_output(self.output_file,properties)
                    elif any( ["* O   R   C   A *" in l for l in firstlines]):
                        self.QM_output=ORCA_output(self.output_file,properties)

            #check if connectivity exists; if not, create it by the distances:
            self.check_connectivity()

            #if there is a NEB_xyz_file:
            if len(str_args)>0 and self.NEB_xyz_file!="":
                self.MEP_xyz=self.MEP_xyz(self.NEB_xyz_file,self.output_file)

            #update coordinates from a gaussian or orca ouput file
            for f in str_args:
                if f in ["first","last","first-optimized","last-optimized","best-rms","lower-energy","higher-energy"] or f.startswith("step="):
                    self.update_coords_from_output(f)
            
            self.translate_gaussian_route_sections()

            #if there is a fchk file in the directory with the same name, set fchk_file to it
            if os.path.isfile(self.output_file[:-4]+".fchk"): self.fchk_file=self.output_file[:-4]+".fchk"
            if os.path.isfile(self.output_file[:-4]+".fchk"): self.fchk_file=self.output_file[:-4]+".fchk"

            #if there is a .hess file in the same directory, set orca_hess_file to it+
            if os.path.isfile(self.output_file[:-4]+".hess"): self.orca_hess_file=self.output_file[:-4]+".hess"
            if os.path.isfile(self.output_file[:-4]+"_internal.hess"): self.orca_internal_hess_file=self.output_file[:-4]+"_internal.hess"

            #calculate molecular weight
            self.molecular_weight=np.sum( [self.atomic_weights[a.symbol.lower()] for a in self.atom_list] )

        #adds the atoms of other_molecule to current molecule, updating their numbers
        #keywords, output file names, etc of current molecule will be kept (although they are of no use)
        #the redundant internal coordinates list will also be updated
        def join(self,other_molecule):

            #append the coordinates of the other molecule to the cart_coordinates in QM_output and ORCA_output objects (if they exists)
            if other_molecule.QM_output.cart_coordinates!=None and self.QM_output.cart_coordinates!=None: self.QM_output.cart_coordinates+=other_molecule.QM_output.cart_coordinates

            #update the atoms
            initial_number_of_atoms=len(self.atom_list)
            #add the atoms:
            for a in other_molecule.atom_list: self.atom_list.append(a)
            #update the new atoms numbers and connections
            for a in self.atom_list[initial_number_of_atoms:]:
                a.atom_number=a.atom_number+initial_number_of_atoms 
                for c in a.connection:
                    c[0]+=initial_number_of_atoms
            
            #update the redundant internal coordinates definitions
            initial_number_of_int_coord_definitions=len(self.int_coord_definitions)
            #add the internal coordinates definitions
            for i in other_molecule.int_coord_definitions: self.int_coord_definitions.append(i)
            #update the numbers
            for i in self.int_coord_definitions[ initial_number_of_int_coord_definitions:]:
                for n in i: n+=initial_number_of_int_coord_definitions

            #update ref_to_new dictionary (the old_to_new_numbers will be outdated)
            for k in other_molecule.ref_to_new_numbers.keys():
                self.ref_to_new_numbers[k]=other_molecule.ref_to_new_numbers[k]+initial_number_of_atoms

            #update the connectivity section
            self.connectivity_lines_sections[0]=( self.__print_connectivity() )
            
  
        #remove the atoms which numbers are in list of atoms; list_of_atoms can also be a single atom -a number- or all atoms corresponding to an element (eg. "H"), except those in list "keep"
        def remove_atoms(self,list_of_atoms_to_remove,keep=[]):

            old_number_of_atoms=len(self.atom_list)

            if type(list_of_atoms_to_remove)==int: list_of_atoms_to_remove=[list_of_atoms_to_remove]
            elif type(list_of_atoms_to_remove)==str:
                list_of_atoms_to_remove=[a.atom_number for a in self.atom_list if a.symbol.lower()==list_of_atoms_to_remove.lower()]
            if keep!=[]:
                if type(keep)==int: keep=[keep]
                for k in keep: list_of_atoms_to_remove.remove(k)
  

            #delete atoms:
            self.atom_list[:]=[a for a in self.atom_list if a.atom_number not in list_of_atoms_to_remove]

            for a in self.atom_list:
                a.connection[:]=[c for c in a.connection if c[0] not in list_of_atoms_to_remove]

            #a dictionary relating new and old atom numbers
            old_to_new={}
            i=1
            for a in self.atom_list:
                old_to_new[a.atom_number]=i
                i+=1

            #modify atom numbers and connectivity according to old_to_new
            for a in self.atom_list:
                a.atom_number=old_to_new[a.atom_number]
                for c in a.connection: c[0]=old_to_new[c[0]]

            #remove the internal coordinate definitions if they were present (the change of atom names make it meaningless)
            self.int_coord_definitions=[]
            self.connectivity_lines_sections[0]=( self.__print_connectivity() )

            #work on atom properties: in bond order lists, remove elements corresponding to atoms removed
            for a in self.atom_list:
                for k in a.properties.keys():                    
                    if type(a.properties[k])==list and len(a.properties[k])==old_number_of_atoms:
                        a.properties[k]=[ a.properties[k][l] for l in range(0,len(a.properties[k])) if l+1 not in list_of_atoms_to_remove]
                    elif type(a.properties[k])==np.ndarray and len(a.properties[k])==old_number_of_atoms:
                        a.properties[k]=np.array([ a.properties[k][l] for l in range(0,len(a.properties[k])) if l+1 not in list_of_atoms_to_remove])


                    

        #translate the keywords in the route sections
        def translate_gaussian_route_sections(self,route_sections=[]):

            # the method below splits the gaussian route file using characters. 
            # This method adds "@" to the options in the route section between two " characters. 
            # This allow that splitting the gaussian route files (using spaces as delimiters) does not break the "external" sections.
            # later, @ characters will be replaced again to spaces.
            # the alternative is a regexp nightmare
            def __block(s): 
                aux=s.split('"') 
                for i in range(1,len(aux),2):
                    aux[i]=aux[i].replace("\n"," ").replace(" ","@") 
                return '"'.join(aux)

            if route_sections==[]:route_sections=self.route_sections
            # extract information from the route sections:
            for route in route_sections:
                keywords=__block(route).lower().split()
                geom=""
                lot=""                
                lotHL=""
                lotLL=""
                freq=""
                opts=[""]
                extra=[""]
                for k in keywords:
                    if "geom=" in k.lower():
                        geom=k
                        continue                      
                    elif "oniom" in k.lower()  and "oniompcm" in k.lower(): 
                        temp_lot=k    #####change this for not-ONIOM calculations!!!!!!!
                        temp_lotHL=temp_lot[5:].strip("=")[1:][:-1].split(":")[0]
                        temp_lotLL=temp_lot[5:].strip("=")[1:][:-1].split(":")[1]
                        lotHL=temp_lotHL.replace("@"," ")  #"Unblock" (see __block method above)
                        lotLL=temp_lotLL.replace("@"," ")
                        lot=temp_lot.replace("@"," ")
                        continue
                    elif "freq" in k.lower():
                        freq=k
                        continue      
                    elif "opt" in k.lower():
                        opts=k.replace("opt=(","").replace(")","").split(",")
                        continue
                    elif "#" not in k: 
                        extra.append(k)
                self.geom_sections.append(geom)
                self.lotHL_sections.append(lotHL)
                self.lotLL_sections.append(lotLL)
                self.lot_sections.append(lot)
                self.freq_sections.append(freq)
                self.opt_sections.append(opts)
                self.extra_sections.append(extra)


        #return an Atom object given its atom_number
        def atom(self,n_atom):
            # less efficient
            #for a in self.atom_list:
            #    if a.atom_number==n_atom: return a
            return self.atom_list[n_atom-1]
        
        #METHODS FOR PRINTING IN DIFFERENT FORMATS

        # returns a string with the connectivity in gaussian input format based on the information of each atom
        def __print_connectivity(self):
            s=""
            for atom in self.atom_list:
                s += str(atom.atom_number)
                for c in atom.connection:
                    if c[0]>atom.atom_number: s+= "  "+str(c[0])+" "+str(c[1])
                s+= "\n"
            return s

        #returns a string with a gaussian input file; if a name of the file is given, it will be written in the file. 
        def print_new_file(self,filename="",header=""):
            i=0
            s=""

            if filename!="":
                if filename[:-4]!=".com" or filename[:-4]!=".gjf" or filename[:-4]!=".inp": filename=filename+".com"
            elif filename=="":
                if self.input_file!="" and len(self.input_file.split(".")>1) and self.input_file[:-4] in [".com",".gjf",".inp"]:
                    if self.input_file.split(".")[-2].isdigit(): n=str(int(self.input_file.split(".")[-2])+1)
                    else: n="1"                        
                    filename=".".join(self.input_file.split(".")[:-2])+"."+n+"."+self.input_file.split(".")[-1]

                elif self.input_file!="" and len(self.input_file.split(".")>1) and self.input_file[:-4] not in [".com",".gjf",".inp"]:
                    if self.input_file.split(".")[-1].isdigit(): n=str(int(self.input_file.split(".")[-1])+1)
                    else: n="1"                        
                    filename=".".join(self.input_file.split(".")[:-1])+"."+n+"."

                elif self.input_file!="" and len(self.input_file.split(".")==1):
                    filename=self.input_file+".1"

            if self.chk=="":
                if self.input_file[:-4] in [".com",".gjf",".inp"]: chk_file_name=".".join(filename.split(".")[:-1])+".chk"
                else: chk_file_name=filename+".chk" 
            else: chk_file_name=self.chk

            while i<len(self.route_sections):
                coordinates=""
                for A in self.atom_list:
                    #using the coord read and processed in the input file (could be required if the coord will be modified, for example, for building model system with the correct H atom coord)
                    if A.atom_type=="0" or A.atom_type=="": atom_type_string=""
                    else: atom_type_string="-"+str(A.atom_type)
                    if A.charge!=0.0: atom_type_string+="-"+str(A.charge)
                    new_coord_line="%- 10s  %2s     %- 15.8f %- 15.8f %- 15.8f %2s" %(A.symbol+atom_type_string," 0",float(A.coord[0]),float(A.coord[1]),float(A.coord[2]),A.atom_layer )
                    if A.interface!=0:   #place interface atoms, only H atom supported
                        # TO DO: place the atom at the right (calculated) distance
                        atom_bound_type=self.get_H_MM3type(A.interface)
                        if A.interface==0:str_interface=""
                        else: str_interface=str(A.interface)
                        new_coord_line=new_coord_line+" H-"+str(atom_bound_type)+" "+str_interface+"\n"
                    else: new_coord_line=new_coord_line+"\n"
                    coordinates+=new_coord_line

                if i!=0: s=s+"--Link1--\n"
                s=s+self.chk+self.mem+self.nprocshared+self.route_sections[i]+"\n"
                if len(self.connectivity_lines_sections)<=i:conn=self.connectivity_lines_sections[0]
                else: conn=self.connectivity_lines_sections[i]
                s=s+"\n"+self.titles_sections[i]+"\n"+self.chg_and_mult_sections[i]+coordinates+"\n"+conn+"\n"
                
                if "modred" in self.route_sections[i] or "addgic" in self.route_sections[i]: s=s+self.modred_sections[i]+"\n"
                if "gen " in self.route_sections[i] or "genecp " in self.route_sections[i]:  s=s+self.genecp_sections[i]+"\n"
        
                #WARNING FOR TESTING: CHECK IF THE NUMBER OF SPACES IS CORRECT IN ALL CIRCUMSTANCES
                i=i+1
            if filename!="":
                if filename[:-4]!=".com" or filename[:-4]!=".gjf" or filename[:-4]!=".inp": filename=filename+".com"
                with open(filename,"w") as f: f.write(s)   
            return s

        #returns a string with the format of an amber rdcrd file; if a name of the file is given, it will be written in the file. 
        def print_mdcrd(self,filename=""):
            s=""
            coords=[]
            for A in self.atom_list:
                coords.append("%6.3f"%(float(A.coord[0])))
                coords.append("%6.3f"%(float(A.coord[1])))
                coords.append("%6.3f"%(float(A.coord[2])))
            for i in range(0,len(coords)):
                s+="  "+coords[i]
                if (i+1)%10==0:s+="\n"
            if filename!="":
                if filename[-4:]!=".crd" or filename[:-5]!=".mcrd": filename=filename+".crd"
                with open(filename,"w") as f: f.write(s)            
            return s

        #returns a string with the format of an xyz file; if a name of the file is given, it will be written in the file.
        def print_xyz(self,filename=""):
            format_line="{:2s}       {:9.5f}      {:9.5f}      {:9.5f}"
            s=str(len(self.atom_list))+"\n"
            if self.input_file=="" and self.output_file!="":
                s+=self.output_file+"\n"
            else: s+=self.input_file+"\n"
            for A in self.atom_list:
                s+=format_line.format(A.symbol,A.coord[0],A.coord[1],A.coord[2])+"\n"
            if filename!="":
                if filename[-4:]!=".xyz": filename=filename+".xyz"
                with open(filename,"w") as f: f.write(s)
            return s

        #returns a string with the format of an pdb file; if a name of the file is given, it will be written in the file.
        def print_pdb(self,filename=""):
            format_line="{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}                      {:>2s}{:2s}"
            format_connect="{:6s}{:5d}{:5d}{:5d}"
            s="REMARK file created from:"
            if self.input_file=="" and self.output_file!="": s+=self.output_file+"\n"                
            else: s+=self.input_file+"\n"
            i=0
            for A in self.atom_list:
                i+=1
                s+=format_line.format("HETATM",i,A.symbol,"","","",0,"",A.coord[0],A.coord[1],A.coord[2],A.symbol,"")+"\n"
            s+="END\n"
            for A in self.atom_list:
                s+="CONNECT"+"{:5d}".format(A.atom_number)
                for c in A.connection:
                    s+="{:5d}".format(c[0])
                s+="\n"
            if filename!="":
                if filename[-4:]!=".pdb": filename=filename+".pdb"
                with open(filename,"w") as f: f.write(s)
            return s

        #returns a string with the format of an ORCA file. only the number of processors and the memory are used. If a name of the orca file is given, it will be written in the file.
        def print_orca(self,filename="",header="",tail=""):
            format_line="{:2s}       {:9.5f}      {:9.5f}      {:9.5f}"
            s=header
            if "%pal" not in header.lower():
                if self.nprocshared.strip()!="": s+="%pal nprocs "+ self.nprocshared.split("=")[1].strip() +" \n end \n"; nproc= int(self.nprocshared.split("=")[1])
                else: s+="%pal nprocs 7\n end\n"; nproc=7
            if "%maxcore" not in header.lower():     
                if "Gb" in self.mem: mem=round (float(self.mem.split("=")[1].split("Gb")[0])*1024 )
                elif "GB" in self.mem: mem=round (float(self.mem.split("=")[1].split("GB")[0])*1024 )
                elif "gb" in self.mem: mem=round (float(self.mem.split("=")[1].split("gb")[0])*1024 )
                elif "Mb" in self.mem: mem=round (float(self.mem.split("=")[1].split("Mb")[0]) ) 
                elif "MB" in self.mem: mem=round (float(self.mem.split("=")[1].split("MB")[0]) )
                elif "mb" in self.mem: mem=round (float(self.mem.split("=")[1].split("mb")[0]) )
                elif "MW" in self.mem: mem=round (float(self.mem.split("=")[1].split("MW")[0])*8 )
                elif "Mw" in self.mem: mem=round (float(self.mem.split("=")[1].split("Mw")[0])*8 )
                elif "mw"in self.mem: mem=round (float(self.mem.split("=")[1].split("mw")[0])*8 ) 
                else: mem=30000           
                s+="\n%MaxCore " + str(int(round(mem/nproc)))+" \n"

                if len(self.chg_and_mult_sections[0].split() )==2: s+="\n * xyz "+self.chg_and_mult_sections[0]
                else: s+="\n * xyz "+self.chg_and_mult_sections[0].split()[0]+"  "+self.chg_and_mult_sections[0].split()[1]
                # try to translate any other keyword?... maybe in a future
            for A in self.atom_list:
                s+=format_line.format(A.symbol,A.coord[0],A.coord[1],A.coord[2])+"\n"
            s+="end\n"
            if tail!="": s+=tail
            if filename!="":
                if filename[-4:] not in [".inp",".com"]: filename+=".inp"
                with open(filename,"w") as f: f.write(s)
            return s

        #returns a string with a gaussian input file including only the atoms in the HL layer, attached to the atoms in the interface (not scaled!). if a name of the file is given, it will be written in the file.
        def print_model_file(self,filename=""):

            coordinates=""
            new_conn=""
            i=1
            old_numbers=[]  #keep a track of former atom_numbers, before omitting the atoms in the LL layer
            
            for A in self.atom_list:
                new_coord_line=""
                if A.atom_layer=="H":
                    new_coord_line="%- 10s  %2s     %- 15.8f %- 15.8f %- 15.8f %2s" %(A.symbol+"-"+A.atom_type," 0",float(A.coord[0]),float(A.coord[1]),float(A.coord[2]),"H" )+"\n"
                    old_numbers.append(i)
                elif A.interface!=0:
                        atom_bound_type=self.get_H_MM3type(A.interface)
                        new_coord_line="%- 10s  %2s     %- 15.8f %- 15.8f %- 15.8f %2s" %(A.symbol+"-"+str(atom_bound_type)," 0",float(A.coord[0]),float(A.coord[1]),float(A.coord[2]),"H" )+"\n"
                        old_numbers.append(i)
                coordinates+=new_coord_line               
                i=i+1
            #correct the connectivity changing the old atom numbers with the new ones.
            for l in self.connectivity_lines_sections[0].split("\n"):
                wc=l.split()
                if len(wc)>0 and int(wc[0]) in old_numbers:
                    new_conn=new_conn+str(old_numbers.index(int(wc[0])) +1 )+" "
                    for d in range(1,len(wc)-1,2):
                        if int(wc[d]) in old_numbers:
                            new_conn=new_conn+str(old_numbers.index(int(wc[d]) )+1 )+" "
                            new_conn=new_conn+str(wc[d+1])+" "
                        new_conn=new_conn
                    new_conn=new_conn+"\n"
                
            # for the file for the model system, only basic header will be printed: no modredundant, genecp, etc; no composed jobs either. 
            # only geom section is included (for indicating connectivity)
            new_route="# geom=connectivity"
            title="ONLY MODEL SYSTEM: atoms in the HL layer and H atoms in the interface (not scaled!)"
            s=self.chk+self.mem+self.nprocshared+new_route+"\n"
            s=s+"\n"+title+"\n\n"+self.chg_and_mult_sections[0]+coordinates+"\n"+new_conn+"\n"
            if filename!="":
                if filename[:-4]!=".com" or filename[:-4]!=".gjf" or filename[:-4]!=".inp": filename=filename+".com"
                with open(filename,"w") as f: f.write(s) 
            return s

        #print a fake gaussian output file to open it with gaussview (just like OfakeG output)
        #currently only for visualizing the MEP trajectory after a NEB calculation #TO DO: prepare it also for optimization, TS searches and hessian calculations.
        def print_fake_gau_out(self,filename=""):
            s=" 0 basis functions\n 0 alpha electrons\n 0 beta electrons\nGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad"
            t0="\nGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
            t1="                         Standard orientation:\n"
            t1+=" ---------------------------------------------------------------------\n"
            t1+=" Center     Atomic      Atomic             Coordinates (Angstroms)\n"
            t1+=" Number     Number       Type             X           Y           Z\n"
            t2=" ---------------------------------------------------------------------\n"
            t3="         Item               Value     Threshold  Converged?\n"

            s+=t0
            if self.MEP_xyz!="":
                counter=0
                for coord,e in zip(self.MEP_xyz.cart_coordinates,self.MEP_xyz.energies):
                    counter+=1
                    s+=t1
                    s+=t2
                    i=1
                    for a,c in zip(self.MEP_xyz.atom_symbol_list,coord):
                        s+="{:6}".format(i)+"{:11}".format(self.atomic_numbers[a.lower()])+"{:12}".format(0)+"{:16.6f}".format(c[0])+"{:12.6f}".format(c[1])+"{:12.6f}".format(c[2])+"\n"
                        i+=1
                    s+=t2+"\n SCF Done:"+"{:20.9f}".format(e)+"\n\n"
                    s+=t0
                    s+=" Step number"+"{:4}".format(counter)+"\n"
                    s+=t3
                    s+=" Maximum Force            0.029897     0.000300     NO\n"
                    s+=" RMS     Force            0.004866     0.000100     NO\n"
                    s+=" Maximum Displacement     0.091184     0.004000     NO\n"
                    s+=" RMS     Displacement     0.015596     0.002000     NO\n"
                    s+=t0
                s+="\n Normal termination of Gaussian\n"

            if filename!="":
                if filename.endswith(".fake.out")==False and filename.endswith(".out")==True: filename=filename[:-4]+".fake.out"
                elif filename.endswith(".out")==False: filename=filename+".fake.out"
                with open(filename,"w") as f: f.write(s) 
            return s        


        # not yet ready...
        """
        def print_mol_file(self,filename=""):
            number_of_atoms=str(len(self.atom_list))

            #bond section
            nb=0
            sb=""
            already_bound=[]
            for atom in self.atom_list:
                for c in atom.connection:
                    if [atom.atom_number,c[0]] not in already_bound and [c[0],atom.atom_number] not in already_bound:
                        nb+=1
                        sb += str(nb)+"     "+ str(atom.atom_number)+"     "+str(c[0])+"     "+str(c[1])+"\n"
                        already_bound.append([atom.atom_number,c[0]])


            #atom section
            n=0
            sa=""
            for atom in self.atom_list:
                sa="%- 10s  %2s     %- 15.8f %- 15.8f %- 15.8f %2s" %(A.symbol+atom_type_string," 0",float(A.coord[0]),float(A.coord[1]),float(A.coord[2]),A.atom_layer )


            #molecule section
            s="@<TRIPOS>MOLECULE\n\n"+number_of_atoms+" "+str(n)+" 0 0 0\nSMALL\nNO_CHARGES\n****\nNo comments\n\n"
            
            if filename!="":
                if filename.endswith(".mol")==False: filename=filename+".mol" 
                with open(filename,"w") as f: f.write(s) 
            return s 
        """


        # METHODS TO CALCULATE GEOMETRICAL FEATURES OF THE MOLECULE.
        # calculate dihedral angle of 4 atoms (if len(atoms)==4) or the angle between 3 atoms (if len(atoms)==3)
        def angle(self,atoms):

            if len(atoms)==4:
                if type(atoms)==list and type(atoms[0])==int:
                    q1=self.atom(atoms[1]).coord-self.atom(atoms[0]).coord
                    q2=self.atom(atoms[2]).coord-self.atom(atoms[1]).coord
                    q3=self.atom(atoms[3]).coord-self.atom(atoms[2]).coord
                if type(atoms)==type(self.atom_list) and type(atoms[0])==type(self.atom_list[0]):
                    q1=atoms[1].coord-self.atom(atoms[0]).coord
                    q2=atoms[2].coord-self.atom(atoms[1]).coord
                    q3=atoms[3].coord-self.atom(atoms[2]).coord
                n11=np.cross(q1,q2)
                n1=n11/(np.sum(n11**2)**0.5)
                n22=np.cross(q2,q3)
                n2=n22/(np.sum(n22**2)**0.5)
                u1=n2.copy()
                u3=q2/(np.sum(q2**2)**0.5)
                u2=np.cross(u1,u3)
                return (180/math.pi)*math.atan2(np.dot(n1,u2),np.dot(n1,n2))

            if len(atoms)==3:
                if type(atoms)==list and type(atoms[0])==int:                
                    q1=self.atom(atoms[1]).coord-self.atom(atoms[0]).coord
                    q2=self.atom(atoms[1]).coord-self.atom(atoms[2]).coord
                if type(atoms)==type(self.atom_list) and type(atoms[0])==type(self.atom_list[0]):
                    q1=atoms[1].coord-self.atom(atoms[0]).coord
                    q2=atoms[1].coord-self.atom(atoms[2]).coord                                
                c=np.dot(q1,q2)/((np.sum(q1**2)*np.sum(q2**2)))**0.5
                s=np.sum( np.cross(q1,q2)**2)**0.5 /((np.sum(q1**2)*np.sum(q2**2)))**0.5
                return (180/math.pi)*math.atan2(s,c)

        #calculate the distance between two atoms. They might be atoms from current molecule and another molecule
        def distance(self,atoms,other_molecule=None):
            if other_molecule==None:
                if type(atoms)==list and type(atoms[0])==int:
                    return np.sum((self.atom(atoms[1]).coord-self.atom(atoms[0]).coord)**2)**0.5
                #both "atoms" elements are atom objects (they can be from different molecules)
                if type(atoms)==type(self.atom_list) and type(atoms[0])==type(self.atom_list[0]):
                    #return np.sum((atoms[1].coord-self.atom(atoms[0]).coord)**2)**0.5 
                    return np.sum((atoms[1].coord-atoms[0].coord)**2)**0.5
            if other_molecule!=None:           
                if type(atoms)==list and type(atoms[0])==int:
                    return np.sum((other_molecule.atom(atoms[1]).coord-self.atom(atoms[0]).coord)**2)**0.5

        #return the nearest atom to one atom given.
        def nearest_atom(self,atom,exclude_H=False):
            if type(atom)==int: atom=self.atom(atom)
            dist=999.0
            nearest_atom_number=0
            for i,a  in enumerate (self.atom_list):
                if self.distance([atom,a])<dist and self.distance([atom,a])>0.1:
                    if exclude_H==True and a.symbol.lower()=="h": continue
                    dist=self.distance([atom,a])
                    nearest_atom_number=i
            return self.atom_list[nearest_atom_number]

        #returns the ratio between the distance between two atoms and the sum of the covalent radius
        def distance_rcov_ratio(self,atoms,other_molecule=None):
            if other_molecule==None:
                if type(atoms)==list and type(atoms[0])==int:
                    s_r_cov = self.r_cov[ self.atom(atoms[1]).symbol.lower()  ]  +  self.r_cov[ self.atom(atoms[0]).symbol.lower()  ]
                    distance=np.sum((self.atom(atoms[1]).coord-self.atom(atoms[0]).coord)**2)**0.5
                #both "atoms" elements are atom objects (they can be from different molecules)
                if type(atoms)==type(self.atom_list) and type(atoms[0])==type(self.atom_list[0]):
                    s_r_cov =self.r_cov[ atoms[1].symbol.lower()  ] + self.r_cov[ atoms[0].symbol.lower()  ]
                    distance=np.sum((atoms[1].coord-atoms[0].coord)**2)**0.5
                    #return np.sum((atoms[1].coord-self.atom(atoms[0]).coord)**2)**0.5                 
            if other_molecule!=None:           
                if type(atoms)==list and type(atoms[0])==int:
                    distance=np.sum((other_molecule.atom(atoms[1]).coord-self.atom(atoms[0]).coord)**2)**0.5
                    s_r_cov=self.r_cov[ other_molecule.atom(atoms[1]).symbol.lower()  ]  +  self.r_cov[ self.atom(atoms[0]).symbol.lower()  ]
            return distance/s_r_cov

        #returns distance matrix of current molecule; diagonal element is set to 0
        def distance_matrix(self,relative=False):
            distance_matrix=np.zeros((len(self.atom_list),len(self.atom_list)))
            for i in range(0,len(self.atom_list)):
                for j in range(0,len(self.atom_list)):
                    if relative:
                        distance_matrix[i,j]=self.distance_rcov_ratio([i+1,j+1])
                    else:
                        distance_matrix[i,j]=self.distance([i+1,j+1])
            return distance_matrix

        #returns distance matrix of current molecule; note that diagonal element is set to 0        
        def inv_distance_matrix(self):
            inv_distance_matrix=np.zeros((len(self.atom_list),len(self.atom_list)))
            for i in range(0,len(self.atom_list)):
                for j in range(0,len(self.atom_list)):
                    if i!=j: inv_distance_matrix[i,j]=1/self.distance([i+1,j+1])
            return inv_distance_matrix                


        #returns the distance between two atoms minus the sum of covalent radius
        #useful to find the atoms that binds two fragments
        def excess_distance(self,atoms):

            if type(atoms)==list and type(atoms[0])==int:
                s_r_cov = self.r_cov[ self.atom(atoms[1]).symbol.lower()  ]  +  self.r_cov[ self.atom(atoms[0]).symbol.lower()  ]
                return np.sum((self.atom(atoms[1]).coord-self.atom(atoms[0]).coord)**2)**0.5-s_r_cov
            if type(atoms)==type(self.atom_list) and type(atoms[0])==type(self.atom_list[0]):
                s_r_cov =self.r_cov[ atoms[1].symbol.lower()  ] + self.r_cov[ atoms[0].symbol.lower()  ]
                return np.sum((atoms[1].coord-self.atom(atoms[0]).coord)**2)**0.5-s_r_cov  


        def linear_bend_angle(self,atoms):
            
            if type(atoms)==list and type(atoms[0])==int:
                p1=self.atom(atoms[0]).coord
                p2=self.atom(atoms[1]).coord
                p3=self.atom(atoms[2]).coord
                p4=self.atom(atoms[3]).coord
            if type(atoms)==type(self.atom_list) and type(atoms[0])==type(self.atom_list[0]):
                p1=atoms[0].coord
                p2=atoms[1].coord
                p3=atoms[2].coord
                p4=atoms[3].coord                
            #define 3 ortogonal vectors, the first connects atoms 1 and 3
            u1=p3-p1
            u1=u1/(np.sum(u1**2)**0.5)  #normalized
            #the second is perpendicular to the first vector and goes toward point 4
            proj=u1.dot(p4-p2)*u1
            u2=p4 - p2 - proj
            u2=u2/(np.sum(u2**2)**0.5) #normalized
            #the fourth is ortogonal to u1 and u2:
            u3=np.cross(u1,u2)
            u3=u3/(np.sum(u3**2)**0.5)

            #project each of the p1,p2,and p3 vectors over u1 u2 and u3 
            q1=p1-p2
            q2=p3-p2
            q1_u1=u1.dot(q1)
            q1_u2=u2.dot(q1)
            q1_u3=u3.dot(q1)
            q2_u1=u1.dot(q2)
            q2_u2=u2.dot(q2)
            q2_u3=u3.dot(q2)

            q1_u1xu2=np.array([q1_u1,q1_u2])
            q1_u1xu3=np.array([q1_u1,q1_u3])
            q2_u1xu2=np.array([q2_u1,q2_u2])
            q2_u1xu3=np.array([q2_u1,q2_u3])
            c1=np.dot(q1_u1xu2,q2_u1xu2)/((np.sum(q1_u1xu2**2)*np.sum(q2_u1xu2**2)))**0.5
            s1=np.sum( np.cross(q1_u1xu2,q2_u1xu2)**2)**0.5 /((np.sum(q1_u1xu2**2)*np.sum(q2_u1xu2**2)))**0.5
            c2=np.dot(q1_u1xu3,q2_u1xu3)/((np.sum(q1_u1xu3**2)*np.sum(q2_u1xu3**2)))**0.5
            s2=np.sum( np.cross(q1_u1xu3,q2_u1xu3)**2)**0.5 /((np.sum(q1_u1xu3**2)*np.sum(q2_u1xu3**2)))**0.5

            return 360-(180/math.pi)*math.atan2(s1,c1),360-(180/math.pi)*math.atan2(s2,c2)

            """
            #copied from: SANG-HO LEE, KIM PALMO, SAMUEL KRIMM Journal of Computational Chemistry, Vol. 20, No. 10, 1067􏰏1084 (1999)
            if type(atoms)==list and type(atoms[0])==int:
                q21=self.atom(atoms[1]).coord-self.atom(atoms[0]).coord
                q23=self.atom(atoms[1]).coord-self.atom(atoms[2]).coord
                q24=self.atom(atoms[1]).coord-self.atom(atoms[3]).coord
            e21=q21/(np.sum(q21**2)**0.5)
            e23=q23/(np.sum(q21**2)**0.5)
            e24=q24/(np.sum(q21**2)**0.5)
            qu=np.cross(e23,e24)
            u=qu/(np.sum(qu**2)**0.5)
            w=np.cross(u,e23)
            gmm1=(180/math.pi)*math.asin(u.dot(np.cross(e21,e23)))
            gmm2=(180/math.pi)*math.asin(u.dot(e21))
            print (gmm1)
            print (gmm2)
            """

        # return a list with the redundant internal coordinates values; if int_coord_definitions is not given, self.int_coord_definitions will be used
        # if bohrs or angstroms are specified, the distances and angles will be changed 
        def get_internal_coordinates_values(self,int_coord_definitions=[],*arg):
            str_args=[f for f in arg if type(f)==str] #only strings in the arguments
            internal_coordinates_values=[]
            if len(int_coord_definitions)==0:
                int_coord_definitions=self.int_coord_definitions
            for ic in int_coord_definitions:
                if len(ic)==2:
                    value=self.distance(ic)
                    if "bohr" in str_args: value=value*self.angstrom_to_bohr
                elif len(ic)>2 and len(ic)<5:
                    value=self.angle(ic)
                    if "radians" in str_args: value=value*self.degree_to_radian
                elif len(ic)==5:
                    value1,value2=self.linear_bend_angle(ic[:-1])
                    if ic[-1]==-1: value=value1
                    elif ic[-1]==-2: value=value2
                    if "radians" in str_args: value=value*self.degree_to_radian
                internal_coordinates_values.append(value)
            return internal_coordinates_values

        def get_internal_coordinates_differences(self,reference_molecule,int_coord_definitions=[],atom_number_correspondence={}):

            internal_coordinate_differences=[]
            internal_coordinate_definitions=[]
            if type(int_coord_definitions)==list and len(int_coord_definitions)==0:
                int_coord_definitions=self.int_coord_definitions
            if type(int_coord_definitions)==str and int_coord_definitions in ["dihds","dihedrals"]:
                internal_coordinate_definitions=[ic for ic in self.int_coord_definitions if len(ic)==4]
            if atom_number_correspondence=={}:
                atom_number_correspondence= dict( [(value,key) for key,value in self.ref_to_new_numbers.items()] )


            for ic in internal_coordinate_definitions:

                if len(ic)<5:                    
                    new_ic=[atom_number_correspondence[i] for i in ic]
                    if len(ic)==2:
                        value1=self.distance(ic)
                        value2=reference_molecule.distance(new_ic)
                    if len(ic)>2 and len(ic)<5:
                        value1=self.angle(ic)
                        value2=reference_molecule.angle(new_ic)

                    if value1<0: value1=360+value1
                    if value2<0: value2=360+value2
                    diff = abs(value2-value1)
                    if diff>180:diff=360-diff
                    #for debugging:
                    #if diff>120:
                    #    print ("for "+str(ic)+" in current molecule: "+str(value1)+" corresponding to: "+str(new_ic)+" in reference molecule: "+str(value2)+"; difference:"+str(diff))
    
                    internal_coordinate_differences.append(diff)
                if len(ic)==5:
                    new_ic=[atom_number_correspondence[i] for i in ic[0:4]]
                    new_ic.append(ic[5])
                    value1a,value1b=self.linear_bend_angle(ic[:-1])
                    value2a,value2b=reference_molecule.linear_bend_angle(new_ic[:-1])
                    internal_coordinate_differences+=[abs(value1a-value2a),abs(value1a+value2a)]

            return internal_coordinate_differences

        #returns a 1D np.array with the differences in the coordinates of current atoms and reference_molecule
        #the 1D format is useful to multiply with the hessian matrix; it could be reshaped to a 3xN format using: v.reshape(int(len(v)/3,3), where v is the 1D np.array
        #is this needed?
        def get_cart_coordinates_differences(self,reference_molecule,atom_number_correspondence={}):

            cart_coordinate_differences=np.empty(0)
            if atom_number_correspondence=={}:
                atom_number_correspondence= dict( [(value,key) for key,value in self.ref_to_new_numbers.items()] )
            
            for a in self.atom_list:
                new_atom_number=atom_number_correspondence[a.atom_number]
                new_atom=reference_molecule.atom(new_atom_number)
                coord_diff=new_atom.coord-a.coord
                cart_coordinate_differences=np.concatenate((cart_coordinate_differences,coord_diff))
            
            return cart_coordinate_differences


        #METHODS FOR READING FROM GAUSSIAN FCHK FILES
        #read a simmetric matrix as formated in gaussian fchk files
        def __read_sim_matrix_from_fchk_lines(self,lines):
            n=int(lines[0].split("N=")[1])
            dim=int(((8*n+1)**0.5-1)/2)
            matrix=np.zeros((dim,dim))
            temp_list=[]
            for line in lines[1:]: temp_list+=line.split()
            k=0
            for i in range(0,dim+1):
                for j in range (0,i):
                    matrix[i-1,j]=float(temp_list[k])
                    matrix[j,i-1]=float(temp_list[k])
                    k+=1
            return matrix

        #read the list of internal coordinates as formatted in gaussian fchk files
        def __read_int_coord_definitions_from_fchk_lines(self,lines):
            temp_list=[]
            for line in lines[1:]: temp_list+=line.split()
            int_coord_definitions=[]
            for j in range(0,len(temp_list),4):
                internal_coordinate=[int(temp_list[j])]
                for k in range(1,4):
                    if (temp_list[j+k]!="0") and ("-" not in temp_list[j+k]):
                        internal_coordinate+= [int(temp_list[j+k])]
                    if "-" in temp_list[j+k]:
                        w=temp_list[j+k].strip("-")
                        internal_coordinate+=[int(w[:-1]),int(w[-1])] 
                int_coord_definitions.append(internal_coordinate)
            return int_coord_definitions

        #read XYZ vectors as formatted in gaussian fchk files (used in cartesian coordinates, gradients,etc); returns a list of coordinates in np.arrays
        def __read_XYZ_from_fchk_lines(self,lines):
            temp_list=[]
            for line in lines[1:]: temp_list+=line.split()
            XYZ=[]
            for j in range(0,len(temp_list),3):
                XYZ.append( np.array([ float(temp_list[j]),float(temp_list[j+1]),float(temp_list[j+2]) ]) )
            return XYZ

        # isolates blocks of information in gaussian fchk files
        def __extract_block_from_fchk(self,flag,lines=[]):
            block=[]
            if lines==[]:
                with open(self.fchk_file,"r") as f: lines=f.readlines()
            for i in range(0,len(lines)):
                if lines[i].startswith(flag):
                    block.append(lines[i])
                    i+=1
                    while str.isdigit(lines[i][11]): 
                        block.append(lines[i])
                        i+=1
            return block

        # read the list of redundant internal coordinates from a fchk file and sets self.int_coord_definitions to it
        def __read_int_coord_definitions_from_fchk_file(self,fchk_file=""):
            if fchk_file=="": fchk_file=self.fchk_file
            with open(fchk_file,"r") as f: lines=f.readlines()
            block= self.__extract_block_from_fchk("Redundant internal coordinate indices",lines)
            self.int_coord_definitions=self.__read_int_coord_definitions_from_fchk_lines(block)

        # read the cartesian coordinates from a fchk file and sets self.cart_coordinates and the atoms .coord to it
        def __read_coordinates_from_fchk_file(self,fchk_file=""):
            if fchk_file=="": fchk_file=self.fchk_file
            with open(fchk_file,"r") as f: lines=f.readlines()
            block= self.__extract_block_from_fchk("Current cartesian coordinates",lines)
            self.cart_coordinates=self.__read_XYZ_from_fchk_lines(block)
            for atom,c in zip(self.atom_list,self.cart_coordinates): 
                atom.coord=c

        # read the cartesian hessian from a fchk file and sets self.cart_hess to it
        def __read_cart_hess_from_fchk_file(self,fchk_file=""):
            if fchk_file=="": fchk_file=self.fchk_file
            with open(fchk_file,"r") as f: lines=f.readlines()
            block= self.__extract_block_from_fchk("Cartesian Force Constants",lines)
            self.cart_hess= self.__read_sim_matrix_from_fchk_lines(block)

        # read the internal hessian from a fchk file and sets self.int_hess to it
        # the fchk file must have been converted with the -3 flat:  fchk -3 name_of_chk.chk 
        def __read_int_hess_from_fchk_file(self,fchk_file=""):
            if fchk_file=="": fchk_file=self.fchk_file
            with open(fchk_file,"r") as f: lines=f.readlines()
            block= self.__extract_block_from_fchk("Internal Force Constants",lines)
            self.int_hess= self.__read_sim_matrix_from_fchk_lines(block)

        # read the normal modes from a fchk file
        def __read_vib_normal_modes_from_fchk_file(self,fchk_file=""):

            if fchk_file=="": fchk_file=self.fchk_file
            with open(fchk_file,"r") as f: lines=f.readlines()
            number_of_atoms=0
            for i in range(0,40):
                if "Number of atoms" in lines[i]: number_of_atoms=int(lines[i].split("I")[1])         
            block= self.__extract_block_from_fchk("Vib-Modes",lines)
            temp_list=[]
            for line in block[1:]: temp_list+=line.split()
            normal_modes=[]
            normal_mode=[]
            for i in range(0,len(temp_list),3):
                normal_mode.append([float(temp_list[i]),float(temp_list[i+1]),float(temp_list[i+2])])
                if len(normal_mode)==number_of_atoms:
                    normal_modes.append(normal_mode)
                    normal_mode=[]
            return normal_modes

        #METHODS FOR READING FROM ORCA .HESS FILES
        # read a hessian from a orca .hess file (valid for internal or cartesian)
        def __read_hess_from_orca_hess_file(self,orca_hess_file=""):
            if orca_hess_file=="": orca_hess_file=self.orca_hess_file
            with open(orca_hess_file) as f: lines=f.readlines()
           
            for i in range(0,len(lines)):
                if lines[i].startswith("$hessian"):
                    i+=1
                    dim=int(lines[i])
                    hess=np.zeros((dim,dim))
                    i+=1
                    while lines[i].strip()!="":
                        if lines[i][0:8]=="        " and len(lines[i].split())>0:
                            j_indexes=[int(x) for x in lines[i].split()]
                        if str.isdigit(lines[i][4]) or str.isdigit(lines[i][5]) :
                            i_index=int(lines[i][0:6])
                            #i_index=int(lines[i].split()[0])
                            values=[float(x) for x in lines[i].split()[1:]]
                            for j_index,v in zip(j_indexes,values):
                                #print ("i_index:"+str(i_index)+"   j_index:"+str(j_index)) #borrame
                                hess[i_index,j_index]=v
                        i+=1
            return hess

        #read HESS from either fchk or .hess file (the one that is present)
        def read_hess(self,type="cart"):
            if self.fchk_file!="":
                if type in ["cartesian","cart","xyz"]: self.__read_cart_hess_from_fchk_file()
                elif type in ["internals", "redundant", "int", "red"]: self.__read_int_hess_from_fchk_file()
            elif self.orca_hess_file!="":
                if type in ["cartesian","cart","xyz"]: self.cart_hess= self.__read_hess_from_orca_hess_file()
                elif type in ["internals", "redundant", "int", "red"]: self.int_hess= self.__read_hess_from_orca_hess_file()

        #read vibrational normal modes from either out or fchk files
        def read_normal_modes (self):
            #if there is a fchk file, read it from it because it has more precission than the gaussian output file:
            if self.fchk_file!="":  self.normal_modes=self.__read_vib_normal_modes_from_fchk_file()
            elif self.QM_output.outfile!="": 
                self.QM_output.read_frequencies()
                self.normal_modes=self.QM_output.normal_coordinates


        # NOT NEEDED, ALREADY READ IN ORCA_output:
        def read_int_coord_definitions_from_orca_opt_file(self,orca_opt_file=""):
            if orca_opt_file=="": orca_opt_file=self.orca_opt_file
            with open(orca_opt_file) as f: lines=f.readlines() 
            coords=[]
            for i in range(0,len(lines)):  
                if "$redundant_internals" in lines[i]:
                    i+=1
                    w=lines[i].split()
                    i+=1
                    for l in lines[i:i+int(w[0])]:
                        ww=l.split()
                        coords.append([int(ww[0])+1,int(ww[1])+1])
                    for l in lines[i+int(w[0]):i+int(w[0])+int(w[1])]:
                        ww=l.split()
                        coords.append([int(ww[0])+1,int(ww[1])+1,int(ww[2])+1])
                    for l in lines[i+int(w[0])+int(w[1]):i+int(w[0])+int(w[1])+int(w[2])]:
                        ww=l.split()
                        coords.append([int(ww[0])+1,int(ww[1])+1,int(ww[2])+1,int(ww[3])+1])
                    for l in lines[i+int(w[0])+int(w[1])+int(w[2]):i+int(w[0])+int(w[1])+int(w[2])+int(w[3])]:
                        ww=l.split()
                        coords.append([int(ww[0])+1,int(ww[1])+1,int(ww[2])+1,int(ww[3])+1,int(ww[4])+1])
            return coords

      
        # METHODS RELATED TO CONNECTIVITY AND GROUPS OF ATOMS CONNECTED.




        # use a distance criteria to calculate connectivity; modifies the connections of the atoms
        # with keep_old_orders=True the bond order of the connection is kept from old values (if it is possible)
        def generate_connections_from_distance(self,keep_old_orders=True,H_only_bound_to_C=False,allow_fract_orders=True):

            #a dictionary of list containing each pair of connected atoms, with the order as keys
            #it is used if keep_old_orders=True
            temp_connections={0.5:[],1:[],1.5:[],2:[],3:[]}
            #fill temp_connections and delete current connections from atoms
            for a in self.atom_list:
                for c in a.connection:
                    temp_connections[ c[1] ].append([a.atom_number,c[0]])
                a.connection=[]

            i=1
            for a1 in self.atom_list:
                for a2 in self.atom_list[i:]:

                    b_order=0
                    dist=self.distance([a1.atom_number,a2.atom_number])
                    s_r_cov =self.r_cov[ a1.symbol.lower()  ] + self.r_cov[ a2.symbol.lower()  ]

                    #connections between any pair of atoms (excluding H)
                    if a2.symbol.lower()!="h" and a1.symbol.lower()!="h":
                        if dist<s_r_cov*0.8: b_order=3
                        elif dist<s_r_cov*0.9: b_order=2
                        elif dist<s_r_cov*0.95:
                            if allow_fract_orders: b_order=1.5
                            else: b_order=1
                        elif dist<s_r_cov*1.1: b_order=1
                        elif dist<s_r_cov*1.35: 
                            if allow_fract_orders: b_order=0.5
                            else: b_order=0

                    #connections between H and C atoms
                    elif (a2.symbol.lower()=="h" and a1.symbol.lower()=="c" ) or (a1.symbol.lower()=="h" and a2.symbol.lower()=="c"):
                        if dist<s_r_cov*1.1: 
                            b_order=1
                        elif dist<s_r_cov*1.35: 
                            if allow_fract_orders: b_order=0.5
                            else: b_order=1

                    #connections between H and atoms different than C
                    elif (a2.symbol.lower()=="h" and a1.symbol.lower()!="c") or (a1.symbol.lower()=="h" and a2.symbol.lower()!="c"):
                        if not H_only_bound_to_C:   #since H-atoms on heteroatoms are labile, sometimes it is better to have them disconnected setting H_only_bound_to_C=True...
                            if dist<s_r_cov*1.1:
                                b_order=1
                            elif dist<s_r_cov*1.35:
                                if allow_fract_orders: b_order=0.5  
                                else: b_order=0 
                        else: b_order=0 
                
                    #asign the bond orders as they were previously
                    if b_order!=0:
                        if keep_old_orders:
                            for order in [0.5,1,1.5,2,3]:
                                if any(p in temp_connections[order] for p in [[a2.atom_number,a1.atom_number],[a1.atom_number,a2.atom_number]]): 
                                    b_order=order

                        a1.connection.append([a2.atom_number,b_order,a2.symbol])
                        a2.connection.append([a1.atom_number,b_order,a1.symbol])               
                i+=1

            #what to do with orphan toms? add a bond to the nearest not heteroatom

            for h in [a for a in self.atom_list if len(a.connection)==0]:
                #if the orphan atom is a H atom and the nearest C atom is closer than 1.6 Angstroms, bound them
                if h.symbol.lower()=="h": 
                    dist=999.0
                    closer_atom=0
                    for a in self.atom_list:
                        if a.symbol.lower()!="h" and a.symbol.lower()=="c":
                            d=self.distance([h.atom_number,a.atom_number])
                            if d<1.6 and d<dist: dist=d; closer_atom=a.atom_number
                    if dist<999.0:
                        a=self.atom(closer_atom) 
                        h.connection.append([a.atom_number,1.0,a.symbol])
                        a.connection.append([h.atom_number,1.0,h.symbol])

            #add the connections in forced_connections that does not depend on distances
            self.add_forced_connections()#borrar

            #update the connectivity lines and the groups based on the new connections
            self.connectivity_lines_sections.append( self.__print_connectivity() )
            self.connectivity_lines_sections[0]= self.__print_connectivity() 
            self.groups_of_atoms_by_connection()

        # add the connections to the atoms in the forced_connections list
        def add_forced_connections(self):
            for p in self.forced_connections:
                c_0,c_1=[c[0] for c in self.atom(p[0]).connection],[c[0] for c in self.atom(p[1]).connection] 
                s0,s1 =self.atom(p[0]).symbol, self.atom(p[1]).symbol
                if p[1] not in c_0: self.atom(p[0]).connection.append([p[1],1,s1])         
                if p[0] not in c_1: self.atom(p[1]).connection.append([p[0],1,s0])
                #repeat again changing the order:
                #c_1,c_0=[c[0] for c in self.atom(p[0]).connection],[c[0] for c in self.atom(p[1]).connection]
                #s1,s0 =self.atom(p[0]).symbol, self.atom(p[1]).symbol
                #if p[1] not in c_0: self.atom(p[0]).connection.append([p[1],1,s1])         
                #if p[0] not in c_1: self.atom(p[1]).connection.append([p[0],1,s0])

        # removes the connections for which the order is 0.5
        def remove_non_covalent_connections(self):
            for a in self.atom_list:
                marked_to_remove=[]
                for c in a.connection:
                    if c[1]==0.5: marked_to_remove.append(c)
                for r in marked_to_remove:
                    a.connection.remove(r)
            #restore the connections in forced_connections (in case any of them was deleted)
            self.add_forced_connections()
            self.groups_of_atoms_by_connection()

        # list of atoms (by atom number) in the same group because they are connected
        def groups_of_atoms_by_connection(self):
            groups=[]
            for atom in self.atom_list:
                if not any([(atom.atom_number in gr) for gr in groups]): 
                    groups.append([atom.atom_number])                
                    for a in groups[-1]:
                        for c in self.atom(a).connection:
                            if c[0] not in groups[-1]:groups[-1].append(c[0])
                    groups[-1].sort()                    
            self.groups=groups

        # connect two atoms -either atom numbers or atom objects in the arguments
        # if they were connected with different order, the order is changed
        def connect_atoms (self,a1,a2,order=1):
            if type(a1)==type(1) and type(a2)==type(1):
                for a in self.atom_list:
                    if a.atom_number==a1:
                        atom1=a
                    if a.atom_number==a2:
                        atom2=a
            elif type(a1)==type(self.atom_list[0]) and type(a2)==type(self.atom_list[0]):
                atom1=a1
                atom2=a2
            conn1=[atom2.atom_number,order,atom2.symbol]
            conn2=[atom1.atom_number,order,atom1.symbol]
            for c in atom1.connection:
                if c[0]==conn1[0]: c[1]=conn1[1]
            for c in atom2.connection:
                if c[0]==conn2[0]: c[1]=conn2[1]
            if conn1 not in atom1.connection: atom1.connection.append(conn1)
            if conn2 not in atom2.connection: atom2.connection.append(conn2)

        # remove the connection between two atoms
        def disconnect_atoms (self,a1,a2):
            marked_to_remove=[]
            if type(a1)==type(1) and type(a2)==type(1):
                for a in self.atom_list:
                    if a.atom_number==a1:
                        atom1=a
                    if a.atom_number==a2:
                        atom2=a
            elif type(a1)==type(self.atom_list[0]) and type(a2)==type(self.atom_list[0]):
                atom1=a1
                atom2=a2
            for c in atom1.connection:
                if c[0]==atom2.atom_number: marked_to_remove=[c[0],c[1],c[2]]
            if len(marked_to_remove)!=0: atom1.connection.remove(marked_to_remove)   #check that the connection actually exists, and remove it            
            for c in atom2.connection:
                if c[0]==atom1.atom_number: marked_to_remove=[c[0],c[1],c[2]]
            if len(marked_to_remove)!=0: atom2.connection.remove(marked_to_remove)   #check that the connection actually exists, and remove it
            #if len(marked_to_remove)==0: print ("trying to disconnect two atoms that were not connected")   # for debuggin

        # connect atoms to their n nearest atoms (atoms is a list of number of atoms, single atom number or a symbol)
        def add_n_connections (self,atoms,n):

            if type(atoms)==type(1): atoms=[self.atom(atoms)]
            elif type(atoms)==type("P"):atoms=[a for a in self.atom_list if a.symbol.lower()==atoms.lower()]
            elif type(atoms)==type(self): atoms=[atoms]
            elif type(atoms)==list and type(atoms[0])==type(1): atoms=[self.atoms(a) for a in atoms]

            for atom in atoms:
                excess_distances=[]
                n_nearest_atoms=[]
                for a in self.atom_list:
                    dist=self.distance([a.atom_number,atom.atom_number])
                    s_r_cov =self.r_cov[ a.symbol.lower()  ] + self.r_cov[ atom.symbol.lower()  ]
                    n_nearest_atoms.append(a.atom_number)
                    excess_distances.append(dist-s_r_cov)
                n_nearest_atoms=[e for _,e in sorted(zip(excess_distances,n_nearest_atoms))]

                #remove all atom connections:
                for c in atom.connection: self.disconnect_atoms(atom,self.atom(c[0]))
                #create new connections    
                for i in range (1,n+1):
                    self.connect_atoms(atom,self.atom(n_nearest_atoms[i]))
                    if ([atom.atom_number,n_nearest_atoms[i]] not in self.forced_connections) and ([n_nearest_atoms[i],atom.atom_number] not in self.forced_connections):
                        self.forced_connections.append([atom.atom_number,n_nearest_atoms[i]])

        #disconnect H atoms not bound to C, since H-heteroatoms can move, it is better for matching molecules to remove these bonds
        #also removes bonds of 0.5 order        
        def H_only_bound_to_C (self):
            for a in self.atom_list:
                if a.symbol.lower()!="h": 
                    if a.symbol.lower()!="c":
                        for c in a.connection[:]:  #the [:] makes a copy of a.connections to iterate over it, while remove delete elements of the original a.connection. Otherwise, some elements are not removed.
                            if c[2].lower()=="h": a.connection.remove(c)
                else:
                    for c in a.connection[:]:
                        if c[2].lower()!="c": a.connection.remove(c)
            for a in self.atom_list:
                for c in a.connection[:]:
                    if c[1]==0.5: a.connection.remove(c)

            self.add_forced_connections()
            #update the connectivity lines and the groups based on the new connections
            self.connectivity_lines_sections[0]= self.__print_connectivity() 
            self.groups_of_atoms_by_connection()

        # reset the connections according to the back_up 
        def reset_connections(self):
            #check first if the back_up_connection exist:
            back_up_connectivity_exists=False
            for a in self.atom_list:
                if a.back_up_connection!=[]: back_up_connectivity_exists=True; break
            #if it exist, use it to substitute current connectivity
            if back_up_connectivity_exists:
                for a in self.atom_list: a.connection=a.back_up_connection
                #add forced connections
                self.add_forced_connections()
                #update the connectivity lines and the groups based on the new connections
                self.connectivity_lines_sections[0]= self.__print_connectivity() 
                self.groups_of_atoms_by_connection()
            #if it does not exist, simply call generate_connections_from_distances
            else: self.generate_connections_from_distance()
            self.groups_of_atoms_by_connection()
                
        #checks if connectivity exists; if not, generate it by the distances
        def check_connectivity(self):
            connectivity_exists=False
            for a in self.atom_list:
                if a.connection!=[]: connectivity_exists=True; break
            if connectivity_exists==False: 
                self.generate_connections_from_distance()
            #make a copy of the connections so it can be reset when needed.
            for a in self.atom_list: a.back_up_connection=a.connection




        # the atom in gr1 that is closer to gr2 is disconnected from group 1 and connected to group 2 (useful for H transferred)
        # the atom that is moved from one group to the other will only be connected to a single atom.
        def transfer_atom_from_groups(self,gr1,gr2,order=1.0):
            #find the nearest atoms
            distance=999.0
            n_atom1=0
            n_atom2=0

            for n_a1 in self.groups[gr1]:
                for n_a2 in self.groups[gr2]:
                    n_distance= self.distance([n_a1,n_a2])
                    if n_distance<distance:
                        distance=n_distance
                        n_atom1=n_a1  #caution! n_atom starts with 1, not 0; 
                        n_atom2=n_a2
            #atom objects with these atom numbers:
            for a in self.atom_list:
                    if a.atom_number==n_atom1: atom1=a
                    if a.atom_number==n_atom2: atom2=a

            # atom1 is the atom that will be passed
            # first, remove the connections of other atoms to atom2:
            for a in self.atom_list:
                for c in a.connection: 
                    if c[0]==atom2.atom_number: a.connection.remove(c)
            # next,change atom2 connection
            atom2.connection=[[atom1.atom_number,order,atom1.symbol]]
            # add to atom1 the connection to atom 2:
            atom1.connection.append([atom2.atom_number,order,atom2.symbol])

        # add the nth nearest H(-C) atom to group; returns the number of atoms that were added (if it was not possible to add any, returns 0)
        def add_nearest_H_atom_to_group(self,group,n=0,connect=True,only_HC=True):

            if type(group)==type(1): group=self.groups[group]
            if type(n)==type(1):n=[n]

            if only_HC:
                #list of possible H atoms to add; only included:                         (i)if it is a H                          (ii)if not in current group                 (iii)if it is bound to a C atom
                possible_H_atoms_to_add=[atom.atom_number for atom in self.atom_list if ((atom.symbol in ["H","h","D","d"])  and (atom.atom_number not in group) and (any([d in [c[1] for c in atom.connection] for d in ["C","c"] ]) ) ) ]
            else: 
                #list of possible H atoms to add; only included:                         (i)if it is a H                          (ii)if not in current group                 
                possible_H_atoms_to_add=[atom.atom_number for atom in self.atom_list if ((atom.symbol in ["H","h","D","d"])  and (atom.atom_number not in group)  ) ]

            #list of min distances of every possible H atoms to add to group
            possible_H_atoms_min_distance_to_group=[]
            #list of the atom in group that is nearest to each possible H atoms to add
            nearest_atom_in_group=[]
            #number of H(-C) that were successfully added
            number_of_H_added=0

            if len(possible_H_atoms_to_add)>0:
                for i in range(0,len(possible_H_atoms_to_add)):
                    min_dist=999.0
                    for a_g in group:
                        dist=self.distance[possible_H_atoms_to_add[i],a_g]
                        if min_dist>dist: 
                            dist=min_dist
                            possible_H_atoms_min_distance_to_group[i]=dist
                            nearest_atom_in_group[i]=a_g
            
                #now sort the lists according to the possible_H_atoms_min_distance_to_group
                possible_H_atoms_to_add= [d for _,d in sorted(zip(possible_H_atoms_min_distance_to_group,possible_H_atoms_to_add))]
                nearest_atom_in_group= [d for _,d in sorted(zip(possible_H_atoms_min_distance_to_group,nearest_atom_in_group))]
                possible_H_atoms_min_distance_to_group=sorted(possible_H_atoms_min_distance_to_group)

                number_of_H_added=0
                for nn in n:
                    if nn<len(nearest_atom_in_group):
                        #print ("**** connecting "+str(possible_H_atoms_to_add[n])+" to "+str(nearest_atom_in_group[n])+"(i="+str(n)+",group="+str(group)+")")#for debuggin
                        if connect:#will only connect the H atom to a C atom
                            if self.atom(nearest_atom_in_group[nn]).symbol.lower()=="c": self.connect_atoms(possible_H_atoms_to_add[nn],nearest_atom_in_group[nn],1)                

                        #remove the H(-C) atom from its current group:
                        for g in self.groups:            
                            if possible_H_atoms_to_add[nn] in g: g.remove(possible_H_atoms_to_add[nn])
                        #remove current group from the list of groups (will be added later)
                        self.groups.remove(group)
                        #add H atom to current group
                        group.append(possible_H_atoms_to_add[nn])
                        group.sort()
                        #add the updated group to self.groups, since it is placed the last last, and it will be easier to recover in other methods. 
                        self.groups.append(group) 
                        number_of_H_added+=1
            
            return number_of_H_added               


        # remove the Hydrogen in list Hs_to_remove from its current group, disconnecting it 
        # and updating the groups list  
        # assumes that the H_atom is only bound to one atom; this will be true if remove_non_covalent_connections() was used before
        def remove_H_atom_from_group(self,Hs_to_remove):

            if type(Hs_to_remove)==type(1):Hs_to_remove=[Hs_to_remove]

            for H_to_remove in Hs_to_remove:
                group=[]
                for gr in self.groups:
                    if self.atom(H_to_remove).atom_number in gr: 
                        group=gr; break
                #disconnect atom connections 
                if len(self.atom(H_to_remove).connection)>0:
                    for c in self.atom(H_to_remove).connection:  self.disconnect_atoms(H_to_remove,c[0])
                
                # this is faster than recalculating the groups using groups_of_atoms_by_connection() 
                # besides, the modified group will be placed last, and it will be easier to recover in other methods.
                # first, it deletes the group from which the H was removed
                self.groups.remove(group)
                # next, a new group was added containing only the H that was removed
                self.groups.append(H_to_remove)
                # the old group is updated so it does no longeer contain the H that is removed
                group.remove(H_to_remove)
                #the updated old group is added to the end of the list of groups
                self.groups.append(group)

        #connect the two nearest atoms in group1 and group2; group1 and 2 could be list of atoms or group numbers
        def combine_groups(self,groups,reconnect=False):

            # convert the passed parameters to a list of groups, where each group is a list of atoms
            if type(groups)==type(1):
                groups=self.groups[groups]
            for i in range(0,len(groups)): 
                if type(groups[i])==type(1):groups[i]=self.groups[groups[i]]

            # if only one group is passed, it will join nothing, but the group will be placed last:
            if len(groups)==1:
                self.groups.remove(groups[0])
                self.groups.append(groups[0])
                return #do nothing else
                
            # the list of all the atoms that are in groups
            combined_group=[g for gr in groups for g in gr]
            combined_group.sort()
            #remove the groups from the list of the molecule groups and add the new group that combines all atoms ordered
            # could also make the new bonds first and then call to groups_of_atoms_by_connection(),
            # but this is faster and places the new combined group the last in the list, which is convenient to recover it in other methods.
            for group in groups:
                self.groups.remove(group)
            self.groups.append(combined_group) 
            # create the new bonds; for each group the nearest atom of any of the other groups will be located and a bond will be made.
            if reconnect:
                for group in self.groups:
                    min_distance=999
                    for g in group:
                        for r in np.setdiff1d(combined_group,group):
                            distance=self.excess_distance([g,r])
                            if distance<min_distance:
                                min_distance=distance
                                atom1=g
                                atom2=int(r)
                    self.connect_atoms(atom1,atom2,0.5)
               

        # METHODS TO REPLACE THE COORDINATES OF THE ATOMS
        # update the coords of the atoms using the information on either G16output or ORCAoutput objects
        def update_coords_from_output(self,f=""):

            nc=99999
            if f=="first": nc=0
            if f=="last" : nc=-1 
            if f=="first-optimized": nc=self.QM_output.stationary_points[0]
            if f=="last-optimized" : nc=self.QM_output.stationary_points[-1]
            if f=="best-rms" : nc=self.QM_output.get_smaller_rms_point()
            if f=="lower-energy" : nc=self.QM_output.get_stationary_point_with_lower_energy()
            if f=="higher-energy" : nc=self.QM_output.get_stationary_point_with_higher_energy()
            if f.startswith("step="): nc=int(f.strip("step=") )-2 #why -2? it only works this way...  #check this!!!!!!!!

            if f=="":
                if len(self.QM_output.stationary_points[0])==1: nc=self.QM_output.stationary_points[0]
                if len(self.QM_output.stationary_points[0])==0 and len(self.QM_output.cart_coordinates)==1: nc=0
                if len(self.QM_output.stationary_points[0])==0 and len(self.QM_output.cart_coordinates)>1: nc=self.QM_output.get_smaller_rms_point()
                if len(self.QM_output.stationary_points[0])>1: nc=self.QM_output.get_stationary_point_with_higher_energy()
            if nc!= 99999 : 
                #print (self.QM_output.ESP_charges[nc]) #borrame
                #print (self.QM_output.hirshfeld_charges[nc]) #borrame
                #print (self.QM_output.mayer_pop_analysis[nc]) # borrame
                #print (self.QM_output.mayer_bond_orders) # borrame
                for a,c  in zip(self.atom_list,self.QM_output.cart_coordinates[nc]): a.coord=np.array(c)
                for a,mc in zip(self.atom_list,self.QM_output.mulliken_charges[nc]): 
                    a.mulliken_charge=mc
                    if np.std(self.QM_output.mulliken_charges[nc])!=0:
                        a.mulliken_charge_standarized=(mc-np.mean(self.QM_output.mulliken_charges[nc]))/np.std(self.QM_output.mulliken_charges[nc])
                for a,lc in zip(self.atom_list,self.QM_output.loewdin_charges[nc]): 
                    a.loewdin_charge=lc
                    if np.std(self.QM_output.loewdin_charges[nc])!=0:
                        a.loewdin_charge_standarized=(lc-np.mean(self.QM_output.loewdin_charges[nc]))/np.std(self.QM_output.loewdin_charges[nc])
                for a,cc in zip(self.atom_list,self.QM_output.chelpg_charges[nc]): 
                    a.chelpg_charge=cc
                    if np.std(self.QM_output.chelpg_charges[nc])!=0:
                        a.chelpg_charge_standarized=(cc-np.mean(self.QM_output.chelpg_charges[nc]))/np.std(self.QM_output.chelpg_charges[nc])
                for a,ec in zip(self.atom_list,self.QM_output.ESP_charges[nc]): 
                    a.esp_charge=ec  #????no ec?
                    if np.std(self.QM_output.ESP_charges[nc])!=0:
                        a.ESP_charge_standarized=(ec-np.mean(self.QM_output.ESP_charges[nc]))/np.std(self.QM_output.ESP_charges[nc])
                for a,hc in zip(self.atom_list,self.QM_output.hirshfeld_charges[nc]): 
                    a.hirshfeld_charge=hc
                    if np.std(self.QM_output.hirshfeld_charges[nc])!=0:
                        a.hirshfeld_charge_standarized=(hc-np.mean(self.QM_output.hirshfeld_charges[nc]))/np.std(self.QM_output.hirshfeld_charges[nc])
                for a,mpa in zip(self.atom_list,self.QM_output.mayer_pop_analysis[nc]):
                    a.mayer_pop_analysis=mpa
                for a in self.atom_list:
                    for mbo in self.QM_output.mayer_bond_orders[nc]:
                        if a.atom_number==mbo[0]: a.mayer_bond_orders.append([mbo[1],mbo[2]])
                        if a.atom_number==mbo[1]: a.mayer_bond_orders.append([mbo[0],mbo[2]])                         
                for a,cs in zip(self.atom_list,self.QM_output.chemical_isotropic_shields[nc] ): a.chemical_isotropic_shield=cs

                self.energies=self.QM_output.energies[nc]
                self.electronic_energy=self.QM_output.energies[nc]
                self.gibbs_free_energy=self.QM_output.gibbs_free_energies[nc]
                self.zero_point_energy=self.QM_output.zero_point_energies[nc]
                #caution! for G16 calculations, this overrides the normal coordinates read from a fchk file, that have more precission than those read from the output file
                self.normal_modes=self.QM_output.normal_coordinates[nc]
                for prop in self.properties.keys():
                    self.properties[prop]=self.QM_output.properties[prop][nc]
                
            self.dipole_moment=self.QM_output.dipole_moments[nc]
            self.polarizability=self.QM_output.polarizabilities[nc]
            self.int_coord_definitions=self.QM_output.int_coord_definitions 

            
            #if self.ORCA_output.outfile=="" and self.G16_output.outfile!="": self.__update_coords_from_G16output(f)   y borrar __update_coords_from_G16output
            #elif self.ORCA_output.outfile!="" and self.G16_output.outfile=="": self.__update_coords_from_ORCAoutput(f)

        # update the coords of the atoms using the information in coords; coords could be a (3,n) list, a (3,n) np.array, a list of atoms and a Molecular_structure object
        def update_coords(self,coords):
            if type(coords)==list: n_coords=np.array(coords)
            if type(coords)==np.ndarray: n_coords=coords
            if type(coords)==type(self.atom_list[0]):
                n_coords=np.array([a.coord for a in coords])
            if type(coords)==type(self):
                n_coords=np.array([a.coord for a in coords.atom_list])
            for i in range (0,len(self.atom_list)):
                self.atom_list[i].coord=n_coords[i]
        
        # update the coords of the atoms using the information in an xyz file
        def update_coords_from_xyz(self,xyz_file):
            if not xyz_file.endswith(".xyz") : xyz_file=xyz_file+".xyz"
            coords=[]
            with open(xyz_file) as f:
                for l in f.readlines():
                    c=str.split(l)
                    if len(c)==4:
                        coords.append([c[1],c[2],c[3]])
            self.update_coords(coords)

        def mix_coordinates(self,other_molecule):
            for a,aa in zip(self.atom_list,other_molecule.atom_list):
                a.coord=(a.coord+aa.coord)/2



        #METHODS FOR MOVING AND CHANGING THE GEOMETRY OF THE MOLECULE 
        #my own implementation of the Dijstra algorithm to calculate how many bonds are between two atoms
        #returns a list with the bonds beween each atom (starging with index=) and the atom passed
        #it will be neded to rotate bonds or changing angles

        #translate and rotate atoms coordinates using vector v, that includes information of translation and rotation
        def trans_and_rot(self,v):

            if type(v)==list and len(v)==6: v=np.array(v)
            elif type(v)==list and len(v)==2 and type(v[0])==list and len(v[0])==3: v=np.array(v[0]+v[1])

            curr_coords=np.array( [ a.coord   for a in self.atom_list  ])
            centroid=np.mean(curr_coords,axis=0)
            r=Rotation.from_euler("zyx",v[3:6])   #needs scicpy
            new_coords=r.apply(curr_coords-centroid)+centroid+np.array(v[:3])
            self.update_coords(new_coords)

        def bonds_between(self,atom): return self.__bonds_between(atom)
        def __bonds_between(self,atom):

            if type(atom)==int: atom1=atom
            elif type(atom)==type(self.list_of_atoms[0]): atom1=atom.atom_number

            distances=np.array([[i,9999] for i in range(1,len(self.atom_list)+1)],dtype=int)
            distances[atom1-1,1]=0
            not_visited=np.ones(len(self.atom_list),dtype=bool)
            old_number_of_not_visited=np.sum(not_visited)
            not_visited[atom1-1]=False            
            curr_node_number=atom1
            curr_node_atom=self.atom(curr_node_number)
            curr_node_dist=0
            for c in curr_node_atom.connection: distances[ c[0]-1,1 ]  = curr_node_dist+1
            while np.any( not_visited==True ) and (old_number_of_not_visited > np.sum(not_visited)):
                i=np.argmin(distances[not_visited][:,1])
                curr_node_number=distances[not_visited][i,0]
                curr_node_dist=distances[not_visited][i,1]
                curr_node_atom=self.atom(curr_node_number)
                for c in curr_node_atom.connection: 
                    distances[ c[0]-1,1 ]  = np.where(curr_node_dist+1<distances[ c[0]-1,1 ], curr_node_dist+1, distances[ c[0]-1,1 ])
                old_number_of_not_visited=np.sum(not_visited)
                not_visited[ curr_node_number -1 ]=False
            return distances[:,1].tolist()

        def change_bond_distance(self,atom1,atom2,increment):
            if type(atom1)==int:  atom1_coord=self.atom(atom1).coord
            if type(atom2)==int:  atom2_coord=self.atom(atom2).coord
            if type(atom1)==type(self.atom_list[0]): atom1_coord=atom1.coord
            if type(atom2)==type(self.atom_list[0]): atom2_coord=atom2.coord  
            k=atom2_coord-atom1_coord    
            if type(increment) is str:
                if "%" not in increment and "-" not in increment and "+" not in increment: 
                    
                    increment=float(increment)
                elif "%" in increment and "-" not in increment and "+" not in increment: 
                    increment=( (float(increment.strip("%"))/100 ) -1.0 )* (np.sum(k**2)**0.5)

                elif "%" in increment and "+" in increment: 
                    increment=( float(increment.strip("%").strip("+"))/100   )* (np.sum(k**2)**0.5) 

                elif "%" in increment and "-" in increment: 
                    increment= -(1- (float(increment.strip("%").strip("-") )/100)  ) * (np.sum(k**2)**0.5)                 
            k=increment*k/(np.sum(k**2)**0.5)
            for atom in self.atom_list:
                if not (atom.atom_number in [atom1,atom2]):
                    bond_distances= self.__bonds_between(atom.atom_number)
                    if bond_distances[atom2-1]<9999 and bond_distances[atom2-1]<bond_distances[atom1-1]:
                        atom.coord=atom.coord+k 
                if atom.atom_number==atom2: atom.coord=atom.coord+k

        # Use Rodriguez's formula to rotate around bond between atoms 1 and 2; atoms closer to atom2 are moved.     
        def rotate_bond(self,atom1,atom2,angle): 
            if type(atom1)==int: atom1_coord=self.atom(atom1).coord
            if type(atom2)==int: atom2_coord=self.atom(atom2).coord
            if type(atom1)==type(self.atom_list[0]):  atom1_coord=atom1.coord
            if type(atom2)==type(self.atom_list[0]):  atom2_coord=atom2.coord
            angle=angle* np.pi/180.0 
            k=atom1_coord-atom2_coord
            k=k/(np.sum(k**2)**0.5)            
            for atom in self.atom_list:
                if not (atom.atom_number in [atom1,atom2]):
                    bond_distances= self.__bonds_between(atom.atom_number) 
                    if bond_distances[atom2-1]<9999 and bond_distances[atom2-1]<bond_distances[atom1-1]:
                        v=atom.coord-atom2_coord
                        v_rot=v*np.cos(angle)+ np.cross(k,v)*np.sin(angle) +  k*(np.dot(k,v))*(1-np.cos(angle))
                        atom.coord= v_rot+atom2_coord

        # change angle between atoms 1 (central atom), 2 and 3, moving atoms bound to 3    
        def change_angle(self,atom1,atom2,atom3,angle):
            if type(atom1)==int: atom1_coord=self.atom(atom1).coord
            if type(atom2)==int: atom2_coord=self.atom(atom2).coord
            if type(atom3)==int: atom3_coord=self.atom(atom3).coord
            if type(atom1)==type(self.atom_list[0]): atom1_coord=atom1.coord
            if type(atom2)==type(self.atom_list[0]): atom2_coord=atom2.coord
            if type(atom3)==type(self.atom_list[0]): atom3_coord=atom3.coord
            angle=-angle* math.pi/180.0 
            aux_v=np.cross(atom2_coord-atom1_coord,atom3_coord-atom1_coord)
            aux_v=aux_v/(np.sum(aux_v**2)**0.5) 
            aux_coord= atom1_coord+aux_v
            k=atom1_coord-aux_coord
            k=k/(np.sum(k**2)**0.5)            
            for atom in self.atom_list:
                if not (atom.atom_number in [atom1,atom2]):
                    bond_distances= self.__bonds_between(atom.atom_number) 
                    if bond_distances[atom3-1]<9999 and bond_distances[atom3-1]<bond_distances[atom1-1]:
                        v=atom.coord-atom1_coord
                        v_rot=v*math.cos(angle)+ np.cross(k,v)*math.sin(angle) +  k*(np.dot(k,v))*(1-math.cos(angle))
                        atom.coord= v_rot+atom1_coord

        # reflex atoms coordinates (for creating enantiomers)
        def reflex(self,plane="XY"):
            if type(plane)==str: 
                if plane in ["XY","xy","z"]: m=[1,1,-1]
                elif plane in  ["XZ","xz","y"]: m=[1,-1,1]
                elif plane in  ["YZ","yz","x"]: m=[-1,1,1]
            elif type(plane)==list and len(plane)==3: m=list
            for atom in self.atom_list:atom.coord=atom.coord*m

        # modify the coordinates according to the normal model specified; if red_dist==True, the distance between atom_1 and atom_2 
        # (or the distance between atoms[]) will be reduced. If no atoms are specified, it will try to go toward increasing the number of bonds 
        # (this will probably work bad in displacement reactions, but works on associative or dissociative TSs).
        # if exclude_H_in_red_dist=False, H atoms will also be considered to determine if the TSs is associative or dissociative 
        # (with proton transfer concerted with the bond forming or breaking, it is better to let exclude_H_in_red_dist=True) 
        def aply_normal_mode(self,normal_mode_number=0,atom_1=0,atom_2=0,atoms=[],group_1=0,group_2=0,groups=[],factor=1.0,red_dist=True, exclude_H_in_red_dist=True):

            if atoms!=[] and len(atoms)==2: 
                atom_1=atoms[0]
                atom_2=atoms[1]

            if groups!=[] and len(groups)==2:
                group_1=groups[0]
                group_2=groups[1]
            if type(group_1)==type(1): group_1=self.groups[group_1-1]
            if type(group_2)==type(1): group_2=self.groups[group_2-1]
            

            if self.normal_modes==[]: self.read_normal_modes()
            c=np.array([a.coord for a in self.atom_list])
            coords_1=c+factor*np.array(self.normal_modes[normal_mode_number])
            coords_2=c-factor*np.array(self.normal_modes[normal_mode_number])
            if atom_1!=atom_2: 
                if np.sum(coords_1[atom_1-1]-coords_1[atom_2-1])**2 < np.sum(coords_2[atom_1-1]-coords_2[atom_2-1])**2:
                    if red_dist: self.update_coords(coords_1)
                    else: self.update_coords(coords_2)
                else:
                    if red_dist: self.update_coords(coords_2)
                    else: self.update_coords(coords_1) 

            if group_1!=group_2:
                weights=np.array( [ self.atomic_weights[ a.symbol.lower()  ] for a in self.atom_list  ])
                weights_g1= np.array( [ self.atomic_weights[ self.atom(i).symbol.lower()  ] for i in group_1  ]   )
                weights_g2= np.array( [ self.atomic_weights[ self.atom(i).symbol.lower()  ] for i in group_2  ]   )
                c1_g1= np.array( [ coords_1[i-1] for i in group_1  ])
                c1_g2= np.array( [ coords_1[i-1] for i in group_2  ]) 
                c2_g1= np.array( [ coords_2[i-1] for i in group_1  ])
                c2_g2= np.array( [ coords_2[i-1] for i in group_2  ])
                com1_g1=(np.average(c1_g1,axis=0,weights=weights_g1))
                com1_g2=(np.average(c1_g2,axis=0,weights=weights_g2))
                com2_g1=(np.average(c2_g1,axis=0,weights=weights_g1))
                com2_g2=(np.average(c2_g2,axis=0,weights=weights_g2))

                dist_1= ( np.sum( (com1_g1-com1_g2)**2 ) )**0.5
                dist_2= ( np.sum( (com2_g1-com2_g2)**2 ) )**0.5

                if dist_1>dist_2 and red_dist or (dist_1>dist_2 and not red_dist) : self.update_coords(coords_2)
                if dist_1<dist_2 and red_dist or (dist_1<dist_2 and not red_dist) : self.update_coords(coords_1)



            else:
                distance_changes=0.0
                for a in range(0,len(self.atom_list)):
                    if self.atom_list[a].symbol=="H": continue # do not work with H (to prevent considering H transfers)
                    for aa in range(a+1,len(self.atom_list)):
                        if exclude_H_in_red_dist and self.atom_list[aa].symbol=="H": continue 
                        if self.excess_distance([a+1,aa+1])>2.0 : continue # do not compare distances of atoms that are not close
                        dist_1=np.sum((coords_1[a]-coords_1[aa])**2)
                        dist_2=np.sum((coords_2[a]-coords_2[aa])**2)
                        original_dist=np.sum((c[a]-c[aa])**2)
                        bond_distance_change=(dist_1-dist_2)/original_dist
                        if (bond_distance_change)>1.5*factor or (bond_distance_change)<-1.5*factor : distance_changes+=bond_distance_change
                if distance_changes>0 and red_dist: self.update_coords(coords_2)
                else:self.update_coords(coords_1)


        #calculates the rmsd between self and other molecule; asumes that the number and order of atoms is the same
        def rmsd(self,other_moecule):
            rmsd=0.0
            for a,aa in zip(self.atom_list,other_molecule.atom_list):
                rmsd+=(a.coords-aa.coords)**2
            return rmsd**0.5/len(self.atom_list)


        #implementation of the kabsch algorithm; returns the rmsd, the translation vector, the rotation matrix and the coordinates of the fragment
        #after overlaying the fragment over the atoms of the current molecule described in the list_of_atoms array
        #fragment can be a np.array, a list of coordinates, a list of atom objects or a Molecular_structure object
        #if exclude_H = True and fragment_coords is a Molecular_structure object or a list of atom objects, H will not be used
        #to fit the two structures, the rotational matrix should be applied:   coords=fragment_coords.dot(rotation)+centroid; and then: update_coords(coords)
        def __kabsch (self,coords, fragment_coords,exclude_H=False): 

            centroid=np.mean(coords,axis=0)
            fragment_centroid=np.mean(fragment_coords,axis=0)
            coords=coords-centroid
            fragment_coords=fragment_coords-fragment_centroid

            covariance_matrix=fragment_coords.T.dot(coords)
            u,s,v=np.linalg.svd(covariance_matrix)
            d=np.linalg.det(v)*np.linalg.det(u)
            if d<0.0:
                s[-1]=-s[-1]
                u[:,-1]=-u[:,-1]
            rotation=u.dot(v)
            return centroid,rotation

        def min_rmsd(self,fragment,list_of_atoms=[],exclude_H=False,return_max_dist=False,reflex=False):
                              
            if type(fragment)==np.ndarray: fragment_coords=fragment
            elif type(fragment)==list: 
                if type(fragment[0])==list: fragment_coords=np.array(fragment)
                elif type(fragment[0])==type(self.atom_list[0]):
                    if exclude_H: fragment_coords=np.array([a.coord for a in fragment if a.symbol.lower()!="h"])  
                    else: fragment_coords=np.array([a.coord for a in fragment])
            elif type(fragment)==type(self):
                if exclude_H: fragment_coords=np.array([a.coord for a in fragment.atom_list if a.symbol.lower()!="h"])  
                else: fragment_coords=np.array([a.coord for a in fragment.atom_list]) 

            if reflex: fragment_coords=fragment_coords*np.array([1,1,-1])
            if len(list_of_atoms)==0: list_of_atoms= [a for a in range(1,len(self.atom_list)+1)]
            
            if exclude_H:
                coords=np.array([self.atom(i).coord for i in list_of_atoms if self.atom(i).symbol.lower()!="h"])
                #copy_list_of_atoms=copy.deepcopy(list_of_atoms)
                #for a in copy_list_of_atoms:
                #    if self.atom(a).symbol.lower()=="h": list_of_atoms.remove(a) 
            else:
                coords=np.array([self.atom(i).coord for i in list_of_atoms]) 

            """
            #using scipy rotation:            
            centroid=np.mean(coords,axis=0)
            fragment_centroid=np.mean(fragment_coords,axis=0)
            rotation,rmsd= Rotation.align_vectors(coords-centroid,fragment_coords-fragment_centroid)
            fragment_coords=rotation.apply(fragment_coords-fragment_centroid) + centroid
            rmsd= rmsd / (len(fragment_coords)**0.5)
            """
            
            #using my own implemented kabsch algorithm; note that rmsd result is different!
            fragment_centroid=np.mean(fragment_coords,axis=0)
            fragment_coords=fragment_coords-fragment_centroid
            #print (len(fragment_coords))
            translation,rotation=self.__kabsch(coords,fragment_coords)
            fragment_coords=fragment_coords.dot(rotation)+translation
            rmsd=(np.sum((fragment_coords-coords)**2/len(fragment_coords)))**0.5
            #rmsd=(np.sum((fragment_coords-coords)**4/len(fragment_coords)))**0.25

            if not return_max_dist:
                return rmsd,fragment_centroid,rotation,translation,fragment_coords
            else:
                max_dist= (np.max( np.sum((fragment_coords-coords)**2,axis=1)  ) )**0.5 
                return rmsd,max_dist,fragment_centroid,rotation,translation,fragment_coords

        #similar to min_rmsd(), but instead of searching to minimize the rmsd distance, tries to minimize the "distortion energy":
        #the energy needed to distort one geometry in the other using the hessian and an harmonic approximation
        def min_hess_distort(self,ref_molecule,atom_number_correspondence={}):

            reference_molecule=copy.deepcopy(ref_molecule)

            if atom_number_correspondence=={}:
                atom_number_correspondence= dict( [(value,key) for key,value in self.ref_to_new_numbers.items()] )
                #atom_number_correspondence= self.ref_to_new_numbers

            #get the coordinates of the atoms in current_molecule in a 3xN np.array
            #to change it to a 1-D vector (to multiply with the hessian), use: curr_molecule_coords.reshape(1,-1)
            curr_molecule_coords=np.array( [ a.coord   for a in self.atom_list  ])

            #get the coordinates of the atoms in reference_molecule that have a corresponding atom in current_molecule, in a 3xN np.array
            ref_molecule_coords=np.array( [reference_molecule.atom( atom_number_correspondence[a.atom_number] ).coord for a in self.atom_list] )

            #calculates the distortion energy after moving and rotate the current molecule (note that cart_hess is invariant to rotation and translations)
            def __distortion_energy(v):
                centroid=np.mean(curr_molecule_coords,axis=0)
                ref_centroid=np.mean(ref_molecule_coords,axis=0)
                r=Rotation.from_euler("zyx",v[3:6])
                new_curr_molecule_coords=r.apply( curr_molecule_coords-centroid ) + centroid + v[0:3]
                diff_vector=((new_curr_molecule_coords-centroid-ref_molecule_coords+ref_centroid))*self.angstrom_to_bohr   #abs needed?
                diff_vector=diff_vector.reshape(1,-1)[0]
                #print (abs(diff_vector.dot(self.cart_hess).dot(diff_vector.T)))
                #print (str(v)+":"+str( diff_vector.dot(self.cart_hess).dot(diff_vector.T) )     )
                #print (r.as_quat())
                sum=0
                #identity=np.identity(len(self.cart_hess))
                e=(diff_vector.dot(self.cart_hess).dot(diff_vector.T))
                #print (e)
                return e
                #return (diff_vector.dot(identity).dot(diff_vector.T))

            #optimization using simpex method           
            #initial_simplex=np.array([0.02,0.02,0.02,np.pi,1,1])*(2*np.random.rand(7,6)-1)
            #res=minimize(__distortion_energy,np.array([0,0,0,0,0,0]),method="Nelder-Mead",tol=0.00001,options={'initial_simplex': initial_simplex,'disp': 1})

            #optimization using SLSQP method
            bounds = Bounds([-35, -35, -35,-np.pi,-np.pi,-np.pi], [35, 35, 35,np.pi,np.pi,np.pi])
            #res=minimize(__distortion_energy,np.array([0,0,0,0,0,0]),method="SLSQP",bounds=bounds,options={'disp': 1})
            res=minimize(__distortion_energy,np.array([0,0,0,0,0,0]),method="SLSQP",options={'disp': 1,'maxiter':100000 })
            #res=minimize(__distortion_energy,method="SLSQP",options={'disp': 1})

            v=res.x.tolist()
            #rotation_vector=res_list[3]*np.array([res_list[4],res_list[5],(1-res_list[4]**2-res_list[5]**2)**0.5])
            distortion_energy=res.fun

            #translate and rotate the molecule:
            centroid=np.mean(curr_molecule_coords,axis=0)
            ref_centroid=np.mean(ref_molecule_coords,axis=0)
            r=Rotation.from_euler("zyx",v[3:6])
            new_curr_molecule_coords=r.apply( curr_molecule_coords-centroid ) + centroid + v[0:3]
            self.update_coords(new_curr_molecule_coords)
            

            
            #print ("centroid:"+str(centroid))
            #q=quaternion.from_euler_angles(euler_angles)
            #new_curr_molecule_coords=quaternion.rotate_vectors(q,curr_molecule_coords-centroid)+translate_vector+centroid

            #new_curr_molecule_coords=curr_molecule_coords-centroid
            #new_curr_molecule_coords=quaternion.rotate_vectors( q,new_curr_molecule_coords )
            #new_curr_molecule_coords=new_curr_molecule_coords+centroid+translate_vector
            #borrame
            #diff_vector=((new_curr_molecule_coords-ref_molecule_coords))*self.angstrom_to_bohr
            #print (diff_vector)
            #diff_vector=diff_vector.reshape(1,-1)
            #print (diff_vector)
            #print ("*********")
            #print (diff_vector.dot(self.cart_hess).dot(diff_vector.T))
            #print ("**********")
            #self.update_coords(new_curr_molecule_coords)

            #return distortion_energy, translate_vector, rotation_vector 
            return distortion_energy, v[0:3], v[3:6]



        # return a list of the rings in which the atom is involved based on the connectivity (up to 7 membered rings)
        # each ring is an array with the atoms included in them
        # an indentation nightmare:
        # it takes an atom in a molecule, gets its connectivity, and for each atom in the connectivity, find the connectivity; for each atom in the connectivity,
        # gets the connectivity and checks if the first atom was in there (3-membered ring located); if not, for each atom in the connectivity gets its connectivity, and 
        # checks if the first atom was in there (4-membered ring located).... it also prevents the routes to turn back.         
        def rings (self,atom_number):

            # By using this method to append, permutations are removed
            def __append_once(new_member,list):
                already=False
                for l in list:
                    if l==new_member or sorted(l)==sorted(new_member):already=True
                if not already:
                    if len(set(new_member))==len(new_member):
                        list.append(new_member)
                return list
                
            rings=[]
            atom1=self.atom(atom_number)
            for conn_to_atom1 in atom1.connection:
                atom2=self.atom(conn_to_atom1[0])
                for conn_to_atom2 in atom2.connection:
                    if conn_to_atom2[0]!=atom1.atom_number:
                        atom3=self.atom(conn_to_atom2[0])
                        for conn_to_atom3 in atom3.connection:
                            if conn_to_atom3[0]!=atom2.atom_number:
                                if conn_to_atom3[0]==atom_number:
                                    __append_once([atom_number,conn_to_atom1[0],conn_to_atom2[0]],rings)
                                else:
                                    atom4=self.atom(conn_to_atom3[0])
                                    for conn_to_atom4 in atom4.connection:
                                        if conn_to_atom4[0]!=atom3.atom_number:
                                            if conn_to_atom4[0]==atom_number:
                                                __append_once([atom_number,conn_to_atom1[0],conn_to_atom2[0],conn_to_atom3[0]],rings)  
                                            else:
                                                atom5=self.atom(conn_to_atom4[0])
                                                for conn_to_atom5 in atom5.connection:
                                                    if conn_to_atom5[0]==atom_number:
                                                        __append_once([atom_number,conn_to_atom1[0],conn_to_atom2[0],conn_to_atom3[0],conn_to_atom4[0]],rings)
                                                    else:
                                                        atom6=self.atom(conn_to_atom5[0])
                                                        for conn_to_atom6 in atom6.connection:
                                                            if conn_to_atom6[0]==atom_number:
                                                                __append_once([atom_number,conn_to_atom1[0],conn_to_atom2[0],conn_to_atom3[0],conn_to_atom4[0],conn_to_atom5[0]],rings)
                                                            else:
                                                                atom7=self.atom(conn_to_atom6[0])
                                                                for conn_to_atom7 in atom7.connection:
                                                                    if conn_to_atom7[0]==atom_number:
                                                                        __append_once([atom_number,conn_to_atom1[0],conn_to_atom2[0],conn_to_atom3[0],conn_to_atom4[0],conn_to_atom5[0],conn_to_atom6[0]],rings)
                
            return rings

        #remove?
        def DFS_rings (self):
            rings=[]
            for curr_atom in self.atom_list:
                if len(curr_atom.connection)>1: #omitt atoms with only one connection
                    print ("working with atom: "+str(curr_atom.atom_number))                                         
                    for backward in [False,True]: #will do the DFS in forward and reverse order to assure that all cycles are found
                        stack=[curr_atom]
                        discovered=[]
                        while len(stack)!=0:
                            investigating_atom=stack.pop()
                            #print("investigating: "+str(investigating_atom.atom_number) ) #for debuggin
                            if investigating_atom.atom_number not in discovered:
                                discovered.append(investigating_atom.atom_number)
                                #print(discovered) #for debuggin
                                if backward: investigating_atom_connections=reversed(investigating_atom.connection)
                                else: investigating_atom_connections=investigating_atom.connection 
                                for conn_atom in investigating_atom_connections:
                                    if conn_atom[0] not in discovered and len( self.atom(conn_atom[0]).connection)>1: #exclude atoms with only one connection
                                        #print ("adding to stack "+str(self.atom(conn_atom[0]).atom_number) )
                                        stack.append(self.atom(conn_atom[0])) #add it to the stack to process it later
                                        #if len( self.atom(conn_atom[0]).connection)>1 and self.atom(conn_atom[0]).atom_number==curr_atom.atom_number: #for debuggin
                                        #print (discovered)
                                if curr_atom.atom_number in [x[0] for x in  self.atom(discovered[-1]).connection[:] ] and len(discovered)>2 :
                                    #print ("found a cycle: "+str(discovered)) #for debuggin
                                    #generate cyclic permutations of the new vector:
                                    discovered_permutations= [[discovered[i - j] for i in range(len(discovered))] for j in range(len(discovered), 0, -1)]
                                    if not ( any(d in rings for d in discovered_permutations) ):
                                        #print ("adding cycle") #for debuggin
                                        rings.append(discovered)
                                    #else: print("but not new") #for debuggin
            #for r in rings: print(r) #for debuggin'


        #return an np.array with the center of mass, ( eigenvalues of the inertia matrix and principal axis if calculate_prnpl_axis==True )
        #for the atoms specified in list_of_atoms (if empty, for the whole molecule)
        def center_of_mass(self,list_of_atoms=[],calculate_prnpl_axis=True):

            if len(list_of_atoms)==0:
                weights=np.array( [ self.atomic_weights[ a.symbol.lower()  ] for a in self.atom_list  ])
                coords=np.array( [a.coord for a in self.atom_list])
                
            if len(list_of_atoms)!=0 and type(list_of_atoms[0])==type(1):
                weights=np.array( [self.atomic_weights[ self.atom(a).symbol.lower()]   for a in list_of_atoms])
                coords=np.array( [self.atom(a).coord for a in list_of_atoms])

            if len(list_of_atoms)!=0 and type(list_of_atoms[0])==type(self.atom_list[0]):
                weights=np.array( [self.atomic_weights[a.symbol.lower()]] for a in list_of_atoms)
                coords=np.array( [a.coord for a in list_of_atoms])

            center_of_mass=(np.average(coords,axis=0,weights=weights))
            if calculate_prnpl_axis==True:
                ixx=weights.dot(np.array([v[1]**2+v[2]**2 for v in coords]))
                iyy=weights.dot(np.array([v[0]**2+v[2]**2 for v in coords]))
                izz=weights.dot(np.array([v[0]**2+v[1]**2 for v in coords]))
                ixy=weights.dot(np.array([-v[0]*v[1] for v in coords]))
                ixz=weights.dot(np.array([-v[0]*v[2] for v in coords]))
                iyz=weights.dot(np.array([-v[1]*v[2] for v in coords]))

                inertia_matrix=np.array([[ixx,ixy,ixz],[ixy,iyy,iyz],[ixz,iyz,izz]])
                eig,U=np.linalg.eig(inertia_matrix)

                principal_axis=[u for _,u in sorted(zip(eig,U))]
                eig=sorted(eig)
            
                return center_of_mass,eig,principal_axis
            else: return center_of_mass



        #METHODS SPECIFIC FOR MANIPULATING MM3 ATOM TYPES


        #changes the atom types based on the information on passed atom_types. If force==true, will overwrite atom types. Otherwise will only update them for atoms not assigned
        def update_atom_types(self,atom_types,force=False):
            if type(atom_types)==type(self.atom_list) and type(atom_types[0])==type(self.atom_list[0]):
                list_of_atom_types=[atom.atom_type for atom in atom_types]
            if type(atom_types)==type(self):
                list_of_atom_types=[atom.atom_type for atom in atom_types.atom_list]
            if type (atom_types)==list and type(atom_types[0])==int : list_of_atom_types=atom_types        

            for atom, new_type in zip(self.atom_list,list_of_atom_types):
                #delocalized MM3 pi systems (atom_type>200) only if the atom is in the HL layer:
                if atom.atom_layer=="H" and new_type>200: new_type=new_type-200
                if force:
                    atom.atom_type=str(new_type)
                else:
                    if atom.atom_type=="" or atom.atom_type=="0": 
                        atom.atom_type=str(new_type)
                # for ferrocenes, etc
                if atom.atom_type=="113" or atom.atom_type=="114": atom.charge=-0.2
                if atom.atom_type in ["61","64"]: atom.charge=2.0
                i=i+1
            """ old way, delete if ok
            i=0
            for atom in self.atom_list:
                new_type=int(list_of_atom_types[i])
                #delocalized MM3 pi systems (atom_type>200) only if the atom is in the HL layer:
                if atom.atom_layer=="H" and new_type>200: new_type=new_type-200
                if force:
                    atom.atom_type=str(new_type)
                else:
                    if atom.atom_type=="" or atom.atom_type=="0": 
                        atom.atom_type=str(new_type)
                # for ferrocenes, etc
                if atom.atom_type=="113" or atom.atom_type=="114": atom.charge=-0.2
                if atom.atom_type in ["61","64"]: atom.charge=2.0
                i=i+1
            """


        #set the MM3 atom type of atom based on the connectivity
        def set_MM3_atom_type(self,atom_number,MM3_FF=None,interactive=False):
            if type(atom_number)!=type(1): atom_number=atom_number.atom_number #in case an atom object and not the atom number is given
            a_t,a_t_warning=self.__determine_MM3_atom_type(atom_number,MM3_FF,interactive=interactive)
            self.atom(atom_number).atom_type=str(a_t)
            self.atom(atom_number).atom_type_warning=(a_t_warning)

        
        #set all MM3 atom types based on the connectivity
        def set_MM3_atom_types(self,MM3_FF=None,interactive=False):
            for a in self.atom_list:
                self.set_MM3_atom_type(a,MM3_FF=None,interactive=interactive)


        # built-in engine to find-out atom-types...  mistakes in the assignments?
        # it returns the atom type and a warning message (that is not used right now)
        # it is private to force using set_MM3_atom_type
        # note: atom types larger than 200 are used for conjugate pi systems
        # they correspond to the atom_type-200 equivalents 
        def __determine_MM3_atom_type(self,atom_number,MM3_FF=None,interactive=False):

            problems=[]
            atom=self.atom(atom_number)
            s=""

            # H atoms:
            if atom.symbol=="H" and len(atom.connection)==0: #flying H atoms
                dist=1000.0
                nearest_symbol=""
                for atom2 in self.atom_list:
                    if atom2.atom_number!=atom.atom_number:
                        if self.distance([atom.atom_number,atom2.atom_number])<dist:
                            dist=self.distance([atom.atom_number,atom2.atom_number])
                            nearest_symbol=atom2.symbol
                if nearest_symbol=="O": return 21, "unbound H atom; assigned to type 21 because it is near a O atom, but it may also be a enol H (73)" 
                elif nearest_symbol=="N": return 23, "unbound H atom; assigned to type 23 because it is near a N atom, but it may also be a ammonium H (48) or amide H (28)"  
                elif nearest_symbol=="C": return 5, "unbound H atom; assigned to type 4 because it is near a C atom, but it may also be an acetilene H (124)" 
                elif nearest_symbol=="S": return 44, "unbound H atom; assigned to type 44 because it is near a S atom"

            if atom.symbol=="H" and len(atom.connection)>0: #only when the H is connected to something
                for c in atom.connection:
                    if c[2]=="O": # bound to an Oxygen atom
                        atom2=self.atom(c[0])
                        for c2 in atom2.connection:    
                            if c2[2]=="C":
                                atom3=self.atom(c2[0])
                                if len(atom3.connection)==3:
                                    for c3 in atom3.connection:
                                        if c3[2]=="O" and c3[1]>1: return 24,s     #     "COOH CARBOXYL"
                                        if c3[2]=="C" and c3[1]>1: return 73,s #"H-O ENOL/PHENOL" 
                                if len(atom3.connection)==4:
                                    atom3_hydrogen_substituents = sum(s.count("H") for s in atom3.connection)
                                    atom3_carbon_substituents = sum(s.count("C") for s in atom3.connection)
                                    if (atom3_hydrogen_substituents + atom3_carbon_substituents)==4: # no heteroatoms
                                        return 21,s #"-OH ALCOHOL"
                                    else:                               
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 21 (H in R3-C-O-H ALCOHOL);"
                                        s=s+" however, some of the R substituents are neither H or C "
                                        problems.append(s)
                                        return 21,s
                    if c[2]=="N": # bound to a Nitrogen atom
                        atom2=self.atom(c[0])
                        if len(atom2.connection)==3: #amine, amide... trisubstituted N
                            for c2 in atom2.connection:    
                                if c2[2]=="C":
                                    atom3=self.atom(c2[0])
                                    if len(atom3.connection)==3: # amide, urea, enamine
                                        for c3 in atom3.connection:
                                            if c3[2]=="O": return 28,s #"R-C(=O)N-H AMIDE"
                                            if c3[2]=="S" and len(atom3.connection)==3: 
                                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 28 (R-C(=O)N-H AMIDE);"
                                                s=s+" it is the H atom in a thioamide or thiourea"
                                                problems.append(s)   
                                                return 28,s
                                            if c3[2]=="C" and (c3[1]==2 or c3[1]==1.5): 
                                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 23 (R2-NH AMINE/IMINE);"
                                                s=s+" however, it is an aniline or an enamine"
                                                return 23,s  # NH AMINE/IMINE
                                    if len(atom3.connection)==4: # amine
                                        atom3_hydrogen_substituents = sum(s.count("H") for s in atom3.connection)
                                        atom3_carbon_substituents = sum(s.count("C") for s in atom3.connection)
                                        if (atom3_hydrogen_substituents + atom3_carbon_substituents)==4: #no heteroatoms
                                            return 23,s  # NH AMINE/IMINE
                                        else: 
                                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 23 (NH AMINE/IMINE);"
                                            s=s+" however, some of the R substituents are neither H or C "
                                            problems.append(s)
                                            return 23,s
                        if len(atom2.connection)==4: # tetrasubstituted N
                            atom2_hydrogen_substituents = sum(s.count("H") for s in atom2.connection)
                            atom2_carbon_substituents = sum(s.count("C") for s in atom2.connection)
                            if (atom2_hydrogen_substituents + atom2_carbon_substituents)==4: #no heteroatoms                    
                                return 48,s #"AMMONIUM"
                            else:        
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 48 (R3-N-H AMONIUM);"
                                s=s+" however, some of the R substituents are neither H or C. "
                                problems.append(s)
                                return 48,s                   

                    if c[2]=="S":  # bound to a S atom
                        atom2=self.atom(c[0])
                        atom2_hydrogen_substituents = sum(s.count("H") for s in atom2.connection)
                        atom2_carbon_substituents = sum(s.count("C") for s in atom2.connection)
                        if atom2_carbon_substituents + atom2_hydrogen_substituents == len(atom2.connection): # no heteroatoms
                            if len(atom2.connection)==2:
                                return 44,s #"SH THIOL"
                            else:
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 44 (SH THIOL);"
                                s=s+" however, S atoms has more than 2 substituents. "
                                problems.append(s)
                                return 44,s  #"SH THIOL"
                        else:
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 44 (SH THIOL);"
                            s=s+" however, S atom is substituted by atoms different than C or H. "
                            problems.append(s)
                            return 44,s  #"SH THIOL"                                          
                                                
                                            
                    if c[2]=="C": # bound to a C atom
                        atom2=self.atom(c[0])
                        if len(atom2.connection)==2: #acetilene
                            acetilene=True
                            for c in atom2.connection:
                                if c[0]!=atom2.atom_number and c[2]!="C": acetilene=False
                            if acetilene==True:
                                return 124,s #H-C ACETYLENE
                            else: 
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 124 (R#C-H ACETYLENE);"
                                s=s+" however,  R substituent is not C. "
                                problems.append(s)
                                return 124,s #H-C ACETYLENE
                        else:                    
                            return 5,s     # H EXCEPT ON N,O,S 

        
                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 5 (H EXCEPT ON N,O,S);"
                s=s+" however,  R substituent is not C "
                problems.append(s)
                return 5,s


            if atom.symbol=="C":
                number_of_oxygen_substituents= sum(s.count("O") for s in atom.connection)
                number_of_nitrogen_substituents= sum(s.count("N")for s in atom.connection)
                number_of_carbon_substituents= sum(s.count("C")for s in atom.connection)
                number_of_hydrogen_substituents= sum(s.count("H")for s in atom.connection)

                # check for ferrocenes
                for r in self.rings(atom_number): # check for the cycles involving this atom
                    if len(r)==5:
                        for a in self.atom_list:
                            if a.symbol in ["Fe","fe","Co","co","Ni","ni","Cr","cr","V","v","Mn","mn","Si","si","Ge","ge","Sn","sn","Pb","pb","Se","se","Te","te","Mg","mg","Ca","ca","Sr","sr","Ba","ba","La","la","Ce","ce","Pr","pr","Nd","nd","Pm","pm","Sm","sm","Eu","eu","Gd","gd","Tb","tb","Dy","dy","Ho","ho","Er","er","Tm","tm","Yb","yb","Lu","lu"]:
                                if self.distance([a.atom_number,atom.atom_number])<3.0:
                                    s= "Ferrocene-like C assigned"
                                    #print ("ferrocene C!"+ str(atom_number)+" with distance to M:"+ str(self.distance([a.atom_number,atom.atom_number])))
                                    if  number_of_hydrogen_substituents==1: 
                                        return 113,s  # do not return 313... should not use pisystem with cyclopentadienyl ring!!!!
                                    else: 
                                        return 114,s


                if len(atom.connection)==2: #alkyne,ketene, allene
                    if number_of_carbon_substituents==1 and number_of_hydrogen_substituents==1:
                        if atom.connection[0][1]==2:
                            return 4,s #"CSP ALKYNE"
                        elif atom.connection[0][1]<2:
                            return 204,s #"CSP ALKYNE" delocalized (>200)
                    if number_of_carbon_substituents==2:
                        if atom.connection[0][1]>2 and atom.connection[1][1]>2: #no possibility of delocalized (no pibond definition in MM3)
                            return 68,s #"ALLENE CSP =C="
                        else:
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 68 (ALLENE CSP =C=);"
                            s=s+" however,  at least one single bond found in central atom "
                            problems.append(s)
                            return 68,s #"ALLENE CSP =C="
                    if number_of_carbon_substituents==1 and number_of_oxygen_substituents==1:
                        if atom.connection[0][1]>2 and atom.connection[1][1]>2:  #no possibility of delocalized (no pibond definition in MM3)
                            return 106,s #"=C=O KETENE"
                        else:
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 105 (=C=O KETENE);"
                            s=s+" however,  at least one single bond found in central atom "
                            problems.append(s)                   
                            return 106,s #"=C=O KETENE"
                        if number_of_nitrogen_substituents==1 and number_of_oxygen_substituents==1: 
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 106 (R=C=O KETENE);"
                            s=s+" however,  it seems a isocyanate "
                            problems.append(s)                               
                            return 106,s #"=C=O KETENE"               
                        if number_of_carbon_substituents==0 and number_of_oxygen_substituents==1: 
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 106 (R=C=O KETENE);"
                            s=s+" however,  R substituent is not C "
                            problems.append(s)                               
                            return 106,s #"=C=O KETENE"

                if len(atom.connection)==3:  #sp2 carbon atom except ketene or allene           

                    if number_of_oxygen_substituents==1: #if only one oxygen:
                        if any(c[2]=="O" and c[1]>2 for c in atom.connection): # if it is a carbonyl; 
                            for r in self.rings(atom_number): # in a cycle
                                if len(r)==3:
                                    return 67,s # CYCLOPROPANONE; no possibility of delocalized (no pibond definition in MM3)
                                if len(r)==4:
                                    if number_of_nitrogen_substituents==1:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 58 (CYCLOBUTANONE);"
                                        s=s+" however, it corresponds to a b-lactame; consider atom_type=3 instead"
                                        problems.append(s)
                                        return 58,s 
                                    return 58,s # CYCLOBUTANONE; no possibility of delocalized (no pibond definition in MM3)
                                if number_of_nitrogen_substituents>0 : #amide or urea
                                    if any(c[2]=="N" and c[1]==2 for c in atom.connection):
                                        return 3,s #     "CSP2 CARBONYL"  I do not find anything that fits better to an amide
                                    if any(c[2]=="N" and c[1]==1.5 for c in atom.connection):
                                        return 203,s #     "CSP2 CARBONYL"  delocalized                            
                            for c in atom.connection:
                                if c[2]=="O": atom2=self.atom(c[0])
                            if len(atom2.connection)>1: # if the carbonyl oxygen is substituted:
                                return 71,s #"KETONIUM CARBON"; no possibility of delocalized (no pibond definition in MM3)

                        if any(c[2]=="O" and c[1]==1 for c in atom.connection): # if the oxygen is bound with a single bond:
                            if any(c[2]=="C" and c[1]==2 for c in atom.connection): # if it is bound to a carbon atom with a double bond:
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                s=s+" it seems an enol or enolate"
                                problems.append(s)                        
                                return 2,s #"CSP2 ALKENE for the enol
                            if any(c[2]=="C" and c[1]==1.5 for c in atom.connection): # if it is bound to a carbon atom with a 1.5 order bond:
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                s=s+" it seems an enol or enolate"
                                problems.append(s)                        
                                return 202,s #"CSP2 ALKENE for the enol delocalized

                            if any(c[2]=="C" and c[1]==2 for c in atom.connection): # if it is bound to a carbon atom with a bond order higher than 1:
                                for r in self.rings(atom_number): # is in a cycle
                                    if len(r)==3:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 38 (CSP2 CYCLOPROPENE);"
                                        s=s+" this assesment omits that it is bound to a O atom"
                                        return 38,s 
                                    if len(r)==4:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 57 (CSP2 CYCLOBUTENE);"
                                        s=s+" this assesment omits that it is bound to a O atom"
                                        return 57,s  
                                    if len(r)==5:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                        s=s+" this assesment omits that it is in a 5-membered ring and bound to a O atom"
                                        return 2,s 
                                    if len(r)==6:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                        s=s+" it probably corresponds to a phenol"
                                        return 2,s       
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                s=s+" this assesment omits that it is bound to a O atom"
                                problems.append(s)                        
                                return 2,s #"CSP2 ALKENE

                            if any(c[2]=="C" and c[1]==1.5 for c in atom.connection): # if it is bound to a carbon atom with a bond order higher than 1:
                                for r in self.rings(atom_number): # is in a cycle
                                    if len(r)==3:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 38 (CSP2 CYCLOPROPENE);"
                                        s=s+" this assesment omits that it is bound to a O atom"
                                        return 238,s 
                                    if len(r)==4:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 57 (CSP2 CYCLOBUTENE);"
                                        s=s+" this assesment omits that it is bound to a O atom"
                                        return 257,s  
                                    if len(r)==5:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                        s=s+" this assesment omits that it is in a 5-membered ring and bound to a O atom"
                                        #return 2,s
                                        return 202,s 
                                    if len(r)==6:
                                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                        s=s+" it probably corresponds to a phenol"
                                        #return 2,s
                                        return 202,s       
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                s=s+" this assesment omits that it is bound to a O atom"
                                problems.append(s)                        
                                return 202,s #"CSP2 ALKENE


                    if number_of_oxygen_substituents>1: # if two or more  oxygens:
                        if any(c[2]=="O" and c[1]>1 for c in atom.connection): # if one of the oxygens is a carbonyl
                            for r in self.rings(atom_number): # is in a cycle
                                if len(r)==4:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 58 (CYCLOBUTANONE);"
                                    s=s+" however, it corresponds to a butirolactone; consider atom_type=3 instead"
                                    problems.append(s)
                                    return 58,s
                                if len(r)==5:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 58 (CYCLOBUTANONE);"
                                    s=s+" however, it corresponds to a valerolactone; consider atom_type=3 instead"
                                    problems.append(s)
                                    return 58,s
                            if any(c[2]=="O" and c[1]==2 for c in atom.connection):        
                                return 3,s    #     "CSP2 CARBONYL: amide, carboxylic acid, ester, carbamate...
                            if any(c[2]=="O" and c[1]==1.5 for c in atom.connection):        
                                return 203,s    #     "CSP2 CARBONYL: amide, carboxylic acid, ester, carbamate...


                        if any(c[2]=="C" and c[1]==2 for c in atom.connection): # if it is bound to a C atom with a bound order 2:
                            for r in self.rings(atom_number): # is in a cycle
                                if len(r)==5:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                    s=s+" this assesment omits that it is connected to two O atoms (probably an ester enolate)."
                                    problems.append(s)
                                    return 2,s
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" this assesment omits that it is connected to two O atoms (probably an ester enolate)."
                            problems.append(s)
                            return 2,s 

                        if any(c[2]=="C" and c[1]==1.5 for c in atom.connection): # if it is bound to a C atom with a bound order 1.5:
                            for r in self.rings(atom_number): # is in a cycle
                                if len(r)==5:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                    s=s+" this assesment omits that it is connected to two O atoms (probably an ester enolate)."
                                    problems.append(s)
                                    return 202,s
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" this assesment omits that it is connected to two O atoms (probably an ester enolate)."
                            problems.append(s)
                            return 202,s   



                    if number_of_oxygen_substituents==0 and number_of_nitrogen_substituents==0: # if there are no oxygens and no nitrogens

                        for r in self.rings(atom_number):
                            if len(r)==3: # is in a cycle
                                return 38,s #CSP2 CYCLOPROPENE
                            if len(r)==4:
                                return 57,s #CSP2 CYCLOBUTENE

                        if any(c[2]==2 for c in atom.connection): # C=C bond localized
                            return 2,s #"CSP2 ALKENE"
                        if any(c[2]==2 for c in atom.connection): # C=C bond localized
                            for r in self.rings(atom_number):
                                if len(r)==6:
                                    return 2,s   #CSP2 ALKENE
                                if len(r)==5:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                    s=s+" this assesment omits that it is in a 5 membered ring."
                                    return 2,s
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" this assesment omits that bond order is smaller than 2."
                            return 2,s  # CSP2 ALKENE

                        if any(c[2]==1.5 for c in atom.connection): # C=C bond delocalized
                            for r in self.rings(atom_number):
                                if len(r)==6:
                                    return 202,s   #CSP2 ALKENE delocalized
                                if len(r)==5:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                    s=s+" this assesment omits that it is in a 5 membered ring."
                                    return 202,s
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" this assesment omits that bond order is smaller than 2."
                            return 202,s  # CSP2 ALKENE delocalized

                    if number_of_oxygen_substituents==0 and number_of_nitrogen_substituents==1:
                        for r in self.rings(atom_number):# is in a cycle
                            if any(c[2]=="C" and c[1]==2 for c in atom.connection):
                                if len(r)==5:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                    s=s+" this assesment omits that it seems the C atom in an pyrrole."
                                    problems.append(s) 
                                    return 2,s # CSP2 ALKENE
                                if len(r)==6:
                                    for c in atom.connection:
                                        if c[2]=="N" and c[0] in r:
                                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                            s=s+" it seems the C atom in an pyridine."
                                            problems.append(s)
                                            return 2,s # CSP2 ALKENE 

                            if any(c[2]=="C" and c[1]==1.5 for c in atom.connection):  # same as before, delocalized
                                if len(r)==5:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                    s=s+" this assesment omits that it seems the C atom in an pyrrole."
                                    problems.append(s) 
                                    return 202,s # CSP2 ALKENE
                                if len(r)==6:
                                    for c in atom.connection:
                                        if c[2]=="N" and c[0] in r:
                                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                                            s=s+" it seems the C atom in an pyridine."
                                            problems.append(s)
                                            return 202,s # CSP2 ALKENE 

                        if any(c[2]=="N" and c[1]==2 for c in atom.connection):
                            return 2,s # CSP2 ALKENE for imine, oxime, hydrazone
                        if any(c[2]=="N" and c[1]==1.5 for c in atom.connection):
                            return 202,s # CSP2 ALKENE for imine, oxime, hydrazone, delocalized

                    if number_of_oxygen_substituents==0 and number_of_nitrogen_substituents==2:
                        if any(c[1]==2 for c in atom.connection): 
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" it seems the C atom in an amidine"
                            problems.append(s)
                            return 2,s # CSP2 ALKENE
                        if any(c[1]==1.5 for c in atom.connection): 
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" it seems the C atom in an amidine"
                            problems.append(s)
                            return 202,s # CSP2 ALKENE                 


                    if number_of_oxygen_substituents==0 and number_of_nitrogen_substituents==3:
                        if any(c[1]==2 for c in atom.connection): 
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" it seems the C atom in an guanidine"
                            problems.append(s)
                            return 2,s # CSP2 ALKENE 
                        if any(c[1]==1.5 for c in atom.connection): 
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 2 (CSP2 ALKENE);"
                            s=s+" it seems the C atom in an guanidine"
                            problems.append(s)
                            return 202,s # CSP2 ALKENE 

                    if any(c[1]==2 for c in atom.connection): #if none of the above, but there is a double bond:
                        return 2,s # CSP2 ALKENE 
                    if any(c[1]==1.5 for c in atom.connection): #if none of the above, but there is a double bond:
                        return 202,s # CSP2 ALKENE 
                    else:
                        return 29,s   # RADICAL 
                
                if len(atom.connection)==4: #sp3 carbon atom
                    for r in self.rings(atom_number):
                        if len(r)==3:
                            return 22,s #CYCLOPROPANE
                        if len(r)==4: 
                            return 56,s #CSP3 CYCLOBUTANE
                    return 1,s # "CSP3 ALKANE"

            # O atoms:
            if atom.symbol=="O":

                if len(atom.connection)==1: # carbonyl,etc
                    c=atom.connection[0] #no need for a loop, there is just one atom connected

                    if c[2]=="C": # bound to C
                        atom2=self.atom(c[0])
                        atom2_number_of_oxygen_substituents= sum(s.count("O") for s in atom2.connection)
                        atom2_number_of_nitrogen_substituents= sum(s.count("N") for s in atom2.connection)
                        atom2_number_of_carbon_substituents= sum(s.count("C") for s in atom2.connection)
                        atom2_number_of_hydrogen_substituents= sum(s.count("H") for s in atom2.connection)

                        if len(atom2.connection)==3: # if the carbon atom bound to the O is trigonal
                            if atom2_number_of_carbon_substituents==2: # if it is only bound to C (excepting the initial O atom)
                                for c2 in atom2.connection:
                                    if c2[0]!=atom.atom_number: #exclude the initial O atom
                                        atom3=self.atom(c2[0])
                                        if any(c3[2]=="O" and c3[1]>1 for c3 in atom3.connection): #if there is another carbonyl
                                            for c2p in atom2.connection: # search to the other side of the carbon atom bound to the O
                                                if c2p[0]!=atom.atom_number and c2p[0]!=c2[0]: # exclude the initial O atom and the substituent to the other side
                                                    atom3p=self.atom(c2p[0])
                                                    if any(c3p[2]=="O" and c3p[1]>1 for c3p in atom3.connection): #if there is also a carbonyl on this side
                                                        return 115,s #"O=C(C=O)(C=O)" #no pibond for this type, no possibility of delocalization
                            
                            if atom2_number_of_carbon_substituents==1 and atom2_number_of_oxygen_substituents==2: # if there are two O (one is the initial O) and one C
                                for c2 in atom2.connection:
                                    if c2[0]!=atom.atom_number and c2[2]=="C": #exclude the initial O atom; let's start with the side with the C atom substituent
                                        atom3=self.atom(c2[0])
                                        if any(c3[2]=="O" and c3[1]>1 for c3 in atom3.connection): #if there is another carbonyl
                                            for c2p in atom2.connection: # search to the other side of the carbon atom bound to the O; we already know this is an O
                                                if c2p[0]!=atom.atom_number and c2p[0]!=atom3.atom_number: # exclude the initial O atom and the other substituent
                                                    atom3p=self.atom(c2p[0])
                                                    if any (c3p[2]=="H" for c3p in atom3p.connection): #if the atom3p is bound to a H atom
                                                        return 116,s #"O=C(C=O)(OH)"  #no pibond for this type, no possibility of delocalization
                                                    if all (c3p[2]=="C" for c3p in atom3p.connection): #if the atom3p is bound to a C atom
                                                        for c3p in atom3p.connection:
                                                            if c3p[0]!=atom2.atom_number: #not for the C atom bound to the initial O atom
                                                                atom4p=self.atom(c3p[0])
                                                                if any(c4p[2]=="O" and c4p[1]>1 for cp4 in atom4p.connection):
                                                                    return 121,s #"O=C(C=O)(O-C=O)"   #no pibond for this type, no possibility of delocalization
                                                        return 117,s #"O=C(C=O)(OC)"  #no pibond for this type, no possibility of delocalization
                            
                            if atom2_number_of_carbon_substituents==1 and atom2_number_of_oxygen_substituents==1: # if there are one O (the initial O) and one C
                                for c2 in atom2.connection:
                                    if c2[0]!=atom.atom_number and c2[2]=="C": #exclude the initial O atom; let's start with the side with the C atom substituent
                                        atom3=self.atom(c2[0])
                                        if any(c3[2]=="O" and c3[1]>1 for c3 in atom3.connection): #if there is another carbonyl
                                            for c2p in atom2.connection: # search to the other side of the carbon atom bound to the O
                                                if c2p[0]!=atom.atom_number and c2p[0]!=atom3.atom_number:
                                                    if c2p[2]=="N":
                                                        return 118,s #"O=C(C=O)(N<)"  #no pibond for this type, no possibility of delocalization
                                                    if c2p[2]=="F" or c2p[2]=="Cl" or c2p[2]=="Br" or c2p[2]=="I":
                                                        return 119,s #"O=C(C=O)(X)"  #no pibond for this type, no possibility of delocalization
                                                    if c2p[2]=="C":
                                                        atom3p=self.atom(c2p[2])
                                                        if any (c3p[2]=="C" and c3p[2]==2 for c3p in atom3p.connection):
                                                            return 120,s #"O=C(C=O)(C=C<)"
                                                        if any (c3p[2]=="C" and c3p[2]==1.5 for c3p in atom3p.connection):
                                                            return 320,s #"O=C(C=O)(C=C<)"     delocalized                                             
                                        if any(c3[2]=="C"and c3[1]>1 for c3 in atom3.connection): # if there is a C=C bond
                                            for c2p in atom2.connection: # search to the other side of the carbon atom bound to the O, we already know this is an O
                                                if c2p[0]!=atom.atom_number and c2p[0]!=atom3.atom_number: # exclude the initial O atom and the C=C
                                                    if c2p[2]=="O":
                                                        atom3p=self.atom(c2p[0])
                                                        if any(c3p[2]=="H" for c3p in atom3p.connection):
                                                            return 87,s #"O=C(OH)(C=C<)"
                                                        if all (c3p[2]=="C" for c3p in atom3p.connection):
                                                            atom4p=self.atom(c3p[0])
                                                            if any (c4p[2]=="O" and c4p[1]>1 for c4p in atom4p.connection):
                                                                return 102,s #"O=C(C=C)(O-C=O)"                                                    
                                                            return 92,s #"O=C(O-C)(C=C<)"
                                                        if any(c3p[2]=="N" for c3p in atom3p.connection):
                                                            return 96,s #"O=C(N<)(C=C<)"
                                                        if any(c3p[2]=="F" or c3p[2]=="Cl" or  c3p[2]=="Br" or c3p[2]=="I" for c3p in atom3p.connection ):
                                                            return 99,s # "O=C(X)(C=C<)"
                                                        if any(c3p[2]=="C" for c3p in atom3p.connection):
                                                            atom4p=self.atom(c3p[0])
                                                            if any (c4p[2]=="C" and c4p[1]==2 for c4p in atom4p.connection):
                                                                return 101,s #"O=C(C=C)(C=C)"
                                                            if any (c4p[2]=="C" and c4p[1]==1.5 for c4p in atom4p.connection):
                                                                return 301,s #"O=C(C=C)(C=C)"   #delocalized
                                            #if everything else failed since the identification of the C=C bond:
                                            if c[1]==2: 
                                                return 81  #O=C-C=C< 
                                            if c[1]==1.5:
                                                return 281  #O=C-C=C<  #delocalized

                            if atom2_number_of_nitrogen_substituents==2:
                                return 94,s #"O=C(N<)(N<)" #no pibond for this type, no possibility of delocalization
                            
                            if atom2_number_of_nitrogen_substituents==1 and atom2_number_of_oxygen_substituents==1:
                                for c2 in atom2.connection:
                                    if c2[0]!= atom.atom_number and c2[2]=="O": # continue checking on the side of the O atom
                                        atom3=self.atom(c2[0])
                                        if any(c3[2]=="H" for c3 in atom3.connection):
                                            return 85,s #"O=C(OH)(N<)"
                                        if all(c3[2]=="C" for c3 in atom3.connection):
                                            atom4=self.atom(c3[0])
                                            if any (c4[2]=="O" and c4[1]>1 for c4 in atom4.connection):
                                                return 97,s #"O=C(N<)(O-C=O)"
                                            return 90,s #"O=C(O-C)(N<)""
                            if atom2_number_of_nitrogen_substituents==1: # in case all the above fail, search for a more generic ase
                                if any(c2[2]=="F" or c2[2]=="Cl" or c2[2]=="Br" or c2[2]=="I" for c2 in atom2.connection):
                                    return 95,s #"O=C(N<)(X)"
                                return 79,s #"O=C-N< (AMIDE)"

                            if atom2_number_of_carbon_substituents>0:  # in case all the above fail, search for a more generic case
                                for c2p in atom2.connection:
                                    if c2p[0]!=atom.atom_number and c2p[2]=="C": # exclude the initial atom and any other atom that is not a C
                                        atom3=self.atom(c2p[0])
                                        if any (c3p[2]=="O" and c3p[1]>1 for c3p in atom3.connection): # if there is another carbonyl on any of the two sides
                                            if c[1]==2:
                                                return 76,s  #O=C-C=O
                                            if c[1]==1.5:
                                                return 276,s #O=C-C=O, delocalized


                            if atom2_number_of_carbon_substituents+atom2_number_of_oxygen_substituents+atom2_number_of_nitrogen_substituents<2:
                                if all (c2[2]=="F" or c2[2]=="Cl" or c2[2]=="Br" or c2[2]=="I" for c2 in atom2.connection):
                                    return 98,s #"O=C(X)(X)"
                                if any (c2[2]=="F" or c2[2]=="Cl" or c2[2]=="Br" or c2[2]=="I" for c2 in atom2.connection):
                                    if c[1]==2:
                                        return 80,s #"O=C-X (HALIDE)"
                                    if c[1]==1.5:
                                        return 280,s #"O=C-X (HALIDE)" delocalized

                            # the most generic case:
                            if atom2_number_of_carbon_substituents!=2:
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 7 (O=C CARBONYL);"
                                s=s+" however,  one of the C substituents is not a C atom. "
                                problems.append(s)
                                if c[1]==2:
                                    return 7,s    #     "O=C CARBONYL"
                                if c[1]==1.5:
                                    return 207,s #     "O=C CARBONYL", delocalized
                            
                            if c[1]==2:
                                return 7,s    #     "O=C CARBONYL"
                            if c[1]==1.5:
                                return 207,s    #     "O=C CARBONYL", delocalized

                        if len(atom.connection)==2: #KETENE O=C=
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 7 (O=C CARBONYL);"
                            s=s+" however,  it is the O atom in a Ketene (there are no specific parameters for ketene C)."
                            problems.append(s)
                            if c[1]==2:
                                return 7,s    #     "O=C CARBONYL"
                            if c[1]==1.5:
                                return 207,s    #     "O=C CARBONYL", delocalized

                        # if everything has failed, write a carbonyl anyway... 
                        if c[1]==2:
                            return 7,s    #     "O=C CARBONYL"
                        if c[1]==1.5:
                            return 207,s    #     "O=C CARBONYL", delocalized

                    if c[2]=="N": # bound to N:
                        if c[1]==2:
                            return 69,s #"AMINE OXIDE OXYGEN"
                        if c[1]==1.5:
                            return 269,s #"AMINE OXIDE OXYGEN", delocalized

                    if c[2]=="P": # bound to P: use the same than the carbonyl:
                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 7 (O=C CARBONYL);"
                        s=s+" however,  it is the O atom in a phosphate (there are no specific parameters for P=O)."
                        problems.append(s)
                        if c[1]==2:
                            return 7,s #"O-P=O PHOSPHATE" 
                        if c[1]==1.5:
                            return 207,s #"O-P=O PHOSPHATE" 

                    if c[2]=="S": # bound to S, use the same than the carbonyl:
                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 7 (O=C CARBONYL);"
                        s=s+" however,  it is the O atom in a sulophone or sulphoxide (there are no specific parameters for S=O)."
                        problems.append(s)
                        return 7,s #  "O=C CARBONYL"  

                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 7 (O=C CARBONYL);"
                    s=s+" it is X=O where  X is not C, S, P, N."
                    problems.append(s)            
                    return 7,s #  "O=C CARBONYL" default, unsafe to define it delocalized

                if len(atom.connection)==2: #ether, alcohol, etc...
                    atom_number_of_hydrogen_substituents= sum(s.count("H") for s in atom.connection)
                    atom_number_of_nitrogen_substituents= sum(s.count("N") for s in atom.connection)
                    atom_number_of_carbon_substituents= sum(s.count("C") for s in atom.connection)

                    if any ((c[1]>1 and c[2]=="C") for c in atom.connection):
                        return 70,s #"KETONIUM OXYGEN"
                    if any (c[2]=="P" for c in atom.connection):
                        return 159,s # "O-P=O PHOSPHATE"
                    if atom_number_of_hydrogen_substituents==1 and atom_number_of_nitrogen_substituents==1:
                        return 145,s # ">N-OH HYDROXYAMINE"

                    if atom_number_of_carbon_substituents==2:
                        for r in self.rings(atom.atom_number):
                            if len(r)==3:
                                return 49,s #"EPOXY"
                        atom2=self.atom(atom.connection[0][0])
                        atom2p=self.atom(atom.connection[1][0])
                        if len(atom2.connection)==3 and len(atom2p.connection)==3:
                            atom2_number_of_oxygen_substituents= sum(s.count("O") for s in atom2.connection)
                            atom2p_number_of_oxygen_substituents= sum(s.count("O") for s in atom2p.connection)
                            if atom2_number_of_oxygen_substituents>1 and atom2p_number_of_oxygen_substituents>1:
                                if any(c2[2]=="O" and c2[0]!=atom.atom_number and c2[1]==2 for c2 in atom2.connection):
                                    if any(c2p[2]=="O" and c2p[0]!=atom.atom_number and c2p[1]==2 for c2p in atom2p.connection):
                                        return 148,s #"-O- ANHYDRIDE (LOC)"
                                if any(c2[2]=="O" and c2[0]!=atom.atom_number and c2[1]>1 for c2 in atom2.connection):
                                    if any(c2p[2]=="O" and c2p[0]!=atom.atom_number and c2p[1]>1 for c2p in atom2p.connection):
                                        return 149,s #"-O- ANHYDRIDE (DELO)"                        
                            for r in self.rings( atom.atom_number):
                                if len(r)==5:
                                    if any(c2[2]=="C" and c2[1]>1 for c2 in atom2.connection):
                                        if any(c2p[2]=="C" and c2p[1]>1 for c2p in atom2p.connection):
                                            return 41,s #"OSP2 FURAN"
                        return 6,s #"C-O-H, C-O-C, O-O"

                    if atom_number_of_hydrogen_substituents==1 and atom_number_of_carbon_substituents==1:
                        for c in atom.connection:
                            if c[2]=="C": # for the single carbon substituent:
                                atom2=self.atom(c[0])
                                if len(atom2.connection)==3:
                                    if sum(c2.count("O")  for c2 in atom2.connection)==2 :
                                        return 75,s #"O-H, O-C (CARBOXYL)"
            
                    #if all of the above failed_
                    return 6,s #"C-O-H, C-O-C, O-O"
                        
            # N atoms
            if atom.symbol=="N":
                
                number_of_carbon_substituents= sum(s.count("S") for s in atom.connection)
                number_of_oxygen_substituents= sum(s.count("O") for s in atom.connection)
                number_of_nitrogen_substituents= sum(s.count("N") for s in atom.connection)
                number_of_hydrogen_substituents= sum(s.count("N") for s in atom.connection)
                number_of_sulfur_substituents= sum(s.count("S") for s in atom.connection)


                if len(atom.connection)==1:
                    return 10,s  #"NSP"

                if len (atom.connection)==2:

                    if number_of_nitrogen_substituents==2:
                        return 45,s # "AZIDE (CENTER N)"
                    
                    if number_of_nitrogen_substituents==1 and number_of_carbon_substituents==1:                
                        if any(c[2]=="N" and c[1]=="2" for c in atom.connection):
                            if any(c[2]=="C" and c[1]==1 for c in atom.connection):
                                return 107,s    # "-N=N- AZO (LOCAL)"
                            if any(c[2]=="C" and c[1]>1 for c in atom.connection):
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 107 (C-N=N-C AZO (LOCAL));"
                                s=s+" however,  bond order between N and C is larger than 1"
                                problems.append(s)
                                return 107,s    # "-N=N- AZO (LOCAL)"
                        if any(c2[2]=="N" and c[1]>1 for c in atom.connection):
                            for c2 in atom.connection:
                                if c2[2]=="N" and c2[1]==2:
                                    atom2=self.atom(c2[0])
                                    if any(c3[2]=="O" for c3 in atom2.connection):
                                        if any(c3[2]=="C" for c3 in atom2.connection):
                                            return 109,s  # "-N= AZOXY (LOCAL)"
                                if c2[2]=="N" and c2[1]>1:
                                    atom2=self.atom(c2[0])
                                    if any(c3[2]=="O" for c3 in atom2.connection):
                                        if any(c3[2]=="C" for c3 in atom2.connection):
                                            if any(c[1]==2 for c in atom.connection):
                                                return 144,s  # "-N= AZOXY (DELOC)"
                                            if any(c[1]==1.5 for c in atom.connection):
                                                return 344,s  # "-N= AZOXY (DELOC)"

                
                    if number_of_oxygen_substituents==1 and number_of_carbon_substituents==1:
                        for c2 in atom.connection:
                            if c2[2]=="O":
                                atom2=self.atom(c2[0])
                                if len(atom2.connection)==2:
                                    if any (c3[2]=="H" for c3 in atom2.connection):
                                        for c2p in atom.connection:
                                            if c2p[2]=="C" and c2p[1]>1:
                                                return 108 #"=N-OH OXIME"
                                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 108 (=N-OH OXIME);"
                                            s=s+" however,  bond order between N and C is not larger than 1"
                                            problems.append(s)
                                            return 108,s #"=N-OH OXIME"
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 108 (=N-OH OXIME);"
                                    s=s+" however,  O atom is not bound to H "
                                    problems.append(s)
                                    return 108,s #"=N-OH OXIME"
                                if len(atom2.connection)==1:
                                    s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 143 (=N-O AXOXY (DELOC));"
                                    s=s+" but it is not clear what it means "
                                    problems.append(s)
                                    if any(c[1]==2 for c in atom.connection):
                                        return 143,s #"=N-O AXOXY (DELOC)" ???
                                    if any(c[1]==1.5 for c in atom.connection):
                                        return 343,s #"=N-O AXOXY (DELOC)" ???

                    if number_of_oxygen_substituents==0: 
                        if any(c2[2]=="C" and c2[1]==2 for c2 in atom.connection):
                            if any(c2p[2]=="C" and c2p[1]==1 for c2p in atom.connection):
                                return 72,s #"=N- IMINE (LOCALZD)"
                            if any(c2p[2]=="H" and c2p[1]==1 for c2p in atom.connection):
                                return 72,s #"=N- IMINE (LOCALZD)" 

                        if any(c2[2]=="C" and c2[1]==1.5 for c2 in atom.connection):
                            if all (c2[2]=="C" for c2 in atom.connection):
                                for r in self.rings(atom.atom_number):
                                    if len(r)==6:
                                        return 237,s # "-N=C-/PYR (DELOCLZD)"
                                s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 37 (-N=C-/PYR (DELOCLZD);"
                                s=s+" but it is not in a pyridine "
                                problems.append(s)
                                return 237,s # "-N=C-/PYR (DELOCLZD)"
                    # if everything failed:
                    return 72,s #"=N- IMINE (LOCALZD)" 


                if len(atom.connection)==3:
                    
                    if number_of_oxygen_substituents==2:
                        return 46,s    #     "NITRO"
                    
                    if number_of_oxygen_substituents==1:
                        for c in atom.connection:
                            if c[2]=="O":
                                atom2=self.atom(c[0])
                                if any(c2[2]=="H" for c2 in atom2.connection):
                                    return 146,s  #">N-OH HYDROXYAMINE"
                    
                    if number_of_oxygen_substituents==1 and number_of_nitrogen_substituents==1:
                        for c in atom.connection:
                            if c[2]=="N":
                                if c[1]>1:
                                    atom2=self.atom(c[0])
                                    if len(atom2.connection)==2:
                                        for cp in atom.connection:
                                            if cp[2]=="O":
                                                atom2p=self.atom(cp[0])
                                                if len(atom2p.connection)==1:
                                                    if any(c[1]==2 for c in atom.connection):
                                                        return 43,s    #     "=N-O AZOXY (LOCAL)"
                                                    if any(c[1]==1.5 for c in atom.connection):
                                                        return 243,s    #     "=N-O AZOXY (LOCAL)" delocalized

                    if number_of_sulfur_substituents==1:
                        for c in atom.connection:
                            if c[2]=="S":
                                atom2=self.atom(c[0])
                                if sum(s.count("O") for s in atom2.connection)==2:
                                    return 155,s    #     "NSP3 SULFONAMIDE"
                        s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 155 (NSP3 SULFONAMIDE);"
                        s=s+" only because N is bound to S, but it does not look a sulfonamide"
                        problems.append(s)
                        return 155,s    #     "NSP3 SULFONAMIDE"
                    
                    if sum(s.count("Li") for s in atom.connection)==1:
                        return 164,s    #     "NSP3 LI-AMIDE"

                    if number_of_nitrogen_substituents==1:
                        for c in atom.connection:
                            if c[2]=="N":
                                atom2=self.atom(c[0])
                                if len(atom2.connection)>2:
                                    return 150,s    #     "NSP3 HYDRAZINE"
                        
                        if number_of_carbon_substituents>1: 
                            if any(c[1]>1 for c in atom.connection): #also check that it is not an amide
                                for c in atom.connection:
                                    if c[2]=="C": #only check in C atoms
                                        atom2=self.atom(c[0])
                                        if any(c2[2]=="O" and c2[1]==2 for c2 in atom2.connection): #if there is a carbonyl, it correspond to either an amide or urea
                                            return 9,s #"NSP2 AMIDE "
                                        if any(c2[2]=="O" and c2[1]>1 for c2 in atom2.connection): 
                                            return 151,s #"NSP2 AMIDE (DELOC)"  maybe this should also be 9 
                                    #if no carbonyl found:                            
                                    for r in self.rings( atom.atom_number):
                                        if len(r)==6:
                                                return 311,s    #    "-N(+)= PYRIDINIUM" delocalized
                                        if len(r)==5:
                                                return 240,s    #     "NSP2 PYRROLE" delocalized
                                    return 110    #    "-N(+)= IMMINIUM"
                                    
                    # in the rest of the cases:
                    return  8,s    #     "NSP3"
                            
                if len(atom.connection)==4:
                    return 39,s    #    "NSP3 AMMONIUM"
                
                #where do we have to return: 9    N     "NSP2"  

            if atom.symbol=="S":

                number_of_carbon_substituents= sum(s.count("C") for s in atom.connection)
                number_of_sulfur_substituents= sum(s.count("S") for s in atom.connection)
                number_of_oxygen_substituents= sum(s.count("O") for s in atom.connection)
                number_of_nitrogen_substituents= sum(s.count("N") for s in atom.connection)
                number_of_hydrogen_substituents= sum(s.count("N") for s in atom.connection)


                if len(atom.connection)==1:
                    return 74,s    #     "S=C"
                
                if len(atom.connection)==2:
                    if number_of_sulfur_substituents==2:
                        return 105,s    #     "-S- POLYSULFIDE"
                    if number_of_sulfur_substituents==1:
                        return 104,s    #     "-S-S- DISULFIDE"
                    if number_of_carbon_substituents==2:
                        atom2=self.atom(atom.connection[0][0])
                        atom2p=self.atom(atom.connection[1][0])
                        if len(atom2.connection)==3 and len(atom2p.connection)==3:
                            for r in self.rings( atom_number):
                                if len(r)==5:
                                    return 242,s    #     "SSP2 THIOPHENE", delocalized
                        if atom_charge==1:
                            return 16,s    #    ">S+ SULFONIUM" 
                        return 15,s #     "-S- SULFIDE"

                    if number_of_carbon_substituents==1 and number_of_hydrogen_substituents==1:
                        if atom_charge==1:
                            return 16,s    #    ">S+ SULFONIUM" 
                        return 15,s #     "-S- SULFIDE"
                
                if len(atom.connection)==3:
                    if number_of_oxygen_substituents==1 and number_of_carbon_substituents==2:
                        return 17,s   #S     ">S=O SULFOXIDE"
                    if number_of_oxygen_substituents==1 and number_of_carbon_substituents==1:
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 17 (>S=O SULFOXIDE);"
                            s=s+" however, only one of the S substituents is a C atom. "
                            problems.append(s)
                            return 17,s   #S     ">S=O SULFOXIDE"               
                
                if len(atom.connection)==4:
                    if number_of_oxygen_substituents==2 and number_of_carbon_substituents==2:
                        return 18,s    #     ">SO2 SULFONE"
                    if number_of_oxygen_substituents==2 and number_of_nitrogen_substituents==1 and number_of_carbon_substituents==1:
                        return 154,s    #     ">SO2 SULFONAMIDE"
                    if number_of_oxygen_substituents==2:
                            s= "Warning: atom_type for atom: "+str(atom.atom_number)+" assigned 18 (>SO2 SULFONE);"
                            s=s+" however, at least of one of the S substituents is neither C atom nor N. "
                            problems.append(s)
                            return 17,s   #S     ">SO2 SULFONE" 

            if atom.symbol=="P":

                if len(atom.connection)==3:
                    return 25,s    #     ">P- PHOSPHINE"
                if len(atom.connection)==4:
                    if any (c[2]=="O" and c[1]>1 for c in atom.connection):
                        if any(c[1]==2 for c in atom.connection):
                            return 153,s    #     ">P=O PHOSPHATE"
                        if any(c[1]==1.5 for c in atom.connection):
                            return 153,s    #     ">P=O PHOSPHATE"
                    return 60,s    #     "PHOSPHORUS (V)"

            # easier cases:
            if atom.symbol=="F": return 11,s    #     "FLUORIDE"
            if atom.symbol=="Cl": return 12,s   #    "CHLORIDE"
            if atom.symbol=="Br": return 13,s   #    "BROMIDE"
            if atom.symbol=="I": return 14,s    #     "IODIDE" 
            if atom.symbol=="Si": return 19,s   #    "SILANE"
            if atom.symbol=="B":
                if len(atom.connection)==3: return 26,s    #     ">B- TRIGONAL"
                else:return 27,s    #     ">B< TETRAHEDRAL"  
            if atom.symbol=="Ge": return 31,s    #    "GERMANIUM"
            if atom.symbol=="Sn": return 32,s    #    "TIN"
            if atom.symbol=="Pb": return 33,s    #    "LEAD (IV)"
            if atom.symbol=="Se": return 34,s    #    "SELENIUM"
            if atom.symbol=="Te": return 35,s    #    "TELLURIUM"
            if atom.symbol=="He": return 51,s   #    "HELIUM"
            if atom.symbol=="Ne": return 52,s    #    "NEON"
            if atom.symbol=="Ar": return 53,s    #    "ARGON"
            if atom.symbol=="Kr": return 54,s    #    "KRYPTON" 
            if atom.symbol=="Xe": return 55,s    #    "XENON"
            if atom.symbol=="Mg": return 59,s    #    "MAGNESIUM"
            if atom.symbol=="Fe":
                if atom.charge==3.0:
                    return 62,s    #    "IRON (III)"
                else: return 61,s    #    "IRON (II)"
            if atom.symbol=="Ni":
                if atom.charge==3.0:
                    return 64,s    #    "NICKEL (III)"
                else: return 63,s    #    "NICKEL (II)"
            if atom.symbol=="Co":
                if atom.charge==3.0:
                    return 66,s    #    "COBALT (III)"
                else: return 65,s    #    "COBALT (II)"
            if atom.symbol=="Ca": return 125,s   #    "CALCIUM"
            if atom.symbol=="Sr": return 126,s   #    "STRONTIUM"
            if atom.symbol=="Ba": return 127,s   #    "BARIUM"
            if atom.symbol=="La": return 128,s   #    "LANTHANUM"
            if atom.symbol=="Ce": return 129,s   #    "CERIUM"
            if atom.symbol=="Pr": return 130,s   #    "PRAESEODYMIUM"
            if atom.symbol=="Nd": return 131,s   #    "NEODYMIUM"
            if atom.symbol=="Pm": return 132,s   #    "PROMETHIUM"
            if atom.symbol=="Sm": return 133,s   #    "SAMARIUM"
            if atom.symbol=="Eu": return 134,s   #    "EUROPIUM"
            if atom.symbol=="Gd": return 135,s   #    "GADOLINIUM"
            if atom.symbol=="Tb": return 136,s   #    "TERBIUM"
            if atom.symbol=="Dy": return 137,s   #    "DYSPROSIUM"
            if atom.symbol=="Ho": return 138,s   #    "HOLMIUM"
            if atom.symbol=="Er": return 139,s   #    "ERBIUM"
            if atom.symbol=="Tm": return 140,s   #    "THULIUM"
            if atom.symbol=="Yb": return 141,s   #    "YTTERBIUM"
            if atom.symbol=="Lu": return 142,s   #    "LUTETIUM"

            # if everything fails...

            if interactive:
                print ("Not able to find atom type for atom: ")
                print (atom.print_atom())
            if atom.atom_type!="" and type(atom.atom_type)==type(1):
                if interactive: print ("it has been assigned to: "+atom.atom_type +" based on the information in the input file")
                return int(atom.atom_type), "unable to assign atom type, copied from the input file"

            if MM3_FF:    
                suggestions=""
                possible_types=[]
                for b in MM3_FF.list_of_Atom_types:
                    if b.symbol==atom.symbol:
                        suggestions+="type:"+str(b.a_type)+" for: "+b.symbol+"  "+b.descr+ " (coordination:"+str(b.coord)+")\n"
                        possible_types.append(b.a_type)
                if interacive: 
                    print ("Suggestions:\n"+suggestions)
        
                    a_type= raw_input ("Manually assign atom type: ")
                    if a_type.isdigit(): 
                        a_type=int(a_type)
                    else: a_type=0
                    s="warning... manually assigned atom type"
                else: 
                    a_type=possible_types[0]
                    s="unable to assign atom type, using the first atom type for the element based on MM3_FF used"
            else:
                a_type=0
                s="unable to assign atom type"

                
            return a_type,s 


        #returns type for H atom bound to atom number: "bound"; only works with H atoms. Needed for setting-up interface between HL and LL layers
        def get_H_MM3type(self,bound): 
            B=self.atom(bound)
            atom_bound_type=5
            if B.atom_type in ["6","145","159"]: atom_bound_type="21"
            if B.atom_type in ["8","37","40","43","72","107","109","110","111","146","150","155"]: atom_bound_type="23"   
            if B.atom_type in ["75"]: atom_bound_type="24"
            if B.atom_type in ["9"]: atom_bound_type="28"
            if B.atom_type in ["15","42"]: atom_bound_type="44"
            if B.atom_type in ["39"]: atom_bound_type="48"
            if B.atom_type in ["41"]: atom_bound_type="73"
            if B.atom_type in ["4"]: atom_bound_type="124"
            return atom_bound_type

        # includes in a MM3 prm file the information needed for configuring pi-system (the script used for setting up external calculation should take care of moving this information
        # from the ".prm" file to the ".key" file )
        def print_MM3_pi_system(self,prm_file_name=""):

            if prm_file_name=="":
                prm_file_name=self.input_file.rsplit(".",1)+"prm"

            prm_file=open(prm_file_name,"r+")
            prm_lines=prm_file.readlines()
            prm_file.seek(0)
            #reset the pisystem if already in file

            i=0
            while i<len(prm_lines):
                if len(prm_lines)>i+3:
                    if "Information of the MM3 PI-SYSTEMS" in prm_lines[i+3]  and not prm_lines[i].startswith("#PISYSTEM"): 
                        prm_file.write(prm_lines[i])
                    elif prm_lines[i].startswith("#PISYSTEM"): i+=1
                    else:
                        i+=6
                        while (i+1<len(prm_lines) or prm_lines[i].startswith("      ########") ): 
                            i+=1
                
                else:
                    prm_file.write(prm_lines[i])            
                i+=1

            prm_file.truncate()
            prm_file.close()

            text="##    No need of specifying PI-SYSTEMS           \n"            
            if len(self.MM3_pi_systems)>0:
                text= "\n      ####################################################\n"
                text+="      ##                                                ##\n"
                text+="      ##    Information of the MM3 PI-SYSTEMS           ##\n"
                text+="      ##           (do not un-comment)                  ##\n"
                text+="      ####################################################\n"
                for pi_system in self.MM3_pi_systems:
                    t=[]
                    for p in pi_system:
                            t.append(str(p.atom_number))
                    text+="\n#PISYSTEM "+" ".join(t)
            f=open(prm_file_name,"a")
            f.write(text)
            f.write("\n")
            f.close()            

        # uses the information of this object to create a tinker xyz file and a key file
        def print_MM3tinker_file(self,tinker_file_name="",prm_file_name="",key_file_name=""):
            #txyz file
            natoms=len(self.atom_list)

            s="%6d " %(natoms)+"   test of atom types with tinker\n" 
            for i in range(1,natoms+1):
                con=""
                Atom=self.atom(i)
                if int(Atom.atom_type)>200: atom_type=str(int(Atom.atom_type)-200)
                else: atom_type=Atom.atom_type
                s=s+"%6d  %-3s %12.6f %12.6f %12.6f %6d"%(i,Atom.symbol,Atom.coord[0],Atom.coord[1],Atom.coord[2],int(atom_type))
                for c in Atom.connection: con=con+"%6d"%(c[0])
                s=s+con+"\n"
            if tinker_file_name=="":
                tinker_file_name= self.input_file.rsplit(".",1)[0]+".mm3.tinker.xyz"
            if key_file_name=="":
                key_file_name= tinker_file_name.split(".xyz")[0]+".key"
            file=open(tinker_file_name,"w")
            file.write(s)
            file.close()

            #key file
            if prm_file_name=="":
                if 'TINKER_PARAMS' in os.environ.keys():
                    key="# Force Field Selection\n PARAMETERS        "+str(os.environ['TINKER_PARAMS']+"mm3.prm")
                else: key="# Force Field Selection\n PARAMETERS    ***write here the location of mm3.prm file***"
            else:
                key="# Force Field Selection\n PARAMETERS        "+prm_file_name

            if len(self.MM3_pi_systems)>0:
                for pi_system in self.MM3_pi_systems:
                    t=[]
                    for p in pi_system:
                            t.append(str(p.atom_number))
                    key=key+"\nPISYSTEM "+" ".join(t)

            #uncomment this for debugging the tinker calculation
            #key+="\nDEBUG\n"

            file=open(key_file_name,"w")
            file.write(key)
            file.close()
            return s
        #updates the atom_types using a list of atom_types
        #can not be called in the constructor because it needs list_of_atom_types

        #update the lists of Atoms for every pi_system
        #can not be called in the constructor of the class because it needs the atom types        
        def update_MM3_pi_systems(self):
            self.MM3_pi_systems=[]            
            for Atm in self.atom_list:
                if int(Atm.atom_type)>200:

                    if (len(self.MM3_pi_systems)==0) or (not any(Atm in p for p in self.MM3_pi_systems)): #if not already in one of the MM3_pi_systems
                        pi_system=[Atm]  #create new pi_system and fill it 
                        updated=True           
                        while updated: #continue until no more elements can be added to pi_system
                            updated=False
                            for i in range (0,len(pi_system)):
                                for c in pi_system[i].connection:
                                    if self.atom_list[c[0]-1] not in pi_system and int(self.atom_list[c[0]-1].atom_type)>200:
                                        pi_system.append(self.atom_list[c[0]-1])
                                        updated=True
                        self.MM3_pi_systems.append(pi_system)


        # METHODS RELATED TO FINDING A FINGERPRINT OF EACH ATOM THAT IS EXCLUSIVE AND SHARED BETWEEN ANALOGUE ATOMS OF SIMILAR MOLECULES
        # initialize an array to include iteratively characteristics of the atom
        # it contain lists with the format: [nC,nN,nO,nS,nP,nX,stereo] where: nH: number of H; nC: number of C;... nX: number of any other element
        # the first element is the symbol ([1,0,0,0,0,0,0] for a C atom; [0,1,0,0,0,0,0] for a O atom, [0,0,0,0,0,0,0] for H atom...)
        # the second element will be the atoms joint to this atom ([2,0,0,0,0,0,0] the atom is joint to 2 C atoms and to 2 H atoms) 
        # the third is the atoms joint to the atoms joint to this atom.... the method __extendfingerprint takes care of doing this iteratively
        # the last element of the list, stereo, will be defined later to differenciate atoms that are equivalent by the connectivity alone
        def __initializefingerprints(self):
            for atom in self.atom_list:
                if atom.symbol=="H": atom.fingerprint=[[0,0,0,0,0,0,0]]
                elif atom.symbol=="C": atom.fingerprint=[[1,0,0,0,0,0,0]]
                elif atom.symbol=="N": atom.fingerprint=[[0,1,0,0,0,0,0]]
                elif atom.symbol=="O": atom.fingerprint=[[0,0,1,0,0,0,0]]
                elif atom.symbol=="S": atom.fingerprint=[[0,0,0,1,0,0,0]]
                elif atom.symbol=="P": atom.fingerprint=[[0,0,0,0,1,0,0]]
                else:  atom.fingerprint=[[0,0,0,0,0,1,0]]

        #auxiliary method:
        #add another vector to the fingerprint of each atom, based of the last vector of the atoms connected to it
        def __extendfingerprint(self):
            i=len(self.atom_list[0].fingerprint)-1
            for atom in self.atom_list:
                newfingerprint=[]
                for k in range( 0,len(self.atom_list[0].fingerprint[0]) ):
                    newfingerprintelement=0
                    for c in atom.connection:
                        #if i>1: newfingerprintelement+= (self.atom(c[0]).fingerprint[i][k]  - atom.fingerprint[i-2][k])
                        newfingerprintelement+= self.atom(c[0]).fingerprint[i][k]
                    newfingerprint.append(newfingerprintelement)
                atom.fingerprint.append(newfingerprint)


        #auxiliary method:
        #test if the fingerprint of two atoms are (element-wise) identical
        # obsolete
        def __has_identical_fingerprint(self,atom1,atom2):
            if len(atom1.fingerprint)!=len(atom2.fingerprint):
                print ("warning: the fingerprints do not have equal sizes")
                return False
            else: return atom1.fingerprint==atom2.fingerprint
                #for fgl1,fgl2 in zip(atom1.fingerprint,atom2.fingerprint):
                    #if fgl1!=fgl2: return False
                    #for fge1,fge2 in zip(fgl1,fgl2):
                    #    if fge1!=fge2:
                    #        return False
                #return True

        #sort the atoms in sets sharing identical fingerprints 
        #when all atoms have different fingerprints every set will contain a single atom, and the length of the list of sets must be equal to the number of atoms
        def __group_of_atoms_by_fingerprints(self):
            groups=[[self.atom_list[0].atom_number]] #first group contains first atom
            for a in self.atom_list[1:]:
                if   not (a.connection==[] and a.symbol.lower()=="h") :
                    included=False
                    for g in groups:
                        if a.fingerprint==self.atom_list[ g[0]-1 ].fingerprint:
                            g.append(a.atom_number)
                            included=True
                    if not included: groups.append([a.atom_number])  #only executed if not break before
            return groups

        # extend the atoms fingerprints until no more different fingerprints are found
        def __generate_fingerprints(self):
        
            while True:
                old_fingerprints=copy.deepcopy(self.__group_of_atoms_by_fingerprints())  
                self.__extendfingerprint()
                if old_fingerprints==self.__group_of_atoms_by_fingerprints(): break 

            #remove the last element of the fingerprint... it was already converged,but it does not work in some cases!!! why?
            for atom in self.atom_list:
                #del atom.fingerprint[-1]  
                continue


        # reduce the multidimensional fingerprint to a single number: the sum of the numbers in the fingerprint multiplied by the atomic number
        # this is used to rank different atoms unambigously (for example, to use one of the other to set a reference)
        def __trace_of_fingerprints (self,atom):
            trace=0
            Z=[12,14,16,32,31,1,1]
            for fp_1 in atom.fingerprint:
                for fp_2,z in zip(fp_1,Z):
                    trace+=fp_2*z
            return trace

        # delete current fingerprints
        def __reset_fingerprints (self):
            for a in self.atom_list: a.fingerprint=[]
                
        # the first time that the fingerprints are created only connectivity information is used. Depending on symmetry, some atoms will have identical fingerprints.
        # in this method, additional information is added to differenciate between different atoms using a procedure that should give identical fingerprint for analogue atoms in conformers.
        # it explores the grous of atoms with identical fingerprints and differentiate them.
        # for pairs of atoms, the fingerprint is modified according to the numbers in the list rules (len(rules)=2)
        # for triads of atoms, it checks if these correspond to methyl (or similar groups) and the classification is done according to dihedrals
        # for sets of more than 3 atoms with identical fingerprint, the fingerprint is modified according to the numbers in the list rules.
        # the default rules will be used if non is given, so it tan be called without knowing any rule
        # then, all elements in the fingerpring are deleted (excepting the first), and the fingerprints are re-calculated including the geometrical information.
        # as a side effect of differentiating a pair (or set) of atoms with identical fingerprints, the fingerprints of connected atoms is also modified, so it is possible
        # (and desirable) that other pairs (sets) are automatically differenciated. 
        # Therefore, first a pair is solved and the loop starts again until no pairs are left. Next, the first methyl groups' H (or terbutyl groups' methyls) are differentiated
        # and the loop starts again to check if there is any pair to solve, any other methyl... finally, sets of more than 2 atoms with identical fingerprint and that are not 
        # in a methyl group are resolved exhausting all combinations. Resolving sets of more than 2 atoms is costly, and for this reason is better to exhaust first all pairs
        # (most of the times there will be no confusion after resolving all pairs and methyl substituents).
        # The method returns a "track" of the dischriminations made; this way, it can be called without rules (using the default rules)
        # and the output generated can be used to exhaustively generate all combinations  that can be used by other methods to generate all possible rules combinations
        def __refine_fingerprints(self,levels=0,rules=[]):

            counter=0
            rules_record=[]

            while True:
                if len(rules)<counter+1: rules.append([])

                if self.__resolve_pair_of_identical_fingerprints(levels,rule=rules[counter]):
                    counter+=1
                    rules_record.append(1)
                    continue  #start the loop again, this way all pairs will be resolved before starting with methyl groups

                if self.__resolve_methyl_groups(levels):
                    counter+=1
                    rules_record.append(0)
                    continue  #start the loop again, resolving any new pair of identical fingerprint generated and then starting with methyl groups
                
                size_of_fingerprint_resolved=self.__resolve_set_of_identical_fingerprints(levels,rule=rules[counter])
                if size_of_fingerprint_resolved>0:
                    counter+=1
                    rules_record.append(size_of_fingerprint_resolved)
                    continue #start the loop again, solving pairs first, next methyl groups, etc.

                break

            return rules_record

        #differentiate, according to the "rule" (or using the default rule) the pair of atoms with identical fingerprints whose distance is smaller
        def __resolve_pair_of_identical_fingerprints(self,levels=0,rule=[2,3]):

                if rule==[]:rule=[1,2]
                centroid=np.mean([a.coord for a in self.atom_list],axis=0)
                pair=[] # the pair of atoms that currently have identical fingerprint and will be differentiated 
                # the pair with the smaller distance will be selected for the next attempt.
                # if the difference in distance between two atoms is only 0.1 A larger than the smaller distance, the pair of atoms closer to the molecule centroid will be employed  
                min_distance=999.0  
                len_group_of_atoms_by_fingerprints_before=len(self.__group_of_atoms_by_fingerprints())
                for g in self.__group_of_atoms_by_fingerprints():                   
                    # will only consider pairs of identical fingerprints that are not unbound H atoms
                    # cases of 3 or more atoms whit identical fingerprints are treated in other methods
                    if len(g)==2 and not (self.atom(g[0]).connection==[] and self.atom(g[0]).symbol.lower()=="h"): 
                        dist=self.distance(g)
                        if pair!=[] and abs(dist-min_distance)<0.1: 
                            new_pair_center =(self.atom(pair[0]).coord+self.atom(pair[1]).coord)*0.5
                            curr_pair_center=(self.atom(g[0]).coord+self.atom(g[1]).coord)*0.5
                            dist_new_centroid= np.sum( (new_pair_center-centroid)**2)
                            dist_curr_centroid=np.sum( (curr_pair_center-centroid)**2)
                            if dist_new_centroid<dist_curr_centroid:
                                min_distance=dist
                                pair=g
                        elif dist<min_distance:
                            min_distance=dist
                            pair=g

                #changes will only be made if it was possible to identify a pair before; in that case, min_distance will be smaller than 999.0
                if min_distance<999.0:
                    #modify, according to the flag ([1,2] or [2,1]) in the rules,the last element of the fingerprint
                    self.atom(pair[0]).fingerprint[0][-1],self.atom(pair[1]).fingerprint[0][-1]=rule[0],rule[1]
                    # delete all list in the atom fingerprints except the first, so they can be recalculated 
                    for a in self.atom_list: del a.fingerprint[1:] 
                    # calculate again the fingerprints starting from the first list of each atom:
                    if levels==0: 
                        self.__generate_fingerprints()
                    else: 
                        for i in range(0,levels):self.__extendfingerprint()
                
                #return True if it was worth to do the change (the number of elements fo "gorup_of_atoms_by_fingerprints" is increased if atoms are differenciated)
                return len_group_of_atoms_by_fingerprints_before<len(self.__group_of_atoms_by_fingerprints())
    
        #differenciate three atoms with identical fingerprints bound to a common atom (for example, the 3 H of a methyl group, or the 3 C methyl atoms of a t-butyl grup)
        def __resolve_methyl_groups(self,levels=0):

                len_group_of_atoms_by_fingerprints_before=len(self.__group_of_atoms_by_fingerprints())

                for g in self.__group_of_atoms_by_fingerprints():
                    if len(g)==3:  #only search in groups of 3 atoms with identical fingeprint
                        #the three atoms are not connected between them:
                        atoms_connected_to_g=[c[0] for c in self.atom(g[0]).connection+self.atom(g[1]).connection+self.atom(g[2]).connection]

                        if not any ([gg in atoms_connected_to_g for gg in g]):
                            #the three atoms must be connected to a common atom (and this common atom is unique)
                            g_connections=[ set([a[0] for a in self.atom(gg).connection]) for gg in g ]
                            if len(set.intersection( g_connections[0], g_connections[1], g_connections[2])) == 1:                                               
                                common_bound=set.intersection( g_connections[0], g_connections[1], g_connections[2]).pop()
                                # find the atom(s) bound to the common_bound atom (excluding any of the atoms in g)
                                # usually there will only be one possible atom (valence 4), but just in case, all bound atoms not in g will be listed...
                                possible_atoms=[a[0] for a in self.atom(common_bound).connection]
                                possible_atoms=list(dict.fromkeys(possible_atoms)) #remove duplicates

                                for a in g: possible_atoms.remove(a)
                                if len(possible_atoms)<1: break # if there is nothing bound to the central atom, break and solve the identical fingerprints by other method...
                                # in case there is more than one possible atom,  the one with the largest atomic number will be used. 
                                # If there is a draw, the sum of their fingerprint elements will be used
                                # the list will contain the atomic number multiplied by a large number plus the traces, so the atomic number will prevalece and only if there is a draw
                                # the traces will be relevant 
                                possible_atoms_order=[self.atomic_numbers[self.atom(a).symbol.lower()]*100000000+self.__trace_of_fingerprints(self.atom(a)) for a in possible_atoms]
                                bound_to_common_1=possible_atoms[possible_atoms_order.index(max(possible_atoms_order))]

                                # the atom bound to bound_to_common_1 (excepting common_bound) with the largest atomic number.
                                # in case of a draw, use the largest trace of their fingerprints:
                                possible_atoms=[c[0] for c in self.atom(bound_to_common_1).connection]
                                possible_atoms=list(dict.fromkeys(possible_atoms)) #remove duplicates
                                possible_atoms.remove(common_bound)
                                if len(possible_atoms)<1: break # if there is nothing bound to the central atom, break and differentiate by other method...
                                #a list with the order of the possible atoms; it will use the atomic number and if there is a draw, the traces of the possible atoms
                                #the list will contain the atomic number multiplied by a large number plus the traces
                                possible_atoms_order=[self.atomic_numbers[self.atom(a).symbol.lower()]*100000000 + self.__trace_of_fingerprints(self.atom(a)) for a in possible_atoms]
                                bound_to_common_2=possible_atoms[possible_atoms_order.index(max(possible_atoms_order))]
                                
                                #calculate the distance between each atom in g and the plane deffined by common_bound,bound_to_common_1 and bound_to_common_2
                                plane_vector=np.cross(self.atom(common_bound).coord-self.atom(bound_to_common_1).coord,self.atom(bound_to_common_2).coord-self.atom(bound_to_common_1).coord)
                                vectors=[self.atom(gg).coord-self.atom(bound_to_common_1).coord for gg in g]
                                distances_to_plane=[abs(v.dot(plane_vector)) for v in vectors]                                                           

                                #change the first element of the fingerprint of each atom:
                                #1 for the atom closer to the plane deffined by common_bound,bound_to_common_1 and bound_to_common_2
                                #2 and 3 are assigned by the other two atoms depending of the dihedral angle with previous atom, the bound_to_common_1 atom and the common_bound and 

                                #the index of the atom of the triad that is closer to the plane 
                                index_of_nearest_atom_to_plane=distances_to_plane.index(min(distances_to_plane))
                                self.atom(g[index_of_nearest_atom_to_plane]).fingerprint[0][-1]=1
                                self.atom(g[index_of_nearest_atom_to_plane]).fingerprint[0][-1]=4
                                not_nearest_atoms=copy.deepcopy(g)
                                not_nearest_atoms.remove(g[index_of_nearest_atom_to_plane])
                                d1=self.angle([g[index_of_nearest_atom_to_plane],bound_to_common_1,common_bound,not_nearest_atoms[0]])
                                d2=self.angle([g[index_of_nearest_atom_to_plane],bound_to_common_1,common_bound,not_nearest_atoms[1]])
                                if d1>d2: 
                                    self.atom(not_nearest_atoms[0]).fingerprint[0][-1]=2
                                    self.atom(not_nearest_atoms[1]).fingerprint[0][-1]=3
                                    #self.atom(not_nearest_atoms[0]).fingerprint[0][-1]=5
                                    #self.atom(not_nearest_atoms[1]).fingerprint[0][-1]=6
                                else:
                                    self.atom(not_nearest_atoms[1]).fingerprint[0][-1]=2
                                    self.atom(not_nearest_atoms[0]).fingerprint[0][-1]=3
                                    #self.atom(not_nearest_atoms[0]).fingerprint[0][-1]=6
                                    #self.atom(not_nearest_atoms[1]).fingerprint[0][-1]=5
                                
                                # delete all list in the atom fingerprints except the first, so they can be recalculated again
                                for a in self.atom_list:
                                    del a.fingerprint[1:] 
                                # calculate again the fingerprints starting from the first list of each atom:
                                if levels==0: 
                                    self.__generate_fingerprints()
                                else: 
                                    for i in range(0,levels):self.__extendfingerprint()

                #return True if it was worth to do the change (the number of elements fo "gorup_of_atoms_by_fingerprints" is increased if atoms are differenciated)
                return len_group_of_atoms_by_fingerprints_before<len(self.__group_of_atoms_by_fingerprints())

        #differentiate, according to the "rule" (or using the default rule) a set of atoms with identical fingerprints
        def __resolve_set_of_identical_fingerprints(self,levels=0,rule=[]):

                len_group_of_atoms_by_fingerprints_before=len(self.__group_of_atoms_by_fingerprints())
                size=0
                for g in self.__group_of_atoms_by_fingerprints():
                    if len(g)>2 and not (self.atom(g[0]).connection==[] and self.atom(g[0]).symbol.lower()=="h"): 
                        size=len(g)
                        if rule==[]: rule=range(1,size+1)
                        for a,r in zip(g,rule): self.atom(a).fingerprint[0][-1]=r
                        # delete all list in the atom fingerprints except the first, so they can be recalculated 
                        for a in self.atom_list: del a.fingerprint[1:] 
                        # calculate again the fingerprints starting from the first list of each atom:
                        if levels==0: 
                            self.__generate_fingerprints()
                        else: 
                            for i in range(0,levels):self.__extendfingerprint() 
                        break 

                #return True if it was worth to do the change
                return  size

        #made public for debuggin, safe to remove
        def set_fingerprints(self,levels=0,rules=[],refine=True):self.__set_fingerprints(levels=levels,refine=refine) 

        #the function that should be called to generate fingerprints for the atoms, adding aditional terms as specified by "levels" or until there are no more differences
        # rules will be used to resolve pairs of identical fingerprints according to their order
        # returns the number of last fingerprint elements that should be set to 0 or 1 to resolve all pairs of fingerprints that are equivalent by the connectivity
        def __set_fingerprints(self,levels=0,rules=[],refine=True):
            #initialize the fingerprints of each atom
            self.__initializefingerprints()
            #generate temporary fingerprints using only connectivity information
            # if levels=0, it continues until all fingerprints are different
            if levels==0: self.__generate_fingerprints()
            # if levels!=0, will run "levels" times
            else:
                for times in range(0,levels): self.__extendfingerprint()
                    
            #add geometrical information and recalculate fingerprints    
            if refine: 
                return self.__refine_fingerprints(levels,rules)
            else: 
                return []
            

        #use the fingerprint of each atom to renumber the atoms of current molecules like in reference_molecule 
        # (if the number of atoms is different, the order is preserved, but not the atom number)
        # returns a warning string that is emtpy if everything was OK
        # it updates the connectivity and two dictionaries: one with the old atom numbers as keys and the new atom numbers as values
        # and another with the reference_molecule atom numbers as keys asn the corresponding new atom numbers as values
        def __renumber_by_fingerprints(self, reference_molecule,reference_molecule_group=[]):

            #in case reference_molecule_group is omitted, use all atoms
            if reference_molecule_group==[]:reference_molecule_group=[atom.atom_number for atom in reference_molecule.atom_list]

            #just to check that everything was ok
            warning=""

            #dictionaries to make the conversion of atoms
            #a dictionary containing old atom numbers as keys and new atom numbers as values
            old_to_new_numbers={}
            #a dictionary containing atom numbers in reference_molecule as keys and new atom numbers as values
            ref_to_new_numbers={} 
            old_to_intermediate_numbers={}
            ref_to_old_numbers={} 

            #check if fingerprints were calculated (and calculate them if not)
            if len(self.atom_list[0].fingerprint)==0:
                self.__set_fingerprints()
            if len(reference_molecule.atom_list[0].fingerprint)==0:
                reference_molecule.__set_fingerprints()

            #some useful lists:
            #the list of atoms that are not connected in curent molecule (their fingerprint is simply a set of zeros, that sum zero)
            unbound_atoms=[atom for atom in self.atom_list if sum(sum(atom.fingerprint,[]))==0 ] 
            #the list of atoms in reference molecule that will be used in the comparison (filtered by reference_molecule_group)
            ref_atoms=[atom for atom in reference_molecule.atom_list if atom.atom_number in reference_molecule_group]
            #the list of are not connected in the list ref_atoms (their fingerprint is simply a set of zeros, that sum 0)
            ref_unbound_atoms=[atom for atom in ref_atoms if sum(sum(atom.fingerprint,[]))==0 ] 
            
            for atom in self.atom_list:
                if atom not in unbound_atoms:  #only corrects the numbers of non unbound atoms
                    already=False
                    for r_atom in ref_atoms:
                        if atom.fingerprint==r_atom.fingerprint:
                            if not already:
                                ref_to_old_numbers[r_atom.atom_number]=atom.atom_number
                                atom.atom_number=r_atom.atom_number
                                already=True
                                break
                            else:
                                warning+="Warning: atom "+str(atom.atom_number)+" can be identified as atoms "+str(atom.atom_number)+" and "+str(r_atom.atom_number)+" in reference molecule\n"   
                                atom.atom_number=r_atom.atom_number
                    if not already:
                        warning+="Warning: could not find equivalent atom in reference molecule to atom: "+str(atom.atom_number)+"\n"
            

            #initially, the unbound atoms will be renumbered as one of the unbound atoms in the reference 
            #this might not be the correct order, but will be solved later.
            #first, the lists unbound_atoms and unbound_ref_atoms are sorted according to their symbol, so an atom is not given the number of an a reference atom of a different element  
            #print([a.symbol.lower() for a in unbound_atoms])
            #print (sorted(zip([a.symbol.lower() for a in unbound_atoms],unbound_atoms)))
            #unbound_ref_atoms_sorted_by_element=[  e for _,e  in  sorted(zip([a.symbol.lower() for a in ref_unbound_atoms],ref_unbound_atoms)) ]
            #for a,aa in zip(unbound_atoms_sorted_by_element,unbound_ref_atoms_sorted_by_element):
            #    ref_to_olf_numbers[aa.atom_number]=a.atom_number
            #    a.atom_number=aa.atom_number
            #for i in range(0,len(unbound_atoms_sorted_by_element)):
            #    ref_to_old_numbers[unbound_ref_atoms_sorted_by_element[i].atom_number]=unbound_atoms_sorted_by_element[i].atom_number
            #    unbound_atoms_sorted_by_element[i].atom_number=unbound_ref_atoms_sorted_by_element[i].atom_number            
                
            for i in range(0,len(unbound_atoms)):
                ref_to_old_numbers[ref_unbound_atoms[i].atom_number]=unbound_atoms[i].atom_number
                unbound_atoms[i].atom_number=ref_unbound_atoms[i].atom_number
            
            #sort the list of atoms           
            self.atom_list.sort(key=lambda x: x.atom_number, reverse=False )
            
            #if there was a match, renumber the atoms according to their order in the list, so the numbers are consecutive
            if warning=="":
                for i in range(0,len(self.atom_list)):
                    old_number=ref_to_old_numbers[self.atom_list[i].atom_number]
                    old_to_new_numbers[old_number]=i+1
                    self.atom_list[i].atom_number=i+1
                for k in ref_to_old_numbers.keys():
                    ref_to_new_numbers[k]=old_to_new_numbers[ref_to_old_numbers[k]]

                
                #change the connections and the back_up_connection: *note that the back_up_connection is also changed because it has no sense to store it with previous atom numbers
                for atom in self.atom_list:
                    for c,b_u_c in zip( atom.connection, atom.back_up_connection):
                        c[0]=old_to_new_numbers[c[0]]
                        b_u_c[0]=old_to_new_numbers[b_u_c[0]]
                
                #change the redundant internal coordinates definitions:
                for icd in self.int_coord_definitions:
                    for a in icd:
                        if a>0: a=old_to_new_numbers[a]
                self.old_to_new_numbers=old_to_new_numbers
                self.ref_to_new_numbers=ref_to_new_numbers
                self.connectivity_lines_sections[0]=self.__print_connectivity()
                
                
                # let's try something better for unbound H atoms... this can be done now because unbound atoms does not need to update connectivity
                if len(unbound_atoms)>2: #if there is only one or zero unbound_atoms, there is no need to change it
                    #rotate the molecule to minimize rmsd with respect to reference (H atoms not included in rmsd calculation)
                    rmsd,fragment_centroid,rotation,translation,fragment_coords=reference_molecule.min_rmsd(self,list_of_atoms=list(ref_to_old_numbers.keys()),exclude_H=True)
                    new_coords=np.array([a.coord for a in self.atom_list]).dot(rotation)-fragment_centroid #+fragment_centroid?
                    self.update_coords(new_coords)

                    #and renumber unbound atoms to the nearest unbound atom in the reference, checking that both have the same symbol
                    for atom in unbound_atoms:
                        dist=999.0; new_atom_number=0 
                        for r_atom in ref_unbound_atoms:
                                if r_atom.symbol==atom.symbol:
                                    d=np.sum((r_atom.coord-atom.coord)**2)
                                    if dist>d: 
                                        dist=d
                                        new_atom_number=r_atom.atom_number
                    
                        ref_to_old_numbers[new_atom_number]=atom.atom_number
                        #ref_to_old_numbers[atom.atom_number]=new_atom_number
                        atom.atom_number=new_atom_number
                
                #sort the list of atoms again          
                self.atom_list.sort(key=lambda x: x.atom_number, reverse=False )
                """
                #change the connections and the back_up_connection: *note that the back_up_connection is also changed because it has no sense to store it with previous atom numbers
                for atom in self.atom_list:
                    for c,b_u_c in zip( atom.connection, atom.back_up_connection):
                        c[0]=old_to_new_numbers[c[0]]
                        b_u_c[0]=old_to_new_numbers[b_u_c[0]]

                #change the redundant internal coordinates definitions:
                for icd in self.int_coord_definitions:
                    for a in icd:
                        if a>0: a=old_to_new_numbers[a]
                self.old_to_new_numbers=old_to_new_numbers
                self.ref_to_new_numbers=ref_to_new_numbers
                self.connectivity_lines_sections[0]=self.__print_connectivity()
                """
            return warning

        """
        # try to guess the fragment of reference_molecule that can be overlaid to current molecule;
        # method used only in "reorder_by_fingerprint", that is no longer used 
        def __select_fragment_to_compare(self,reference_molecule):

            # will work better if long bonds are removed...
            reference_molecule.remove_non_covalent_connections()
            reference_molecule.groups_of_atoms_by_connection()
            reference_molecule.groups_of_atoms_by_connection()
            print (reference_molecule.groups)

            # define a list with the number of H, C, O, N, S, P, and rest of atoms in current molecule
            reference_molecule_fragment=[]
            n_atoms=len(self.atom_list)
            n_H=0; n_C=0; n_O=0; n_N=0; n_S=0; n_P=0
            for a in self.atom_list:
                if a.symbol=="H": n_H+=1
                if a.symbol=="C": n_C+=1
                if a.symbol=="O": n_O+=1
                if a.symbol=="N": n_N+=1
                if a.symbol=="S": n_S+=1
                if a.symbol=="P": n_P+=1
            form=[n_H,n_C,n_O,n_N,n_S,n_P,n_atoms]
            ref_gr_form=[]

            # define list with the number of H, C, O, N, S, P, and rest of atoms for each fragment in reference_molecule
            for gr in reference_molecule.groups:
                ngr_atoms=len(gr)
                ngr_H=0; ngr_C=0; ngr_O=0; ngr_N=0; ngr_S=0; ngr_P=0
                for a in gr:
                    if reference_molecule.atom(a).symbol=="H": ngr_H+=1
                    if reference_molecule.atom(a).symbol=="C": ngr_C+=1
                    if reference_molecule.atom(a).symbol=="O": ngr_O+=1
                    if reference_molecule.atom(a).symbol=="N": ngr_N+=1
                    if reference_molecule.atom(a).symbol=="S": ngr_S+=1
                    if reference_molecule.atom(a).symbol=="P": ngr_P+=1
                ref_gr_form.append([ngr_H,ngr_C,ngr_O,ngr_N,ngr_S,ngr_P,ngr_atoms])
        
            #if one of the group in reference_molecule has a the list of H, C, O, N, S, P, resto f atoms identical to the list of current molecule:
            for i in range(0,len(reference_molecule.groups)):
                if all([f==gr_f for f, gr_f in zip(form,ref_gr_form[i])]): 
                        reference_molecule_fragment=reference_molecule.groups[i]
                        return reference_molecule_fragment

            # if there is not a perfect match, but the difference is only that the reference_molecule has less H atoms, add the nearest atoms to this group:
            if reference_molecule_fragment==[]:
                group_to_expand=999
                for i in range(0,len(reference_molecule.groups)):
                    if all([f==gr_f for f, gr_f in zip(form[1:-1],ref_gr_form[i][1:-1])]):
                        dif_H=form[0]-ref_gr_form[0][i]
                        dif_n=form[-1]-ref_gr_form[-1][i]
                        if dif_H==dif_n and dif_H>0:
                            group_to_expand=reference_molecule.groups[i]
                if group_to_expand!=999:
                    for j in range (0,dif_H): reference_molecule.add_nearest_H_atom_to_group(group_to_expand)
                    for g in reference_molecule.groups:
                        if group_to_expand[0] in g:
                            reference_molecule_fragment=g
                            return reference_molecule_fragment

            # the number of all kind of atoms (except H) in the self molecule is the same that the sum of two of the groups in reference_molecule combined
            if reference_molecule_fragment==[]:
                form_length=len(ref_gr_form[0])
                for i in range (0,len(reference_molecule.groups)):
                    for j in range (i+1,len(reference_molecule.groups)):
                        #add the numbers of H, C, N... or each pair of groups 
                        combined_gr_form=[(ref_gr_form[i][k]+ref_gr_form[j][k]) for k in range(0,form_length) ]
                        #if the combined groups have all atoms (including H):
                        if all([f==gr_f for f, gr_f in zip(form,combined_gr_form)]):
                            reference_molecule.combine_groups(reference_molecule.groups[i],reference_molecule.groups[j])
                            #the new combined group is the last group after applying combine_groups method
                            reference_molecule_fragment=reference_molecule.groups[-1]
                            return reference_molecule_fragment   
                        #if the combined groups have all atoms except H:                
                        if all([f==gr_f for f, gr_f in zip(form[1:-1],combined_gr_form[1:-1])]):
                            dif_H=form[0]-combined_gr_form[0]
                            dif_n=form[-1]-combined_gr_form[-1]
                            if dif_H==dif_n and dif_H>0:
                                print ("combining groups:")
                                print (reference_molecule.groups[i])
                                print ("and:")
                                print (reference_molecule.groups[j])
                                print ("to yield groups:")
                                reference_molecule.combine_groups(reference_molecule.groups[i],reference_molecule.groups[j])
                                print (reference_molecule.groups)
                                print ("now will have to expand...")
                                group_to_expand=reference_molecule.groups[-1]
                                print (group_to_expand)
                                print ("resulting:")
                                for j in range (0,dif_H): 
                                    reference_molecule.add_nearest_H_atom_to_group(group_to_expand)
                                reference_molecule_fragment=reference_molecule.groups[-1]
                                print (reference_molecule_fragment)
                                return reference_molecule_fragment
        """
        #return an array with the number of atoms of each kind in the group with the format:
        #[number of H, C, O, N, S, P, and rest of atoms]
        def __formula(self,group=[]) :

            if group==[]:
                group=range(0,len(self.atom_list))
            elif type(group)==type(1):
                group=self.groups([group])
            n_H=0; n_C=0; n_O=0; n_N=0; n_S=0; n_P=0
            for g in group:
                a=self.atom(g)
                if a.symbol=="H": n_H+=1
                if a.symbol=="C": n_C+=1
                if a.symbol=="O": n_O+=1
                if a.symbol=="N": n_N+=1
                if a.symbol=="S": n_S+=1
                if a.symbol=="P": n_P+=1
            return [n_H,n_C,n_O,n_N,n_S,n_P,len(group)]
            
        #given a reference molecule and a list of atoms, renumber the atoms in current molecule so superposition of the two molecules
        #yields the smaller rmsd; if there was a problem renumbering the atoms, rmsd=999.9
        def __match_numbers_to_group(self,reference_molecule,reference_molecule_group):

            warning=""

            #check if reference_molecule have calculated its fingerprints; if not, calculate them.
            if len(reference_molecule.atom_list[0].fingerprint)==0: 
                reference_molecule.__set_fingerprints()   
            #the number of fingerprints elements in the reference molecule needed to differentiate all of the atoms:
            ref_fingerprint_length=len(reference_molecule.atom_list[0].fingerprint)-1
            #will work with copies of current molecule; otherwise atom ordering will be changed several times  and it will be a mess
            M_copy=copy.deepcopy(self)

            
            # the first attempt to set fingerprints, using an empty(default) rule;
            # it generates a "rules_record" indicating what is solved (and should be solved in a different way in the future to exhaust possibilities).
            # For example, rules_record: [1,1,0,2] indicates that i was neccessary to distinguish a pair of atoms with identical fingerprints, then another pair, then a methyl group, then aset of 3 atoms with identical fingerprint
            rules_record=M_copy.__set_fingerprints(levels=ref_fingerprint_length,rules=[])
            warning=M_copy.__renumber_by_fingerprints(reference_molecule,reference_molecule_group)
            Ms=[M_copy]



            #if renumbering was successful without warnings:
            if warning=="":
                # in the rest of attemps, every pair will be resolved with a different rule to check wich of these solutions overlay better with the reference molecule
                #rules is all possible combinations of list of numbers that will modify atoms fingerprints to make them unique
                # For example, for rules_record: [1,1,0,2]
                # here, rules is: [[[1, 2], [2, 1]], [[1, 2], [2, 1]], [[1]], [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]]
                rules=[ [list(e) for e in itertools.permutations(range(1,r_r+2),r_r+1)] for r_r in rules_record    ]
                # next, itertrools.product is used to generate all possible scenarios
                # For example, rules after product is: rules[0]= [[1, 2], [1, 2], [1], [1, 2, 3]]; rules[1]=[[1, 2], [1, 2], [1], [1, 3, 2]]....; rules[-1]=[[2, 1], [2, 1], [1], [3, 2, 1]]
                rules= [list(p) for p in itertools.product(*rules)]
                for r in rules[1:]:  #skip the first one (it corresponds to the default rules, and was calculated before to get the rules_record)
                    M_copy=copy.deepcopy(self)
                    M_copy.__set_fingerprints(levels=ref_fingerprint_length,rules=r)
                    warning=M_copy.__renumber_by_fingerprints(reference_molecule,reference_molecule_group)
                    if warning=="": Ms.append(M_copy) 
                
                #for all molecules in Ms, determine rmsd, etc and choose the one with smaller rmsd
                best_rmsd=9999.0 
                best_rmsds_index=0              
                for m,index in zip(Ms,range(0,len(Ms))):
                    rmsd,fragment_centroid,rotation,traslation,fragment_coords=reference_molecule.min_rmsd(m,reference_molecule_group)
                    if rmsd<best_rmsd:
                        best_rmsds_index=index
                        best_rmsd= rmsd 
                        best_fragment_centroid=fragment_centroid
                        best_rotation=rotation
                        best_traslation=traslation
                        best_fragment_coords=fragment_coords
                                    
                best_overlay= Ms[best_rmsds_index] 
                #get differences in internal coordinates with a 90 degrees threshold
                dihd_diff=m.get_internal_coordinates_differences(reference_molecule,"dihds")
                different_conf=len([d for d in dihd_diff if d>90]) #threshold: 90
                #rotate coordinates (using fragment_coords since min_rmsd has already calculated that -it is needed for rmsd-)
                best_overlay.update_coords(best_fragment_coords)
                #rotate dipole moment components (if present)
                if all( [ b!=np.nan  for b in best_overlay.dipole_moment] ) :
                    #print (best_fragment_centroid)
                    #best_overlay.dipole_moment=best_overlay.dipole_moment-best_fragment_centroid
                    #best_overlay.dipole_moment=best_overlay.dipole_moment.dot(best_rotation)+best_traslation
                    best_overlay.dipole_moment=best_overlay.dipole_moment.dot(best_rotation) #is this correct? is it enough to rotate the vectors? no need for traslation?
                #rotate normal modes (if present):
                for nm in best_overlay.normal_modes:
                    nm= np.array(nm).dot(best_rotation).tolist()  #is this correct? is it enough to rotate the vectors? no need for traslation?
                # TO DO: rotate other vector properties
                # TO DO: ROTATE POLARIZABILITY TENSOR

                """
                    m.update_coords(fragment_coords)
                    

                    #get differences in internal coordinates with a 90 degrees threshold
                    #dihd_diff=np.array(m.get_internal_coordinates_differences(reference_molecule,"dihds"))
                    #dihdd.append(dihd_diff.mean())
                    dihd_diff=m.get_internal_coordinates_differences(reference_molecule,"dihds")
                    #threshold: 90
                    different_conf=len([d for d in dihd_diff if d>90])
                    dihdd.append(different_conf)
                #print (rmsds) for debuggin
                

                best_rmsd=min(rmsds)
                best_overlay=Ms[rmsds.index(best_rmsd)]
                best_dihdd=dihdd[rmsds.index(best_rmsd)]
                """

                return best_overlay,best_rmsd,different_conf #best_dihdd 
            else:
                #print (warning) #for debuggin
                return Ms[0],999.9,100           


        #the function that should be called to match two molecules or a fragment and a molecule so the atoms are numbered consequently
        #it checks which groups (and combinations) in ref_molecule can be matched to current molecule, based on the number of H, C, O, N, S, P
        #and checks if the overlay is possible with al combinations
        #the dictionaries relating atom numbers before and after the match, and the correspondence with the reference molecule atom numbers are updated 
        #It returns a new molecule which coordinates are overlaid to the reference molecule. If change_numbers==False, the atom numbers are changed
        #in the new molecule but not in the original.
        def match_numbers_to_molecule(self,ref_molecule,change_numbers=True,fast=False):


            #after matching, the next lists will be populated with the different alternatives
            #the best match corresponds to the one with smaller rmsd 
            matched_molecules=[]
            matched_molecules_rmsd=[]
            mathed_molecules_dihd_diff=[]

            #work with a copy of the reference molecule to prevent chaning anything on it
            reference_molecule=copy.deepcopy(ref_molecule)
            

            #first of all, remove connections between heteroatoms and H atoms in current molecule amd reference molecule
            #since these atoms move, their connectivity might be different in current and reference molecule,
            #therefore, it is more convenient to remove the bonds and break the ambiguity by overlaying structures
            self.H_only_bound_to_C()
            reference_molecule.H_only_bound_to_C()

            # define a list with the number of H, C, O, N, S, P, and all atoms in current molecule
            form=self.__formula()
            # define list with the number of H, C, O, N, S, P, and all atoms for each fragment in reference_molecule
            ref_gr_form=[reference_molecule.__formula(gr) for gr in reference_molecule.groups] 
            form_length=len(ref_gr_form[0])   #len(ref_gr_form[0]) should be 7, but coding this way in case more elements were added to the formulas

            #find out which of the groups of the reference molecule (or which combination these groups have the same number of C, O, N, S, and P atoms)
            reference_molecule_compatible_fragments=[]  #list of the groups (or combination of groups) of ref. molecule have the same formula 
            reference_molecule_compatible_fragments_forms=[] #list of the formulas of the groups (or combination of groups) above
            for i in range(0,len(reference_molecule.groups)):
                combinations_of_groups=[list(ii) for ii in itertools.combinations(range(0,len(reference_molecule.groups)),i+1)]
                for c in combinations_of_groups:
                    combination_of_group_form=[0 for ii in range(0, form_length)]
                    for g in c:
                        combination_of_group_form = [f+ff for f,ff in zip(combination_of_group_form,ref_gr_form[g])]

                    if ( fast==True and combination_of_group_form == form) or (fast==False and all([f==gr_f for f, gr_f in zip(form[1:-1],combination_of_group_form[1:-1])])):
                        #if fast==True, compares the number of H,C,O,N,S,P; if fast==False, compares number of C,O,N,S,P, but not the number of H*************
                        reference_molecule_compatible_fragments.append(c)
                        reference_molecule_compatible_fragments_forms.append(combination_of_group_form)

            
            #old code, faster and more reliable using itertools.combinations:
            """        
            for i in range(0,len(reference_molecule.groups)):
                #if one of the group in reference_molecule has the list of the numbers of C, O, N, S, P, identical to the list of current molecule:
                if all([f==gr_f for f, gr_f in zip(form[0:-1],ref_gr_form[i][0:-1])]):
                    reference_molecule_compatible_fragments.append([i])
                    reference_molecule_compatible_fragments_forms.append(ref_gr_form[i])
                else:
                    # if the combination of two groups in the reference molecule has the list of the numbers of C, O, N, S, P, identical to the list of current molecule:
                    for j in range (i+1,len(reference_molecule.groups)):
                        #add the numbers of H, C, N... of groups i and j  
                        combined_gr_form=[(ref_gr_form[i][n]+ref_gr_form[j][n]) for n in range(0,form_length) ]  
                        if all ( [f==gr_f for f, gr_f in zip(form[0:-1],combined_gr_form[1:-1])] ): #this does not compare H
                        #if all ( [f==gr_f for f, gr_f in zip(form[0:-1],combined_gr_form[0:-1])] ):  #this does compare H
                                reference_molecule_compatible_fragments.append([i,j])
                                reference_molecule_compatible_fragments_forms.append(combined_gr_form)
                        else: # if the combination of 3 groups...
                            for k in range (j+1,len(reference_molecule.groups)):
                                combined_gr_form=[(ref_gr_form[i][n]+ref_gr_form[j][n]+ref_gr_form[k][n]) for n in range(0,form_length)]
                                if all([f==gr_f for f, gr_f in zip(form[1:-1],combined_gr_form[1:-1])]):  #this does not compare H
                                #if all([f==gr_f for f, gr_f in zip(form[0:-1],combined_gr_form[0:-1])]):
                                    reference_molecule_compatible_fragments.append([i,j,k])
                                    reference_molecule_compatible_fragments_forms.append(combined_gr_form)
                                else: # if the combination of 4 groups...
                                    for l in range (k+1,len(reference_molecule.groups)):
                                        combined_gr_form=[(ref_gr_form[i][n]+ref_gr_form[j][n]+ref_gr_form[k][n]+ref_gr_form[l][n]) for n in range(0,form_length)]
                                        if all([f==gr_f for f, gr_f in zip(form[1:-1],combined_gr_form[1:-1])]):#this does not compare H
                                        #if all([f==gr_f for f, gr_f in zip(form[0:-1],combined_gr_form[0:-1])]):
                                            reference_molecule_compatible_fragments.append([i,j,k,l])
                                            reference_molecule_compatible_fragments_forms.append(combined_gr_form)  
                                        else: # if the combination of 5 groups...
                                            for m in range (l+1,len(reference_molecule.groups)):
                                                combined_gr_form=[(ref_gr_form[i][n]+ref_gr_form[j][n]+ref_gr_form[k][n]+ref_gr_form[l][n]+ref_gr_form[m][n]) for n in range(0,form_length)]
                                                if all([f==gr_f for f, gr_f in zip(form[1:-1],combined_gr_form[1:-1])]):#this does not compare H
                                                #if all([f==gr_f for f, gr_f in zip(form[0:-1],combined_gr_form[0:-1])]):
                                                    reference_molecule_compatible_fragments.append([i,j,k,l,m])
                                                    reference_molecule_compatible_fragments_forms.append(combined_gr_form)  
                                            else: # if the combination of 6 groups...
                                                for o in range (m+1,len(reference_molecule.groups)):
                                                    combined_gr_form=[(ref_gr_form[i][n]+ref_gr_form[j][n]+ref_gr_form[k][n]+ref_gr_form[l][n]+ref_gr_form[m][n]+ref_gr_form[o][n]) for n in range(0,form_length)]
                                                    if all([f==gr_f for f, gr_f in zip(form[1:-1],combined_gr_form[1:-1])]):#this does not compare H
                                                    #if all([f==gr_f for f, gr_f in zip(form[0:-1],combined_gr_form[0:-1])]):
                                                        reference_molecule_compatible_fragments.append([i,j,k,l,m,o])
                                                        reference_molecule_compatible_fragments_forms.append(combined_gr_form) 
            """
            #for i in range(0,len(reference_molecule.groups)):print (str(i)+":"+str(reference_molecule.groups[i])) #for debuggin

            #once the compatible combination of groups are identified, iterate through them
            
            for fragments,fragment_forms in zip(reference_molecule_compatible_fragments,reference_molecule_compatible_fragments_forms):

                # reset connections and combine groups  needed?
                reference_molecule.reset_connections()
                reference_molecule.H_only_bound_to_C() 
                #recalculate fingerprints of the reference molecule and the current molecule:
                self.__set_fingerprints()
                reference_molecule.__set_fingerprints()

                #print (self.atom(2).fingerprint)
                #print (reference_molecule.atom(2).fingerprint)

                # list containing the list of atoms to combine
                atoms_in_fragments=[reference_molecule.groups[fragments[i]] for i in range (0,len(fragments))]

                # combine the groups in reference molecule so all atoms in compatible fragments (or combination) are in the last group
                reference_molecule.combine_groups(atoms_in_fragments)
                

                #if there is no need to add any H atom:
                if fragment_forms[0]==form[0] and fragment_forms[-1]==form[-1]:

                    #combine_groups has placed the group of interest last 
                    new_mol,rmsd,dihd_diff=self.__match_numbers_to_group(reference_molecule,reference_molecule.groups[-1])

                    if rmsd<999.9:
                        matched_molecules.append(new_mol)
                        matched_molecules_rmsd.append(rmsd)
                        mathed_molecules_dihd_diff.append(dihd_diff)
                        #print ("fragment:"+str(fragments)+" matched!") # for debugging
                    # reset connections and combine groups  needed?
                    reference_molecule.reset_connections()
                    reference_molecule.H_only_bound_to_C() 


                #this part of the code adds or removes H atoms; it only has sense if the fragments were built without comparing the number of H atoms
                if fast==False:

                    # if it is neccessary to add H atoms (that were bound to C atoms in other groups)
                    if fragment_forms[0]-form[0]==fragment_forms[-1]-form[-1] and  fragment_forms[0]<form[0]:

                        #figure out which is the number of the group to which H(-C) atoms will be added.
                        all_atoms_in_fragments=sum(atoms_in_fragments,[])  # a 1D list of all atoms in the fragments
                        index_of_group=0
                        for g in reference_molecule.groups:
                            if all ([a in all_atoms_in_fragments for a in g]): break    
                            index_of_group+=1

                        # the number of H that should be added:
                        dif_H=form[0]-fragment_forms[0]
                        
                        # the combinations of H(-C) that will be added, where the number n refers to the nth closest H(-C) atom to current group
                        n=[list(a) for a in itertools.combinations(list(range(0,dif_H+3)),dif_H)]
                        for i in n:
                            # reset connections and combine groups  needed?
                            reference_molecule.reset_connections()
                            reference_molecule.H_only_bound_to_C() 
                            reference_molecule.combine_groups(atoms_in_fragments)
                            # will work on a copy of reference molecule so everything is reset on each iteration
                            copy_of_reference_molecule=copy.deepcopy(reference_molecule)
                            copy_of_atoms_in_fragments=copy.deepcopy(atoms_in_fragments) 
                            #add the required H(-C):
                            number_of_H_added=copy_of_reference_molecule.add_nearest_H_atom_to_group(copy_of_reference_molecule.groups[index_of_group],i)
                            #only check those cases in which it was possible to add the required number of H(-C) atoms.
                            if number_of_H_added==dif_H:
                                #update fingerprings (it would only change if the new H is bound to a C atom)
                                copy_of_reference_molecule.__set_fingerprints()
                                #overlay to get rmsd; the last copy_of_reference_molecule.groups is used, since add_nearest_H_atom_to_group took care of placing the group to the end of the list
                                new_mol,rmsd,dihd_diff=self.__match_numbers_to_group(copy_of_reference_molecule,copy_of_reference_molecule.groups[-1])
                                #if there were no errors, add the molecule to the list of candidate molecules.
                                if rmsd<999.9:
                                    matched_molecules.append(new_mol)
                                    matched_molecules_rmsd.append(rmsd)
                                    mathed_molecules_dihd_diff.append(dihd_diff)
                                    #print ("fragment:"+str(fragments)+" matched after adding a H!") # for debugging

                    # if it is neccessary to remove one H(-C) atom:
                    elif fragment_forms[0]-form[0]==fragment_forms[-1]-form[-1] and  fragment_forms[0]>form[0]:
                    #elif fragment_forms[0]==form[0]+1 and fragment_forms[-1]==form[-1]+1:

                        #figure out which is the number of the group from which H(-C) atoms will be removed.
                        all_atoms_in_fragments=sum(atoms_in_fragments,[])  # a 1D list of all atoms in the fragments
                        index_of_group=0
                        for g in reference_molecule.groups:
                            if all ([a in all_atoms_in_fragments for a in g]): break    
                            index_of_group+=1  

                        #list of all posible H atoms that can be removed: this includes all H atoms bound to something, so it could take long...
                        possible_H_atoms_to_remove=[reference_molecule.atom(atom).atom_number for atom in reference_molecule.groups[index_of_group] if (reference_molecule.atom(atom).symbol.lower()=="h") and len(reference_molecule.atom(atom).connection)>0]

                        # the number of H that should be removed:
                        dif_H=fragment_forms[0]-form[0]
                        #we will not be able to do nothing if we need to remove more H atoms than thenumber of possible atoms to remove
                        #if len(possible_H_atoms_to_remove)>dif_H: #this will take a huge amount of time if dif_H>2
                        if len(possible_H_atoms_to_remove)>dif_H and dif_H<2:
                            # the possible combinations of H(-C) that could be removed
                            combinations_of_H_to_remove=[list(a) for a in itertools.combinations(possible_H_atoms_to_remove,dif_H)]

                            for h in combinations_of_H_to_remove:
                                # reset connections and combine groups  needed?
                                reference_molecule.reset_connections()
                                reference_molecule.H_only_bound_to_C() 
                                reference_molecule.combine_groups(atoms_in_fragments)
                                # will work on a copy of reference molecule so everything is reset on each iteration
                                copy_of_reference_molecule=copy.deepcopy(reference_molecule)
                                copy_of_atoms_in_fragments=copy.deepcopy(atoms_in_fragments) 

                                copy_of_reference_molecule.remove_H_atom_from_group(h)
                                new_mol,rmsd,dihd_diff=self.__match_numbers_to_group(copy_of_reference_molecule,copy_of_reference_molecule.groups[-1])
                                if rmsd<999.9:
                                    matched_molecules.append(new_mol)
                                    matched_molecules_rmsd.append(rmsd)
                                    mathed_molecules_dihd_diff.append(dihd_diff)
                                    #print ("fragment:"+str(fragments)+" matched after removing one H!") # for debugging
                                

                
            if len(matched_molecules)>0:
                min_rmsd=min(matched_molecules_rmsd)
                new_mol= matched_molecules[matched_molecules_rmsd.index(min_rmsd)]
                min_dihd_diff=mathed_molecules_dihd_diff[matched_molecules_rmsd.index(min_rmsd)]
                #change numbers of current molecule if it is requested                
                if  change_numbers:
                    for i in range (0,len(self.atom_list)):
                        self.atom_list[i].atom_number=new_mol.old_to_new_numbers[self.atom_list[i].atom_number]
                        for c in self.atom_list[i].connection:
                            c[0]=new_mol.old_to_new_numbers[c[0]]
                    self.atom_list.sort(key=lambda x: x.atom_number, reverse=False )
                    self.old_to_new_numbers=new_mol.old_to_new_numbers
                    self.ref_to_new_numbers=new_mol.ref_to_new_numbers

                #change also the connections numbers of new_mol
                for i in range(0,len(self.atom_list)): new_mol.atom(i).connection=self.atom(i).connection


                return new_mol,min_rmsd,min_dihd_diff

            else:
                print ("unable to match molecules:"+str(ref_molecule.output_file)+" and"+str(self.output_file)+", method: match_numbers_to_molecule() failed because len(matched_molecules)==0 ")


        #matches the atoms of fragment_molecule and remove from current molecule common atoms
        def remove_common_atoms(self,fragment_molecule,fast=True):

            matched_fragment_molecule,rmsd,dihdiff = fragment_molecule.match_numbers_to_molecule(self,fast=fast)
            #print(matched_fragment_molecule.ref_to_new_numbers)   #for debuggin
            atoms_to_remove=list(matched_fragment_molecule.ref_to_new_numbers.keys() )
            #print (atoms_to_remove) #for debuggin
            self.remove_atoms(atoms_to_remove)

        #matches the atoms of fragment_molecule and keep only these atoms in current molecule
        def keep_common_atoms(self,fragment_molecule,fast=True):
            
            matched_fragment_molecule,rmsd,dihdiff = fragment_molecule.match_numbers_to_molecule(self,fast=fast)
            atoms_to_keep=list(matched_fragment_molecule.ref_to_new_numbers.keys() )
            atoms_to_remove=[a for a in range(1,len(self.atom_list)+1) if a not in atoms_to_keep]
            self.remove_atoms(atoms_to_remove)


        #METHODS FOR CALCULATING INTERACTION ENERGIES BETWEEN ATOMS IN CURRENT MOLECULE AND ATOMS IN OTHER MOLECULE
        def electrostatic_energy(self,other_molecule):
            energy=0.0
            for a in self.atom_list:
                for aa in other_molecule.atom_list:
                    dist= angs_to_bohr*(np.sum( (a.coord-aa.coord)**2 )**0.5)
                    energy+=hartrees_to_kcal*a.charge*aa.charge/dist
                    #print ("("+str(a.atom_number)+")"+str(a.charge)+"-("+str(aa.atom_number)+")"+str(aa.charge)+"-dist:"+str(dist)+"  int. energy:"+str(hartrees_to_kcal*a.charge*aa.charge/dist)+" acumulated:"+str(energy))
            return energy
                    
        
        def uff_vdw_energy(self,other_molecule,method="6-12"):
            energy=0.0
            for a in self.atom_list:
                vdw_a=self.uff_vdw[a.symbol.lower()]
                for aa in other_molecule.atom_list:
                    vdw_aa=self.uff_vdw[aa.symbol.lower()]
                    #xij=(vdw_aa[0]*vdw_a[0])**0.5
                    xij=(vdw_aa[0]+vdw_a[0])*0.5
                    Dij=(vdw_aa[1]*vdw_a[1])**0.5
                    #dist= angs_to_bohr* np.sum( (a.coord-aa.coord)**2 )**0.5
                    dist= 1.0* np.sum( (a.coord-aa.coord)**2 )**0.5
                    xij_x=(xij/dist)**6
                    if method=="6-12": energy+=Dij*(-2*xij_x+xij_x**2)
                #print (str(a.atom_number)+"-"+str(energy)) #for debuging
                    
            return energy


        #based on: J. Am. Chem. Soc. 1989, 111 (23), 8576-8582.
        def mm3_vdw_energy(self,other_molecule):

            #make copies of the coordinates of the atoms (to revert the changes on them after calculating the energy)
            initial_coords_self=np.array( [ a.coord   for a in self.atom_list  ]) 
            initial_coords_other=np.array( [ a.coord   for a in other_molecule.atom_list  ])

               
            
            if any([a.atom_type=="" for a in self.atom_list]):
                self.generate_connections_from_distance(H_only_bound_to_C=False,allow_fract_orders=True)
                #self.reset_connections()  #problem: some H atoms were not updated after renumbering
                self.set_MM3_atom_types()
            if any([a.atom_type=="" for a in other_molecule.atom_list]):
                other_molecule.generate_connections_from_distance(H_only_bound_to_C=False,allow_fract_orders=True) 
                #self.reset_connections()  #problem: some H atoms were not updated after renumbering
                other_molecule.set_MM3_atom_types()

            energy=0.0
            a_coord=np.array([0,0,0])
            aa_coord=np.array([0,0,0])

            for a in self.atom_list:
                if int(a.atom_type)<200: a_atom_type=a.atom_type
                else: a_atom_type=str(int(a.atom_type)-200)

                #MM3 reduces the atom distance of H to their bound atoms by a 0.923 factor
                if a_atom_type in ["5","21","23","24","28","36","44","48","73","124"]:
                    if len(a.connection)==1:
                        c_coord=self.atom(a.connection[0][0]).coord
                        a_coord=a.coord*0.923+c_coord*(1-0.923)
                    else: a_coord=a.coord
                else: a_coord=a.coord
                #a_coord=a.coord
                #habría que volver a ponerlo como estaba antes; otro problema es que la conectividad a los heteroátomos se ha perdido!!!

                #cutoff value for short distances between atoms: (from the tinker emm3hb.f file)
                r_dist_cutoff=0.5
                expmerge= (184000*math.exp(-12*r_dist_cutoff)-2.25*(0.5**-6))*(0.5**12)

                for aa in other_molecule.atom_list:

                    if int(aa.atom_type)<200: aa_atom_type=aa.atom_type
                    else: aa_atom_type=str(int(aa.atom_type)-200)   

                    if a_atom_type+"-"+aa_atom_type in self.mm3_vdw_pairs.keys(): 
                        epsilon= self.mm3_vdw_pairs[ a_atom_type+"-"+aa_atom_type ][1]  
                        sum_of_radii=  self.mm3_vdw_pairs[ a_atom_type+"-"+aa_atom_type ][0]
                    elif aa_atom_type+"-"+a_atom_type in self.mm3_vdw_pairs.keys():
                        epsilon= self.mm3_vdw_pairs[ aa_atom_type+"-"+a_atom_type ][1]  
                        sum_of_radii=  self.mm3_vdw_pairs[ aa_atom_type+"-"+a_atom_type ][0]
                    else:
                        epsilon=(self.mm3_vdw[a_atom_type][1]*self.mm3_vdw[aa_atom_type][1])**0.5
                        sum_of_radii= (self.mm3_vdw[a_atom_type][0]+self.mm3_vdw[aa_atom_type][0])


                    if aa_atom_type in ["5","21","23","24","28","36","44","48","73","124"]:
                        if len(aa.connection)==1:
                            c_coord=other_molecule.atom(aa.connection[0][0]).coord
                            aa_coord=aa.coord*0.923+c_coord*(1-0.923)
                        else: aa_coord=aa.coord
                    else: aa_coord=aa.coord
                    #aa_coord=aa.coord
                    #directional H-bonds
                    cos_beta=1
                    if len(a.connection)==1 and ( (a_atom_type in ["21","23","28","44","73","124"] and aa_atom_type in ["2"]) or\
                        (a_atom_type in ["21","23","28","124"] and aa_atom_type in ["4"]) or\
                        (a_atom_type in ["21","23","24","28","44","73","124"] and aa_atom_type in ["6"]) or\
                        (a_atom_type in ["21","23","24","28","73","124"] and aa_atom_type in ["7","8","12",]) or\
                        (a_atom_type in ["21","23","24","28","73"] and aa_atom_type in ["11","13","14","15","17"]) or\
                        (a_atom_type in ["21","23","28","124"] and aa_atom_type in ["22"]) or\
                        (a_atom_type in ["21","23","24","28","73","124"] and aa_atom_type in ["37"]) or\
                        (a_atom_type in ["21","23","28","48","124"] and aa_atom_type in ["47"]) or\
                        (a_atom_type in ["21","23","28","73"] and aa_atom_type in ["50"]) or\
                        (a_atom_type in ["21","23","28","124"] and aa_atom_type in ["79"]) ) :
                        v1=a.coord-self.atom(a.connection[0][0]).coord; mod_v1=np.linalg.norm(v1)
                        v2=aa.coord-self.atom(a.connection[0][0]).coord; mod_v2=np.linalg.norm(v1)
                        cos_beta=v1.dot(v2)/(mod_v1*mod_v2)
                    if len(aa.connection)==1 and ( (aa_atom_type in ["21","23","28","44","73","124"] and a_atom_type in ["2"]) or\
                        (aa_atom_type in ["21","23","28","124"] and a_atom_type in ["4"]) or\
                        (aa_atom_type in ["21","23","24","28","44","73","124"] and a_atom_type in ["6"]) or\
                        (aa_atom_type in ["21","23","24","28","73","124"] and a_atom_type in ["7","8","12",]) or\
                        (aa_atom_type in ["21","23","24","28","73"] and a_atom_type in ["11","13","14","15","17"]) or\
                        (aa_atom_type in ["21","23","28","124"] and a_atom_type in ["22"]) or\
                        (aa_atom_type in ["21","23","24","28","73","124"] and a_atom_type in ["37"]) or\
                        (aa_atom_type in ["21","23","28","48","124"] and a_atom_type in ["47"]) or\
                        (aa_atom_type in ["21","23","28","73"] and a_atom_type in ["50"]) or\
                        (aa_atom_type in ["21","23","28","124"] and a_atom_type in ["79"]) ):
                        v1=aa.coord-other_molecule.atom(aa.connection[0][0]).coord; mod_v1=np.linalg.norm(v1)
                        v2=a.coord-other_molecule.atom(aa.connection[0][0]).coord; mod_v2=np.linalg.norm(v1)
                        cos_beta=v1.dot(v2)/(mod_v1*mod_v2)


                    dist= np.sum( (a_coord-aa_coord)**2 )**0.5
                    r_dist=(dist/sum_of_radii)
                    if r_dist<r_dist_cutoff:
                        e=epsilon*expmerge*(r_dist**-12)
                    else:
                        e=epsilon*(-2.25*cos_beta*(r_dist**-6)+ 184000*math.exp(-12.0*r_dist))
                    energy+= e
                    #for debuggin:
                    #if dist<1.0:
                    #    print (self.output_file)
                    #    print (other_molecule.output_file)                        
                    #    print ( str(a.atom_number)+"("+str(a_atom_type)+")-"+str(aa.atom_number)+"("+str(aa_atom_type)+") dist:"+str(dist)+" r_dist:"+str(r_dist)+" E:"+str(e)+" cos_beta:"+str(cos_beta)+" tot:"+str(energy))      

                    #if epsilon*(-2.25*(r_dist**-6)+ 184000*math.exp(-12.0*r_dist))<-100:
                    #    print (self.output_file)
                    #    print (other_molecule.output_file)
                    #    print ( str(a.atom_number)+"("+str(a_atom_type)+")-"+str(aa.atom_number)+"("+str(aa_atom_type)+") dist:"+str(dist)+" E:"+str(epsilon*(-2.25*(r_dist**-6)+ 184000*math.exp(-12.0*r_dist)))+" tot:"+str(energy))      
                    
                #print (str(a.atom_number)+"-"+str(energy)) #for debuging

            #restore the coordinates:
            for m,c in zip([self,other_molecule],[initial_coords_self,initial_coords_other]): m.update_coords(c)

            return energy




#M=Molecular_structure("g16MPN1.out")
#print (M.print_mdcrd())


#M3=Molecular_structure("test-opt.out","test-opt.com")
#M4=Molecular_structure("test-small.com")

#a=np.array([[-1.041254  ,  0.549188  ,  1.565571  ],
#            [-1.041254  ,  0.549188  ,  1.565571  ],
#            [-0.188903  , -0.469621  , -0.694882  ]])

#translation,rotation= M3.kabsch(M4,[52,53,54,55,56,57,58,59,100,101])
#print (M4.center_of_mass)
#M4.trans_and_rot(translation,rotation)
#print (M4.center_of_mass)
#print(M4.print_new_file())
#print (translation)
#print (rotation)
#print (type(M3))
#M1=Molecular_structure("test-or1.com")
#print (M1.connectivity_lines_sections)
#M2=Molecular_structure("test-or1.com")
#print (M2.groups)
#M3=Molecular_structure(M2,[4])
#print (M2.groups[3])
#print (M2.print_new_file())
#print (M2.connectivity_lines_sections)
#print (M2.print_new_file())
#rmsd,centroid,rotation,translation,fragment_coords=M2.min_rmsd(M3,M2.groups[3])
#M3.update_coords(fragment_coords)
#print (M3.connectivity_lines_sections)
#print (M3.print_new_file())

#print (type(M5.atom_list))
#print (M5.G16_output.first_freq_movement)
#print (M5.G16_output.dipole_moments)
#print (M5.G16_output.energies)
#print (M5.dipole_moment)
#print (M5.groups)
#M6=Molecular_structure(M5,[2])
#print (M6.print_new_file())
#M5.groups_of_atoms_by_connection()
#print (M5.center_of_mass())
#M5.update_coords_from_xyz("1-4RSexo-4.1.xyz")
#print (M5.print_new_file())

#print(M3.bonds_between(3))
#M3.rotate_bond(35,37,90)
#M3.change_bond_distance(35,37,"+50%")
#M3.change_angle(35,30,37,40)
#r=open("rotated.com","w")
#r.write(M3.print_new_file())
#r.close()



#M=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/product/atrop-PROD-E_im-anti-N·N+_N1_E-N2_E.out","last")
#M=Molecular_structure("atrop-PROD-E_im-anti-N·N+_N1_E-N2_E.out","last")
#M.print_pdb("atrop-PROD-E_im-anti-N·N+_N1_E-N2_E.pdb")




#np.set_printoptions(formatter={'float_kind':"{:.4f}".format})



"""
#m=Molecular_structure("atrop-DADC-protonated-N=N_E-N1_E-N2_E.out","last")
m=Molecular_structure("DADC-hess.out","last")
m.print_pdb("reference-g16.pdb")
m.read_hess()
print(m.cart_hess)
print("===============")
m=Molecular_structure("atrop-DADC-protonated-N=N_E-N1_E-N2_E.out","last")
m.print_pdb("reference_orca.pdb")
m.read_hess()
print(m.cart_hess)

#m=Molecular_structure("DADC-hess.out","last")
m=Molecular_structure("atrop-DADC-protonated-N=N_E-N1_E-N2_E.out","last")

mm=copy.deepcopy(m)
mm.print_pdb("before_rot.pdb")
mm.read_hess()
print (mm.cart_hess)
matchedmm,rms,dihdiff=mm.match_numbers_to_molecule(m)
matchedmm.print_pdb("after_match.pdb")
q=quaternion.from_euler_angles(np.pi/2,0,np.pi/4)
for a in matchedmm.atom_list:
    a.coord=quaternion.rotate_vectors(q,a.coord)
    #+np.array([3,2,1])
matchedmm.print_pdb("after_rot.pdb")
a,b,c=matchedmm.min_hess_distort(m)
print (a)
print (b)
print (c)
matchedmm.print_pdb("after-minhess.pdb")

m=Molecular_structure("atrop-DADC-protonated-N=N_E-N1_E-N2_E.out","last")
m.read_normal_modes()
print (m.normal_modes[1])

m=Molecular_structure("DADC-hess.out","last")
m.read_normal_modes()
print (m.normal_modes[1])
"""

"""

format_s="{:8.5f}"

M=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/atrop-R-anti-E_im-N=N_E-N1_E-N2_E-1-uff.out")
M.print_pdb("ref.pdb")

m=Molecular_structure("atrop-DADC-protonated-N=N_E-N1_E-N2_E.out","last")
m.print_pdb("DADC-orca.pdb")


m=Molecular_structure("DADC-hess.out","last")
m.print_pdb("DADC-G16-original.pdb")

m.read_hess()
m2=copy.deepcopy(m)


#Newmm,rmsd,dihdiff=m.match_numbers_to_molecule(mm)
#Newm,rmsd,dihdiff=mm.match_numbers_to_molecule(m)

centroid=np.mean([a.coord for a in m.atom_list],axis=0)
#r=Rotation.from_euler("xyz",[0,0,np.pi/2])
r=Rotation.random()
t=np.random.rand(3)
for a in m.atom_list:
    a.coord= r.apply(a.coord-centroid)+centroid+t
m.print_pdb("DADC-G16-rotated_random.pdb")
newm,_,_=m.match_numbers_to_molecule(m2)

print (m.ref_to_new_numbers)
a,b,c= m.min_hess_distort(m2)

print (a)
print (b)
print (c)

m.print_pdb("DADC-G16-minhess.pdb")

rmsd=0
print 
for a,aa in zip(m.atom_list,m2.atom_list):
    #print (a.coord)
    #print (aa.coord)
    #print ("----")
    #print (a.coord-aa.coord)
    #print 
    
    rmsd+=np.sum((a.coord-aa.coord)**2)
print (str(rmsd**0.5))


newm.print_pdb("DADC-G16-rmsd.pdb")
#aa,bb,cc=newm.min_hess_distort(m2)
#print("first,rmsd, then, min_hess")
#print(aa)
#print(bb)
#print(cc)
#newm.print_pdb("DADC-G16-rmsd-minhess.pdb")




                

#for a in m.atom_list: a.coord=a.coord+np.array([4,4,4])
#for a in mm.atom_list: a.coord=a.coord+np.array([-4,-4,4])

#aa,bb,cc=mm.min_hess_distort(m)
#print (bb)
#print (cc)
#mm.print_pdb("hessmatch_g16_to_ORCA.pdb")

#a,b,c=m.min_hess_distort(mm)
#print (b)
#print (c)
#m.print_pdb("hessmatch_ORCA_to_g16.pdb")
"""





"""
newM,rmsd,dihdiff=m.match_numbers_to_molecule(M)
newM.print_pdb("rmsdmatch-DADC-orca.pdb")
#m.print_pdb("DADCr.pdb")
a,b,c=newM.min_hess_distort(M)
print (a)
print (b)
print (c)
print ("=============")
aa,bb,cc=m.min_hess_distort(M)
print (aa)
print (bb)
print (cc)
newM.print_pdb("hessmatch-DADC-orca.pdb")
m.print_pdb("hessmatch-or-DADC-orca.pdb")


m=Molecular_structure("DADC-hess.out","last")
m.print_pdb("DADC-g16.pdb")
m.read_hess()
newM,rmsd,dihdiff=m.match_numbers_to_molecule(M)
newM.print_pdb("rmsdmatch-DADC-g16.pdb")
a,b,c=newM.min_hess_distort(M)
#a,b,c=m.min_hess_distort(M)
print (a)
print (b)
print (c)
print ("=============")
aa,bb,cc=m.min_hess_distort(M)
print (aa)
print (bb)
print (cc)
newM.print_pdb("hessmatch-DADC-g16.pdb")
m.print_pdb("hessmatch-or-DADC-g16.pdb")
"""


"""
M=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/atrop-R-anti-E_im-N=N_E-N1_E-N2_E-1-uff.out")
M.print_pdb("ref.pdb")
format_s="{:8.5f}"

m=Molecular_structure("atrop-cat-anion-1-hess.out","last")
m.print_pdb("cat.pdb")
m.read_hess()
newM,rmsd,dihdiff=m.match_numbers_to_molecule(M)
newM.print_pdb("rotated-cat.pdb")
a,b,c=newM.min_hess_distort(M)
print (a)
print (b)
print (c)
newM.print_pdb("re-rotated-cat.pdb")
"""

"""
M_copy=copy.deepcopy(M)
allfiles=os.listdir("/Users/luissimon/calculos/atrop/UFF/fragments/")
outfiles=[f for f in allfiles if f.split(".")[-1]=="out" ]
prodfiles=[f for f in outfiles if f.find("DADC-neut")>-1]
Ms=[Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/"+m, "last") for m in prodfiles ]
results=""
#Ms=Ms[0:1]
for m in Ms:
    filename=m.output_file.split("/")[-1].split(".out")[0]
    m.print_pdb(filename+".pdb")
    newM,rmsd,dihdiff=m.match_numbers_to_molecule(M_copy)
    results+= (m.output_file+": "+format_s.format(rmsd)+"   number of different dihedrals:"+str(dihdiff)+"\n")
    newM.print_pdb("rotated-"+filename+".pdb")
    print (newM.orca_hess_file)
    newM.__read_hess_from_orca_hess_file()

    diff_c=newM.get_cart_coordinates_differences(M)
    print (newM.cart_hess)
    print (diff_c.T.dot(newM.cart_hess).dot(diff_c))
print (results)


M_copy=copy.deepcopy(M)
allfiles=os.listdir("/Users/luissimon/calculos/atrop/UFF/fragments/old")
outfiles=[f for f in allfiles if f.split(".")[-1]=="out" ]
#prodfiles=[f for f in outfiles if f.find("DADC-protonated")>-1]
prodfiles=[f for f in outfiles if f.find("DADC-protonated")>-1]
Ms=[Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/"+m, "last") for m in prodfiles ]
results=""
#Ms=Ms[0:1]
molecules=[]
rmsds=[]
for m in Ms:
    filename=m.output_file.split("/")[-1].split(".out")[0]
    #m.print_pdb(filename+".pdb")
    newM,rmsd,dihdiff=m.match_numbers_to_molecule(M_copy)
    molecules.append(newM)
    rmsds.append(rmsd)
    results+= (m.output_file+": "+format_s.format(rmsd)+"   number of different dihedrals:"+str(dihdiff)+"\n")
    #newM.print_pdb("rotated-"+filename+".pdb")
print (results)

DADC=molecules[rmsds.index(min(rmsds))]


M_copy=copy.deepcopy(M)
allfiles=os.listdir("/Users/luissimon/calculos/atrop/UFF/fragments/old")
outfiles=[f for f in allfiles if f.split(".")[-1]=="out" ]
prodfiles=[f for f in outfiles if f.find("IM")>-1]
Ms=[Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/"+m, "last") for m in prodfiles ]
results=""
#Ms=Ms[0:1]
molecules=[]
rmsds=[]
for m in Ms:
    filename=m.output_file.split("/")[-1].split(".out")[0]
    m.print_pdb(filename+".pdb")
    newM,rmsd,dihdiff=m.match_numbers_to_molecule(M_copy)
    molecules.append(newM)
    rmsds.append(rmsd)
    results+= (m.output_file+": "+format_s.format(rmsd)+"   number of different dihedrals:"+str(dihdiff)+"\n")
    newM.print_pdb("rotated-"+filename+".pdb")
print (results)

im=molecules[rmsds.index(min(rmsds))]

print (DADC.old_to_new_numbers)

DADC.join(im)
DADC.print_pdb("combined.pdb")

print (DADC.old_to_new_numbers)


M_copy=copy.deepcopy(M)
allfiles=os.listdir("/Users/luissimon/calculos/atrop/UFF/fragments/old")
outfiles=[f for f in allfiles if f.split(".")[-1]=="out" ]
prodfiles=[f for f in outfiles if (  f.find("cat-anion")>-1 and  f.find("hess")>-1  ) ]
Ms=[Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/"+m, "last") for m in prodfiles ]
results=""
#Ms=Ms[0:1]
for m in Ms:
    filename=m.output_file.split("/")[-1].split(".out")[0]
    m.print_pdb(filename+".pdb")
    newM,rmsd,dihdiff=m.match_numbers_to_molecule(M_copy)
    results+= (m.output_file+": "+format_s.format(rmsd)+"   number of different dihedrals:"+str(dihdiff)+"\n")
    newM.print_pdb("rotated-"+filename+".pdb")
print (results)



M_copy=copy.deepcopy(M)
allfiles=os.listdir("/Users/luissimon/calculos/atrop/UFF/fragments/old/product/")
outfiles=[f for f in allfiles if f.split(".")[-1]=="out" ]
prodfiles=[f for f in outfiles if f.find("PROD")>-1]
Ms=[Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/product/"+m, "last") for m in prodfiles ]
results=""
#Ms=Ms[0:1]
for m in Ms:
    filename=m.output_file.split("/")[-1].split(".out")[0]
    m.print_pdb(filename+".pdb")
    newM,rmsd,dihdiff=m.match_numbers_to_molecule(M_copy)
    results+= (m.output_file+": "+format_s.format(rmsd)+"   number of different dihedrals:"+str(dihdiff)+"\n")
    newM.print_pdb("rotated-"+filename+".pdb")
print (results)





M=Molecular_structure("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-1-hess.gjf")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-2-sc")
M.rotate_bond(21,96,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-4-sc")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-3-sc")

M=Molecular_structure("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-1-hess.gjf")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-2-sc")
M.rotate_bond(21,96,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-4-sc")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-3-sc")

M=Molecular_structure("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_E-1-hess.gjf")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_E-2-sc")
M.rotate_bond(21,96,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_E-4-sc")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_E-3-sc")

M=Molecular_structure("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_Z-1-hess.gjf")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_Z-2-sc")
M.rotate_bond(21,96,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_Z-4-sc")
M.rotate_bond(7,119,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1-Z-N2_Z-3-sc")


M=Molecular_structure("AatropamSc6-S_C-S_N-N=N_E-N1_E-N2_Z-1-hess.gjf")
M.rotate_bond(7,96,180)
M.print_new_file("AatropamSc6-S_C-S_N-N=N_E-N1_E-N2_Z-1")
M.rotate_bond(21,119,180)
M.print_new_file("AatropamSc6-S_C-S_N-N=N_E-N1_E-N2_Z-3")
M.rotate_bond(7,96,180)
M.print_new_file("AatropamSc6-S_C-S_N-N=N_E-N1_E-N2_Z-4")

M=Molecular_structure("AatropamSc6-S_C-R_N-N=N_E-N1_E-N2_E-1-hess.gjf")
M.rotate_bond(7,96,180)
M.print_new_file("AatropamSc6-S_C-R_N-N=N_E-N1_E-N2_E-1")
M.rotate_bond(21,119,180)
M.print_new_file("AatropamSc6-S_C-R_N-N=N_E-N1_E-N2_E-3")
M.rotate_bond(7,96,180)
M.print_new_file("AatropamSc6-S_C-R_N-N=N_E-N1_E-N2_E-4")

M=Molecular_structure("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-1-hess.gjf")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-1")
M.rotate_bond(21,94,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-3")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_Z-4")

M=Molecular_structure("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-1-hess.gjf")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-2")
M.rotate_bond(21,94,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-4")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-S_N-N=N_E-N1_E-N2_E-3")

M=Molecular_structure("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_Z-2-hess.gjf")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_Z-1")
M.rotate_bond(21,94,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_Z-3")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_Z-4")

M=Molecular_structure("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_E-2-hess.gjf")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_E-1")
M.rotate_bond(21,94,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_E-3")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_Z-N2_E-4")

M=Molecular_structure("AatropamSc6-R_C-R_N-N=N_E-N1_E-N2_Z-3-hess.gjf")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_E-N2_Z-1")
M.rotate_bond(21,94,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_E-N2_Z-2")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_E-N2_Z-4")

M=Molecular_structure("AatropamSc6-R_C-R_N-N=N_E-N1_E-N2_E-2-hess.gjf")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-R_C-R_N-N=N_E-N1_E-N2_E-1")
M.rotate_bond(21,94,180)
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-S_C-R_N-N=N_E-N1_E-N2_E-4")

M=Molecular_structure("AatropamSc6-S_C-R_N-N=N_E_N1_E-N2_Z-1-hess.gjf")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-S_C-R_N-N=N_E_N1_E-N2_Z-2")
M.rotate_bond(21,94,180)
M.print_new_file("AatropamSc6-S_C-R_N-N=N_E_N1_E-N2_Z-4")
M.rotate_bond(7,116,180)
M.print_new_file("AatropamSc6-S_C-R_N-N=N_E_N1_E-N2_Z-3")


M.rotate_bond(40,166,180)

M.rotate_bond(41,149,180)
M.rotate_bond(5,183,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a")
M.rotate_bond(14,132,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-B")
M.rotate_bond(40,166,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-D")
M.rotate_bond(14,132,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-C")
M.rotate_bond(40,166,180)
M.rotate_bond(41,149,180)
M.rotate_bond(5,183,180)

M.rotate_bond(5,183,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-E")
M.rotate_bond(14,132,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-EB")
M.rotate_bond(40,166,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-ED")
M.rotate_bond(14,132,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-EC")
M.rotate_bond(40,166,180)
M.rotate_bond(5,183,180)

M.rotate_bond(41,149,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-F")
M.rotate_bond(14,132,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-FB")
M.rotate_bond(40,166,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-FD")
M.rotate_bond(14,132,180)
M.print_new_file("hsmPBEh3c-S-2SiPh-a-FC")
"""
"""
file_name="atrop-R-gau2-Z_im-N=N_Z-N1_E-N2_E-1-uff.out"
new_fw_file_name=file_name.split(".")[0]+"harm_prod_025"
new_rev_file_name=file_name.split(".")[0]+"harm_sm_025"
ref_file_name=file_name.split(".")[0]+"harm_prod_ref"
M=Molecular_structure(file_name)
#M.print_pdb("atrop-R-gau2-Z_im-N=N_Z-N1_E-N2_E-1-or.pdb")
#M.generate_connections_from_distance()
#print (M.connectivity_lines_sections)
#print (M.atom_list[3].connection)
#M.groups_of_atoms_by_connection()
#for g in M.groups: print (g)
M.disconnect_atoms(83,84)
M.disconnect_atoms(3,4)
M.groups_of_atoms_by_connection()
M.route_sections[0]='#opt=(nomicro,maxstep=10)  external("ext_ref.py orca5_pbe3h3c /home/lsimon/jobs/'+ref_file_name+'.com 0.25") geom=connectivity'
MM=copy.deepcopy(M)
MMM=copy.deepcopy(M)
#for g in M.groups: print (g)
M.aply_normal_mode(atom_1=85,atom_2=87,factor=0.2)
MMM.aply_normal_mode(atom_1=85,atom_2=87,factor=0.2,red_dist=False)
#M.print_pdb("atrop-R-gau2-Z_im-N=N_Z-N1_E-N2_E-1-fw.pdb")
M.remove_atoms(M.groups[0])
MM.remove_atoms(M.groups[0])
M.print_new_file(new_fw_file_name)
MM.print_new_file(ref_file_name)
MMM.print_new_file(new_rev_file_name)

"""

"""
bridged_atoms={}
for a in M.atom_list:
    if a.symbol=="H":
        if len(a.connection)>1:
            print ("bridged H atom: "+str(a.atom_number)+" bound to: "+str(a.connection[0][0])+" and "+str(a.connection[1][0]) )
            bridged_atoms[a.atom_number]=[t for t in a.connection] #need this to copy by value, not by reference
print (bridged_atoms)

for aa  in bridged_atoms.keys():
    for b in bridged_atoms[aa]:
        print ("disconnecting "+str(aa)+" and "+str(b[0]))
        M.disconnect_atoms(aa,b[0])
M.groups_of_atoms_by_connection()
for g in M.groups: print (g)

for aa in bridged_atoms.keys():
    for b in bridged_atoms[aa]:
        for i in range (0,len(M.groups)):
            if b[0] in M.groups[i]: b.append(i)
print (bridged_atoms)

for i in range (0,len(M.groups)):
    counter=0
    for aa  in bridged_atoms.keys():
        for b in bridged_atoms[aa]:
            if b[0] in M.groups[i]: counter+=1; continue
    if counter>1: 
        print("pswitch group identified:"+str(i))
        for aa in bridged_atoms.keys():
            for b in bridged_atoms[aa]:
                if b[3]==i:
                    print ("h-bridge atom: "+str(aa)+" with atom:"+str(b[0]))

"""





"""
M=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/product/atrop-PROD-Z_im-gau2-N·N+_N1_Z-N2_Z.out")
M.print_pdb("reference.pdb")
M.rotate_bond(9,5,-15)
M.rotate_bond(5,10,15)
M.rotate_bond(10,17,-15)
M.rotate_bond(17,24,15)
M.rotate_bond(24,15,-15)
M.rotate_bond(15,9,15)

M.print_pdb("reference2.pdb")
"""

"""
M21=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/atrop-cat-anion-2.out")
M21.print_pdb("cat_anion.pdb")
newM21,rmsd,_=M21.match_numbers_to_molecule(M)
newM21.print_pdb("rotated-cat_anion.pdb")


M22=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/old/atrop-PROD-prot-Z-im-N=N_E-N1_Z-N2_E.out")
M22.print_pdb("product.pdb")
newM22,rmsd,_=M22.match_numbers_to_molecule(M)
newM22.print_pdb("rotated-product.pdb")
"""




"""
M21.set_fingerprints(n-1)
M21.renumber_by_fingerprints(M)
M21.print_pdb("cat00-before.pdb")
rmsd,fragment_centroid,rotation,translation,fragment_coords=M.min_rmsd(M21,M.groups[0])
print ("[0,0]: "+str(rmsd))
M21.update_coords(fragment_coords)
M21.print_pdb("cat00.pdb")
M21.print_xyz("cat00.xyz")

M22=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/atrop-cat-anion-2.out")
M22.set_fingerprints(n-1,[0,1])
M22.renumber_by_fingerprints(M)
rmsd,fragment_centroid,rotation,translation,fragment_coords=M.min_rmsd(M22,M.groups[0])
print ("[0,1]: "+str(rmsd))
M22.update_coords(fragment_coords)
M22.print_pdb("cat01.pdb")
M22.print_xyz("cat01.xyz")

M23=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/atrop-cat-anion-2.out")
M23.set_fingerprints(n-1,[1,0])
M23.renumber_by_fingerprints(M)
rmsd,fragment_centroid,rotation,translation,fragment_coords=M.min_rmsd(M23,M.groups[0])
print ("[1,0]: "+str(rmsd))
M23.update_coords(fragment_coords)
M23.print_pdb("cat10.pdb")
M23.print_xyz("cat10.xyz")

M24=Molecular_structure("/Users/luissimon/calculos/atrop/UFF/fragments/atrop-cat-anion-2.out")
M24.set_fingerprints(n-1,[1,1])
M24.reorder_by_fingerprints(M)
rmsd,fragment_centroid,rotation,translation,fragment_coords=M.min_rmsd(M24,M.groups[0])
print ("[1,1]: "+str(rmsd))
M24.update_coords(fragment_coords)
M24.print_pdb("cat11.pdb")
M24.print_xyz("cat11.xyz")
"""

"""
for m in Ms:
    filename=m.output_file.split("/")[-1].split(".out")[0]
    m.print_pdb(filename+"before.pdb") 
    m.set_fingerprints(n-1)
    m.renumber_by_fingerprints(M)
    rmsd,fragment_centroid,rotation,translation,fragment_coords=M.min_rmsd(m,M.groups[3])
    m.update_coords(fragment_coords)
    print (m.output_file+": "+str(rmsd))
    
    m.print_pdb(filename+".pdb")
    m.center_of_mass()
"""

"""
print (M15.atom(1).fingerprint)
print (M15.atom(2).fingerprint)
print (M15.atom(9).fingerprint)
print (M15.atom(10).fingerprint)
print (M15.atom(12).fingerprint)
print (M15.atom(14).fingerprint)

print

print (M.atom(5).fingerprint)
print (M.atom(6).fingerprint)

print 
print (M.atom(84).fingerprint)
print (M.atom(85).fingerprint)


print "now with M11"
n=len(M.atom(84).fingerprint)
M11.set_fingerprints(n-1)

filename=M11.output_file.split("/")[-1].split(".out")[0]
M11.print_pdb(filename+".pdb")
print (M11.atom(11).fingerprint)
print (M11.atom(12).fingerprint)
print ("on the other side")
print (M11.atom(14).fingerprint)
print (M11.atom(15).fingerprint)
print (M11.atom(16).fingerprint)
"""




"""
M2.set_fingerprints(n-1)
print M2.atom(1).fingerprint
print "***"
i=0
for g in M.groups:
    print str(i)+":" 
    print g
    i+=1
rmsd,fragment_centroid,rotation,translation,fragment_coords = M.min_rmsd(M2,M.groups[1])

print rmsd
print fragment_centroid
print rotation
print fragment_coords

M2.print_xyz("test10.xyz")
M2.update_coords(fragment_coords)

M2.print_xyz("test10_rotated.xyz")
M.print_xyz("test9.xyz")
"""


#M3=Molecular_structure("test11.out")
#print ("test11 ok")


#M2.renumber_by_fingerprints(M)
#print (M2.print_xyz())

"""
for gr in M.groups:
    print gr
M.connect_atoms(94,97)
print("**********************************")
M.groups_of_atoms_by_connection()
for gr in M.groups:
    print gr
print("**********************************")
M.transfer_atom_from_groups(0,1,order=1.0)
M.groups_of_atoms_by_connection()
for gr in M.groups:
    print gr
"""



#print (M.print_orca())



#M2.renumber_by_fingerprints(M)
#print(M2.atom(157).connection)


#print (M.generate_fingerprints())
#M3=Molecular_structure("rnase4-scan.out","first")
#print (M3.G16_output.energies)
#for c in M3.G16_output.cart_coordinates: print len(c)
"""
M3=Molecular_structure("test-opt.out","test-opt.com")

print(M3.bonds_between(3))
M3.rotatebond(45,40,90)
r=open("rotated.com","w")
r.write(M3.print_new_file())
r.close()
M3.set_fingerprints()
M3.DFS_rings()
#print(M3.route_sections)
#print(M3.print_new_file())
#print(M3.distance([1,2]))
#print(M3.angle([1,2,3]))
#print(M3.angle([11,2,1,3]))
#M3.set_fingerprints()
#print(M3.atom(157).connection)
#M3.generate_connections_from_distance()
#print(M3.print_connectivity())
#print (M3.G16_output.get_smaller_rms_point())
#print (M3.atom_list)
#for atom in M3.atom_list:
#    print(atom.print_atom())

#print(M3.atom(157).connection)   
#M3.generate_connections_from_distance()
#print(M3.atom(157).connection)
#print(M3.atom(157).print_atom())
#print(M3.atom(157).connection)
#print(M3.print_connectivity())
#print(M3.print_new_file() )

#print (M.atom_list[0].coord)

#print
#print (M.atom(1).print_atom())

#print (M.atom(1).print_atom())
#print (M.angle([1,2,3,4]))
#print (M.angle([2,1,3]))
#print (M.distance([2,1]))
#print (len(groups))
#M.refine_fingerprints()
#for a in M.atom_list:
#    print (a.fingerprint[-2])
#    print (str(len(a.fingerprint))+"th element:  "+str(a.fingerprint[-1]))



#print (M.print_connectivity() )


name="atrop-R-anti-E_im-N=N_E-N1_E-N2_E-2"
M=Molecular_structure("atropuff-model.com",name+".out","last")
with open(name+"-uff.com","w") as f : f.write (M.print_new_file())



"""



#M1=Molecular_structure("/Users/luissimon/calculos/hosomi-sakurai-list/hsm2Si/hsm-S-2SiPh-g2+FD.out")


  

#M1.lotHL_sections=['external="ext_orca.py orca_pbe3h3c"']
#M1.opt_sections=["modreduntant","maxstep=12","nomicro"]
#M1.route_sections=['# geom=connectivity freq=noraman oniom(external="ext_orca.py orca_pbe3h3c":uff) opt=(modredundant,nomicro,maxstep=12)']
#M1.modred_sections=[" 89 99 S 10 +0.02"]
#print (M1.print_new_file())
