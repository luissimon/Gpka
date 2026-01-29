#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
# script for extracting atomic properties 

import string
import os
import os.path
import sys
sys.path.append('../import')
import copy
import Molecular_structure
print(Molecular_structure.__file__)
import numpy as np
import pandas as pd
import json
import networkx as nx
import spektral

import repeated_molecules
from routes import root_route
from routes import extracted_data_route
from routes import output_files_route
from routes import labels_csv_file_name

from routes import sampl_output_files_route
from routes import sampl_labels_csv_file_name
from routes import sampl_extracted_data_route

#dictionaries to transform the names of descriptors in this script to the names used in publication
#translating them consumes some time but also allows to change the name of the descriptors without further changes in the code
#fomat: script name is key, publication name is value
from feature_names import atom_features_publication_names
from feature_names import vector_features_publication_names
#fomat: pubilcation name is key, script name is value
from feature_names import inv_atom_features_publication_names
from feature_names import inv_vector_features_publication_names   

import json
from NpEncoder import NpEncoder


np.set_printoptions(precision=4,suppress=True,sign=" ") 


#READ OPTIONS FROM COMMAND LINE
if "-weighting" in sys.argv:  weighting=sys.argv[sys.argv.index("-weighting")+1]
else: weighting="gibbs" # "gibbs", "sp", or "zero"
if "-lot" in sys.argv:  level_of_theory=sys.argv[sys.argv.index("-lot")+1]
else: level_of_theory="sM06" # "M06","sM06","swb97xd","wb97xd","pbeh3c"
if "-opt_lot" in sys.argv:  optimization_level_of_theory=sys.argv[sys.argv.index("-opt_lot")+1]
else: optimization_level_of_theory="pbeh3c"
if "-destination" in sys.argv: destination=sys.argv[sys.argv.index("-destination")+1]+"/"
else: destination="./"
if "-sampl" in sys.argv: do_sampl=True
else: do_sampl=False
if "-interactive"   in sys.argv: do_interactive=True
else: do_interactive=False
if "-orca_version" in sys.argv: orca_version=sys.argv[sys.argv.index("-orca_version")+1]
else: orca_version="5"


#CHECK COMMAND LINE OPTIONS AND ASSING VALUES
if weighting not in ["gibbs","sp","zero"]: print ("weighting should be gibbs, sp, or zero"); sys.exit()
if level_of_theory not in ["M06","sM06","swb97xd","wb97xd","pbeh3c","wb97x3c","r2scan3c"]: print ("lot must be M06, sM06, swb97xd, wb97xd, pbeh3c, wb97x3c, or r2scan3c"); sys.exit()
if optimization_level_of_theory not in ["b973c","pbeh3c"]: print ("lot must be b973c or pbeh3c"); sys.exit()

if level_of_theory=="wb97xd": HL_text="_wb97xd_chrg"
elif level_of_theory=="swb97xd": HL_text="_swb97xd_chrg"
elif level_of_theory=="sM06": HL_text="_sm06_chrg"
elif level_of_theory=="M06": HL_text="_m06_chrg"
elif level_of_theory=="pbeh3c": HL_text="_pbeh3c_chrg"
elif level_of_theory=="wb97x3c": HL_text="_wb97x3c_chrg"
elif level_of_theory=="r2scan3c": HL_text="_r2scan3c_chrg"


#the json file where results will be deposited:
if "-out" in sys.argv: output_suffix=sys.argv[sys.argv.index("-out")+1]  
else:
    from datetime import datetime 
    output_suffix=str(datetime.today().year)[2:]  #"25"

if optimization_level_of_theory=="pbeh3c":
    graphs_extracted_file=destination+"molecular_graphs-"+weighting+"-"+level_of_theory+"."+output_suffix+".json"
elif optimization_level_of_theory=="b973c":
    graphs_extracted_file=destination+"molecular_graphs-"+weighting+"-b973c-"+level_of_theory+"."+output_suffix+".json"

#find out the entries that will be processed, filtering out those already done
already_done=[]
if graphs_extracted_file in os.listdir(): 
    if do_interactive:
        print ("file for output exist; (c)ontinue or (o)verwrite?") 
        cont=input()
    else: 
        cont="o"                                                   
    if cont=="c" or cont=="C":
        with open (graphs_extracted_file,"r") as f: lines=f.readlines()
        for l in lines[1:]: already_done.append(json.loads(l)["name"])
        write_keys_list=False
    else: 
        os.system("rm -f "+graphs_extracted_file)


exclude=[]
if "-exclude" in sys.argv: 
    if sys.argv[sys.argv.index("-exclude")+1] in os.listdir(os.getcwd()): 
        with open (sys.argv[sys.argv.index("-exclude")+1],"r") as f: exclude_lines=f.readlines()
        for l in exclude_lines: exclude.append(l.strip())
    else: 
        exclude = sys.argv[sys.argv.index("-exclude")+1].split(",") 

print(exclude)


#ROUTES OF FILES DEPENDING ON OPTIONS
if do_sampl:
    output_files_route=sampl_output_files_route
    labels_csv_file_name=sampl_labels_csv_file_name
    graphs_extracted_file=destination+"molecular_graphs-sampl-"+weighting+"-"+level_of_theory+"."+output_suffix+".json" 
    extracted_data_route=sampl_extracted_data_route


if optimization_level_of_theory=="b973c": 
    output_files_route+="b973c_optimized/"
elif optimization_level_of_theory=="pbeh3c": 
    output_files_route+="PBEh3c_optimized/"
    
optimization_files_route=output_files_route+"optimization/"
HL_route= output_files_route+"SP-"+level_of_theory+"/"
nbo_route=output_files_route+"nbo/"+level_of_theory+"/"


labels=pd.read_csv(extracted_data_route+labels_csv_file_name,encoding='unicode_escape')
labels.set_index("compn",inplace=True)
labels.dropna(how='all', axis=1, inplace=True)
print (labels.head()) #for debuggin...

#read from command line on which entries the script must work
if not any([":" in s for s in sys.argv]): rows=labels.index
else:
        arg_index=[":" in s for s in sys.argv].index(True) 
        if sys.argv[arg_index].startswith(":"): start=0;end=sys.argv[arg_index].split(":")[1]
        elif sys.argv[arg_index].endswith(":"): end=len(labels.index)+1; start=sys.argv[arg_index].split(":")[0]
        else: start,end=sys.argv[arg_index].split(":")[0],sys.argv[arg_index].split(":")[1]
        rows=labels.index[int(start):int(end)]
        print ("will start ad index: "+str(start)+" and end at index: "+str(end))



number_of_errors=0

#list of features that are not assigned to each atom (although they are a list)
not_atomic_properties=["FASA_mol_electronic_spatial_extent_components","FASA_molecular_dipole_moment","FASA_mol_electronic_spatial_extent_components"]


#some conversion factors
hartrees_to_kal_mol=627.5095
RT=0.594  # cal/kmol(298 K)


#auxiliary methods
def get_projected_molecular_dipole(molecule):

    n_atoms=len(molecule.atom_list)
    dipole_moment=np.array(molecule.QM_output.dipole_moments[0],dtype=np.float32)
    diagonal_quadropole_moment_nuc=np.array(molecule.properties["quadrupole_moment"][1][1:4], dtype=np.float32)
    diagonal_quadropole_moment_el=np.array(molecule.properties["quadrupole_moment"][1][1:4], dtype=np.float32)
    diagonal_quadropole_moment_tot=np.array(molecule.properties["quadrupole_moment"][2][1:4], dtype=np.float32)
    polarizability_matrix=np.array(molecule.properties["polarizability_matrix"],dtype=np.float32)
    bond_dipoles=np.zeros((n_atoms,n_atoms),dtype=np.float32)
    bond_polarizabilities=np.zeros((n_atoms,n_atoms),dtype=np.float32)
    bond_e_extent=np.zeros((n_atoms,n_atoms),dtype=np.float32)
    bond_chg_extent=np.zeros((n_atoms,n_atoms),dtype=np.float32)

    #calculate the projection of the molecular dipole over unitary vectors (v) which direction is each bond in the molecule and
    #the projection of the polarizability matrix:    v @ P @ v.T : the response to an electric field with the direction of each bond projected on each bond
    for i,a in enumerate(molecule.atom_list):
        for c in a.connection:   #the projection will be 0 if the atoms are not bound (it will only be searched in the connections of each atom)
            unitary_vector=molecule.atom(c[0]).coord-a.coord
            unitary_vector=unitary_vector/np.linalg.norm(unitary_vector)
            bond_dipoles[i,c[0]-1]=unitary_vector.dot(dipole_moment)
            bond_e_extent[i,c[0]-1]=unitary_vector.dot(diagonal_quadropole_moment_el)
            bond_chg_extent[i,c[0]-1]=unitary_vector.dot(diagonal_quadropole_moment_tot)
            bond_polarizabilities[i,c[0]-1]= unitary_vector.dot( polarizability_matrix@np.transpose(unitary_vector) ) #bond polarizability with respect to a distortion with the same direction than the bond 


    #calculate  v @ P @ vH.T (vH are the unitary vectors corresponding to X-H bonds)
    #it represents the response to an electric field with the direction of each X-H bond, and this reponse is projected over each bond of the molecule.
    #This is a tensor with as many matrix elements as atoms, but is zero for H atoms or atoms not bound to H. 
    #The idea is to keep only the matrix corresponding to the bond that is broken during deprotonation(using the mask to filter it), so it will represent the 
    #projection on each molecule bond when an electric field with the direction of the scissible X-H bond is applied.
    #First, find X-H bonds to project on it the polarizability matrix
    bond_polarizabilities_wr2_H_bonds=[]
    #iterate through all atoms searching for X-H bonds:
    for j,a in enumerate(molecule.atom_list):
        projected_polarizability=np.zeros(3)
        for c in a.connection:
            if molecule.atom(c[0]).symbol.lower()=="h":
                bond_H_unitary_vector=molecule.atom(c[0]).coord-a.coord
                bond_H_unitary_vector=bond_H_unitary_vector/np.linalg.norm(bond_H_unitary_vector)  
                bond_H_unitary_vector_transp=np.transpose(bond_H_unitary_vector)
                projected_polarizability+=polarizability_matrix@bond_H_unitary_vector_transp #sum all contributions of X-H bonds for atom X
        #now calculate the projection the projected_polarizability on each bond
        bond_polarizability_wr2_H_bonds=np.zeros((n_atoms,n_atoms))
        for i,aa in enumerate(molecule.atom_list):
            for cc in aa.connection:  #it will only be done on bound atoms
                unitary_vector=molecule.atom(cc[0]).coord-aa.coord
                unitary_vector=unitary_vector/np.linalg.norm(unitary_vector)
                #bond_polarizability_wr2_H_bonds is a n_atoms x n_atoms matrix; element i,k represents the projection 
                #of the induced dipole moment on the bond between i,k atoms
                bond_polarizability_wr2_H_bonds[i,cc[0]-1]=unitary_vector.dot(projected_polarizability)
        #bond_polarizabilities_wr2_H_bonds is a tensor. The outer dimension (number of atoms) refers to the direction of the electric field, so
        # ith element represent the reaction to an electric field in the direction of the H atoms bound to atom i.
        # the two inner dimensions (number of atoms x number of atoms) refer to the bonds in the molecule, so the i,j,k element represents the induced dipole moment by
        # the electric field in the direction of the bonds of atom i with H atoms, projected on the bond formed by atoms j and k (it is 0 if j and k are not bound)
        bond_polarizabilities_wr2_H_bonds.append(bond_polarizability_wr2_H_bonds)

    return bond_dipoles,bond_polarizabilities,bond_polarizabilities_wr2_H_bonds,bond_e_extent,bond_chg_extent

def get_force_constant_matrix(molecule,relative=False):

    standard_hx_force_constants={"o": -0.484897971,"n": -0.438624438,"s":-0.281700975,"c":-0.343738615,"f":-0.583684055,
                                "cl":-0.328733906,"br":-0.284086875,"i":-0.219412699,"p":-0.226285462,"si":-0.18723526 }

    force_constant_matrix=np.zeros((len(molecule.atom_list),len(molecule.atom_list)))
    for i,a in enumerate(molecule.atom_list):
        for j in range(i+1,len(molecule.atom_list)):
            if molecule.distance_rcov_ratio([i+1,j+1] )<15:
                hess=molecule.cart_hess[(i)*3:(i+1)*3,(j)*3:(j+1)*3]
                unitary_vector=molecule.atom_list[j].coord-molecule.atom_list[i].coord
                unitary_vector=unitary_vector/np.linalg.norm(unitary_vector)
                eig,v=np.linalg.eig(hess)
                fc=np.real(eig.dot( abs( (unitary_vector.dot(v)).T ) ) )
                if relative==True: 
                    if molecule.atom_list[i].symbol.lower()=="h" and molecule.atom_list[j].symbol.lower() in  standard_hx_force_constants.keys():
                        fc=fc/standard_hx_force_constants[molecule.atom_list[j].symbol.lower()]           
                    if molecule.atom_list[j].symbol.lower()=="h" and molecule.atom_list[i].symbol.lower() in  standard_hx_force_constants.keys():
                        fc=fc/standard_hx_force_constants[molecule.atom_list[i].symbol.lower()]
                    if molecule.atom_list[j].symbol.lower()!="h" and  molecule.atom_list[i].symbol.lower()!="h": #it only has sense with X-H bonds
                        fc=0.0
                    if molecule.atom_list[i].symbol.lower()=="h" and molecule.atom_list[j].symbol.lower() not in  standard_hx_force_constants.keys():
                        fc=0.0
                    if molecule.atom_list[j].symbol.lower()=="h" and molecule.atom_list[i].symbol.lower() not in  standard_hx_force_constants.keys():
                        fc=0.0
                force_constant_matrix[i,j],force_constant_matrix[j,i]= fc,fc 

    return np.array(force_constant_matrix)

def get_chemical_shifts(molecule,relative=False):

    if level_of_theory=="swb97xd":
        standard_chemical_shields={"o": 29.871    ,"n": 31.414     ,"s": 30.379     ,"c":   31.475   ,"p":  29.567 }
    elif level_of_theory=="wb97xd":
        standard_chemical_shields={"o": 29.7465   ,"n": 31.306     ,"s": 30.069     ,"c":   31.439   ,"p":  29.299 }
    elif level_of_theory=="M06":
        standard_chemical_shields={"o": 29.2885   ,"n": 31.009     ,"s": 30.121     ,"c":   31.368   ,"p":  29.296 }
    elif level_of_theory=="sM06":
        standard_chemical_shields={"o": 29.719    ,"n": 31.209     ,"s": 30.3185    ,"c":   31.380   ,"p":  29.482 }
    elif level_of_theory=="pbeh3c":
        standard_chemical_shields={"o": 30.757    ,"n": 32.440     ,"s":  31.077    ,"c":   31.9775  ,"p":  29.981 }
    elif level_of_theory=="r2scan3c":
        standard_chemical_shields={"o": 30.1345   ,"n":  31.6753   ,"s":  30.1115   ,"c":   31.6725  ,"p":  29.485 }
    elif level_of_theory=="wb97x3c":
        standard_chemical_shields={"o": 29.5185   ,"n": 31.3773    ,"s":  30.1415   ,"c":   31.371   ,"p":  29.643 }

    if relative==False: 
        return [a.chemical_isotropic_shield for a in molecule.atom_list]
    else:
        chemical_shield=[]
        for a in molecule.atom_list:
            if a.symbol.lower()!="h": chemical_shield.append(0.0)
            else: 
                nearest_atom=molecule.nearest_atom(a,exclude_H=False)
                if nearest_atom.symbol.lower() in standard_chemical_shields.keys():  
                    #chemical_shield.append(a.chemical_isotropic_shield/standard_chemical_shields[nearest_atom.symbol.lower()]) 
                    chemical_shield.append(a.chemical_isotropic_shield-standard_chemical_shields[nearest_atom.symbol.lower()])
                else: chemical_shield.append(0.0)
    return chemical_shield

def get_relative_chg( prop,molecule, chg="hirshfeld"):

    if level_of_theory=="wb97xd":
        standard_charge=   {'nbo_charges': {'o': 0.49883,'n': 0.37201,'s': 0.169545,'c': 0.216585,'p': 0.015260000000000001},
                            'nbo_wiberg_bond_orders': {'o': 0.7528,'n': 0.8641,'s': 0.7528,'c': 0.7528,'p': 0.9935},
                            'nbo_nbi_bond_orders': {'o': 0.86765,'n': 0.9295,'s': 0.86765,'c': 0.86765,'p': 0.9968},
                            'chg_hirshfeld': {'o': 0.1725863139,'n': 0.1024221241,'s': 0.06791736379999999,'c': 0.035183041799999995,'p': -0.0028007566000000004},
                            'chg_voronoy': {'o': 0.17202549445,'n': 0.0845488378,'s': 0.050787776800000003,'c': 0.016594933025,'p': -0.014473655566666667},
                            'chg_mulliken': {'o': 0.24152026555,'n': 0.1199831695,'s': 0.20404007985,'c': 0.19101352885,'p': 0.10266966746666667},
                            'chg_lowdin': {'o': -0.11006040235,'n': -0.0929007778,'s': -0.1840488167,'c': -0.019478534125,'p': -0.12130745496666667},
                            'chg_becke': {'o': 0.4321113305,'n': 0.3739296238,'s': 0.1587268855,'c': 0.08661472567499999,'p': 0.10519333609999999},
                            'chg_ADCH': {'o': 0.4319213384,'n': 0.3738931674,'s': 0.15870532664999998,'c': 0.10043777079999999,'p': 0.105195696},
                            'chg_CHELPG': {'o': 0.4370904984,'n': 0.3715590166,'s': 0.17127069,'c': 0.10727942134999999,'p': 0.092598145},
                            'chg_MK': {'o': 0.43542340845000005,'n': 0.3703189449,'s': 0.17672060905,'c': 0.13856654675000002,'p': 0.09682439463333332},
                            'chg_CM5': {'o': 0.3393364353,'n': 0.2828386456,'s': 0.1327481476,'c': 0.083242792525,'p': 0.0774737913},
                            'chg_12CM5': {'o': 0.4072045011,'n': 0.339406373,'s': 0.1592980366,'c': 0.099891422275,'p': 0.0929685805},
                            'chg_RESP': {'o': 0.43531845565,'n': 0.3701705139,'s': 0.17665694955,'c': 0.1349474418,'p': 0.0967564852},
                            'chg_PEOE': {'o': 0.2057302845,'n': 0.1146232457,'s': 0.09844511,'c': 0.0194090064,'p': 0.0511157583},
                            'bo_mayer': {'o': 0.9579397350000001,'n': 1.00622828,'s': 0.959181525,'c': 0.958419705,'p': 0.9840430333333333},
                            'bo_wiberg': {'o': 1.47048409,'n': 1.24389768,'s': 1.452707965,'c': 1.0101066825,'p': 1.2105345099999998},
                            'bo_mulliken': {'o': 0.547459175,'n': 0.57686189,'s': 0.68378313,'c': 0.770127225,'p': 0.7419388466666667},
                            'bo_fuzzy': {'o': 0.947374575,'n': 0.97477589,'s': 1.101478555,'c': 0.959470505,'p': 1.0346146366666666},
                            'bo_laplacian': {'o': 0.588106335,'n': 0.80299111,'s': 0.83899552,'c': 0.916348085,'p': 0.8479969799999999},
                            'bo_IBSI': {'o': 1.689635,'n': 1.49291,'s': 0.8854500000000001,'c': 1.204155,'p': 0.72317},
                            'v_at_nucleus': {'o': -616.6619000000001,'n': -663.3706,'s': -631.02265,'c': -705.199625,'p': -671.6128666666667},
                        }

    elif level_of_theory=="swb97xd":
        standard_charge=   {'nbo_charges': {'o': 0.514205,'n': 0.39446,'s': 0.20133,'c': 0.22870000000000001,'p': 0.03818666666666667},
                            'nbo_wiberg_bond_orders': {'o': 0.73775,'n': 0.8488,'s': 0.73775,'c': 0.73775,'p': 0.9937999999999999},
                            'nbo_nbi_bond_orders': {'o': 0.8589,'n': 0.9213,'s': 0.8589,'c': 0.8589,'p': 0.9969},
                            'chg_hirshfeld': {'o': 0.17140098264999998,'n': 0.100787386,'s': 0.06800739580000001,'c': 0.033157719749999995,'p': -0.004217904133333333},
                            'chg_voronoy': {'o': 0.17106733305,'n': 0.0813914429,'s': 0.0541260749,'c': 0.0120423579,'p': -0.015627346900000002},
                            'chg_mulliken': {'o': 0.1884347716,'n': 0.020701852,'s': 0.20654445295,'c': -0.032302276825000004,'p': 0.10050543113333332},
                            'chg_lowdin': {'o': 0.02279972635,'n': -0.0288275325,'s': -0.17520213815,'c': -0.0054768921,'p': -0.15087846653333334},
                            'chg_becke': {'o': 0.43795799874999997,'n': 0.3764778339,'s': 0.16787927015,'c': 0.081637949375,'p': 0.1118346376},
                            'chg_ADCH': {'o': 0.43772293075,'n': 0.3764538683,'s': 0.1678340636,'c': 0.096871895675,'p': 0.11182524919999999},
                            'chg_CHELPG': {'o': 0.44167791884999996,'n': 0.3742680649,'s': 0.18305080675000002,'c': 0.0861691261,'p': 0.10174804326666666},
                            'chg_MK': {'o': 0.44086090460000005,'n': 0.3738026011,'s': 0.18761083364999998,'c': 0.1179070139,'p': 0.1045490584},
                            'chg_CM5': {'o': 0.3381507937,'n': 0.2812030591,'s': 0.13283899995,'c': 0.08121696172500001,'p': 0.07605634859999999},
                            'chg_12CM5': {'o': 0.40578157735,'n': 0.3374433519,'s': 0.159407463,'c': 0.0974602172,'p': 0.09126758156666666},
                            'chg_RESP': {'o': 0.44075593495,'n': 0.3736541589,'s': 0.18754690625,'c': 0.1143139945,'p': 0.10448063006666668},
                            'chg_PEOE': {'o': 0.2057302845,'n': 0.1146232457,'s': 0.09844511,'c': 0.0194090064,'p': 0.0511157583},
                            'bo_mayer': {'o': 1.008377185,'n': 0.97685336,'s': 1.010858845,'c': 0.982458305,'p': 1.0746831666666667},
                            'bo_wiberg': {'o': 1.365635555,'n': 1.22268994,'s': 1.471713475,'c': 1.0038537225,'p': 1.21444271},
                            'bo_mulliken': {'o': 0.67977255,'n': 0.13306099,'s': 0.768183965,'c': 0.7970229125000001,'p': 0.8566823266666667},
                            'bo_fuzzy': {'o': 0.9494265,'n': 0.97743699,'s': 1.102003705,'c': 0.9619661125000001,'p': 1.03557185},
                            'bo_laplacian': {'o': 0.66925256,'n': 0.92836282,'s': 0.92204453,'c': 1.0286782700000001,'p': 0.9303369333333333},
                            'bo_IBSI': {'o': 1.689635,'n': 1.49291,'s': 0.8854500000000001,'c': 1.204155,'p': 0.72317},
                            'v_at_nucleus': {'o': -605.73595,'n': -654.5955,'s': -620.38755,'c': -696.86455,'p': -664.6113333333334},
                            }

    elif level_of_theory=="M06":
        standard_charge=   {'nbo_charges': {'o': 0.50336,'n': 0.37276,'s': 0.1661,'c': 0.21522000000000002,'p': 0.012586666666666664},
                            'nbo_wiberg_bond_orders': {'o': 0.7482,'n': 0.8634,'s': 0.7482,'c': 0.7482,'p': 0.9937999999999999},
                            'nbo_nbi_bond_orders': {'o': 0.86495,'n': 0.9292,'s': 0.86495,'c': 0.86495,'p': 0.9969},
                            'chg_hirshfeld': {'o': 0.17331986545,'n': 0.10093287,'s': 0.06449352625,'c': 0.033971035999999996,'p': -0.0054867878},
                            'chg_voronoy': {'o': 0.17261975395,'n': 0.0830280308,'s': 0.0460573435,'c': 0.015627289250000002,'p': -0.018210788699999998},
                            'chg_mulliken': {'o': 0.32032757805,'n': 0.209094652,'s': 0.20052566015,'c': 0.202639978125,'p': 0.08950893433333333},
                            'chg_lowdin': {'o': -0.10726837659999999,'n': -0.0919646654,'s': -0.1881296834,'c': -0.01928508285,'p': -0.12530034356666667},
                            'chg_becke': {'o': 0.4336866693,'n': 0.3669306579,'s': 0.1519521747,'c': 0.08416353215,'p': 0.09391960016666667},
                            'chg_ADCH': {'o': 0.43352492135,'n': 0.3668967178,'s': 0.15193175675,'c': 0.09822651225,'p': 0.09392166623333333},
                            'chg_CHELPG': {'o': 0.43891924685,'n': 0.3656295445,'s': 0.1647269378,'c': 0.10424605077499999,'p': 0.0816843728},
                            'chg_MK': {'o': 0.43711933665,'n': 0.3637850557,'s': 0.17038234325,'c': 0.135534974925,'p': 0.08584806796666666},
                            'chg_CM5': {'o': 0.3400696976,'n': 0.2813493877,'s': 0.12932431215,'c': 0.082030797075,'p': 0.0747877619},
                            'chg_12CM5': {'o': 0.408084276,'n': 0.3376192621,'s': 0.1551894284,'c': 0.098437030875,'p': 0.08974534449999999},
                            'chg_RESP': {'o': 0.43701437855,'n': 0.3636366463,'s': 0.17031886195,'c': 0.1319187706,'p': 0.08578111853333333},
                            'chg_PEOE': {'o': 0.2057302845,'n': 0.1146232457,'s': 0.09844511,'c': 0.0194090064,'p': 0.0511157583},
                            'bo_mayer': {'o': 0.90130771,'n': 0.95001413,'s': 0.9562266500000001,'c': 0.9498424050000001,'p': 0.9770021566666666},
                            'bo_wiberg': {'o': 1.4697270150000001,'n': 1.24380053,'s': 1.456133675,'c': 1.009915745,'p': 1.21237528},
                            'bo_mulliken': {'o': 0.55028609,'n': 0.55521687,'s': 0.66608068,'c': 0.767807245,'p': 0.7168581233333334},
                            'bo_fuzzy': {'o': 0.9456811700000001,'n': 0.97538396,'s': 1.10379687,'c': 0.9595951749999999,'p': 1.03580098},
                            'bo_laplacian': {'o': 0.54798705,'n': 0.77156567,'s': 0.83096226,'c': 0.9032477950000001,'p': 0.8565219766666669},
                            'bo_IBSI': {'o': 1.689635,'n': 1.49291,'s': 0.8854500000000001,'c': 1.204155,'p': 0.72317},
                            'v_at_nucleus': {'o': -611.7636,'n': -660.5866,'s': -630.46865,'c': -704.8297749999999,'p': -672.1591}
                            }

    elif level_of_theory=="sM06":
        standard_charge=   {'nbo_charges': {'o': 0.5171600000000001,'n': 0.39541,'s': 0.19881,'c': 0.22926000000000002,'p': 0.033253333333333336},
                            'nbo_wiberg_bond_orders': {'o': 0.7346,'n': 0.848,'s': 0.7346,'c': 0.7346,'p': 0.9944000000000001},
                            'nbo_nbi_bond_orders': {'o': 0.8571,'n': 0.9209,'s': 0.8571,'c': 0.8571,'p': 0.9972},
                            'chg_hirshfeld': {'o': 0.17243224140000002,'n': 0.1001908683,'s': 0.06561705975,'c': 0.032467605525,'p': -0.007793036566666667},
                            'chg_voronoy': {'o': 0.17165919455,'n': 0.079998794,'s': 0.04945623395,'c': 0.010862455125,'p': -0.020582458833333334},
                            'chg_mulliken': {'o': 0.19958019555,'n': 0.0215904832,'s': 0.2112970002,'c': -0.0328017205,'p': 0.10720634493333335},
                            'chg_lowdin': {'o': 0.0253578338,'n': -0.0294629137,'s': -0.1794578005,'c': -0.007410244775,'p': -0.15683254970000002},
                            'chg_becke': {'o': 0.4369981652,'n': 0.3678711978,'s': 0.1632218459,'c': 0.08073846374999999,'p': 0.09904392396666667},
                            'chg_ADCH': {'o': 0.43677742465,'n': 0.367847952,'s': 0.16317446195000002,'c': 0.095905083325,'p': 0.09903457216666667},
                            'chg_CHELPG': {'o': 0.4408647612,'n': 0.3668510433,'s': 0.1787528306,'c': 0.08784972225000001,'p': 0.08950897116666667},
                            'chg_MK': {'o': 0.43993784525,'n': 0.3657577738,'s': 0.1834283468,'c': 0.120484687375,'p': 0.09206345733333332},
                            'chg_CM5': {'o': 0.33918194135,'n': 0.2806065452,'s': 0.13044870495,'c': 0.08052684897500001,'p': 0.07248121516666667},
                            'chg_12CM5': {'o': 0.40701890220000003,'n': 0.3367275374,'s': 0.1565391169,'c': 0.09663208367500001,'p': 0.08697742293333333},
                            'chg_RESP': {'o': 0.43983287839999996,'n': 0.3656093576,'s': 0.1833645171,'c': 0.1168884989,'p': 0.09199592713333334},
                            'chg_PEOE': {'o': 0.2057302845,'n': 0.1146232457,'s': 0.09844511,'c': 0.0194090064,'p': 0.0511157583},
                            'bo_mayer': {'o': 0.9965518,'n': 0.978011,'s': 1.00074018,'c': 0.9821233374999999,'p': 1.0609337266666667},
                            'bo_wiberg': {'o': 1.363235545,'n': 1.2226635,'s': 1.47483657,'c': 1.0037474675,'p': 1.2163574166666666},
                            'bo_mulliken': {'o': 0.663194445,'n': 0.15956516,'s': 0.75632138,'c': 0.7975682574999999,'p': 0.83985747},
                            'bo_fuzzy': {'o': 0.947957735,'n': 0.97781646,'s': 1.1036871000000001,'c': 0.962233675,'p': 1.0370240266666666},
                            'bo_laplacian': {'o': 0.6587043349999999,'n': 0.92622183,'s': 0.92694922,'c': 1.0263663225,'p': 0.9306397299999999},
                            'bo_IBSI': {'o': 1.689635,'n': 1.49291,'s': 0.8854500000000001,'c': 1.204155,'p': 0.72317},
                            'v_at_nucleus': {'o': -603.14645,'n': -653.4269,'s': -619.6114,'c': -695.6282500000001,'p': -664.958},
                            }

    elif level_of_theory=="pbeh3c": 
        standard_charge=   {'nbo_charges': {'o': 0.50815,'n': 0.39085,'s': 0.182355,'c': 0.22971999999999998,'p': 0.02240333333333333},
                            'nbo_wiberg_bond_orders': {'o': 0.7418,'n': 0.8472,'s': 0.7418,'c': 0.7418,'p': 0.9941},
                            'nbo_nbi_bond_orders': {'o': 0.86125,'n': 0.9204,'s': 0.86125,'c': 0.86125,'p': 0.997},
                            'chg_hirshfeld': {'o': 0.1905610162,'n': 0.1109055985,'s': 0.07776023195000001,'c': 0.036585679,'p': 0.0021802815000000002},
                            'chg_voronoy': {'o': 0.20057303425,'n': 0.1012828991,'s': 0.0906680465,'c': 0.02093110275,'p': -0.001363665},
                            'chg_mulliken': {'o': 0.45471951945,'n': 0.3424360946,'s': 0.22577799599999998,'c': 0.231474828725,'p': 0.11289274093333335},
                            'chg_lowdin': {'o': 0.3703919455,'n': 0.2688547916,'s': 0.1468233268,'c': 0.16494621645000002,'p': 0.07300937596666666},
                            'chg_becke': {'o': 0.4469565783,'n': 0.3888989125,'s': 0.20501285695,'c': 0.09101329989999998,'p': 0.13075999536666666},
                            'chg_ADCH': {'o': 0.4469487045,'n': 0.3888727262,'s': 0.2050235742,'c': 0.103312470575,'p': 0.13077291276666667},
                            'chg_CHELPG': {'o': 0.4540982739,'n': 0.392007253,'s': 0.2170219944,'c': 0.129451173025,'p': 0.1205248201},
                            'chg_MK': {'o': 0.4518616422,'n': 0.3876915247,'s': 0.21827006854999997,'c': 0.164140149675,'p': 0.1229121943},
                            'chg_CM5': {'o': 0.3573093059,'n': 0.2913212718,'s': 0.14259058949999998,'c': 0.084645072525,'p': 0.08245481013333333},
                            'chg_12CM5': {'o': 0.42877104504999997,'n': 0.3495851958,'s': 0.1711087611,'c': 0.10157400845,'p': 0.0989458003},
                            'chg_RESP': {'o': 0.4517566405,'n': 0.3875430412,'s': 0.21820557965,'c': 0.1605050264,'p': 0.1228428746},
                            'chg_PEOE': {'o': 0.2057302845,'n': 0.1146232457,'s': 0.09844511,'c': 0.0194090064,'p': 0.0511157583},
                            'bo_mayer': {'o': 0.782426225,'n': 0.86759229,'s': 0.9205561,'c': 0.93207973,'p': 0.9590220233333334},
                            'bo_wiberg': {'o': 0.8778146099999999,'n': 0.93691806,'s': 0.988363,'c': 0.9728263825,'p': 1.0000357066666667},
                            'bo_mulliken': {'o': 0.5805700149999999,'n': 0.67686849,'s': 0.60310116,'c': 0.7536101024999999,'p': 0.6911617566666667},
                            'bo_fuzzy': {'o': 0.920020115,'n': 0.95926686,'s': 1.0726820799999999,'c': 0.95851575,'p': 1.0272596766666666},
                            'bo_laplacian': {'o': 0.51630396,'n': 0.73264462,'s': 0.651612035,'c': 0.7920171699999999,'p': 0.70830269},
                            'bo_IBSI': {'o': 1.689635,'n': 1.49291,'s': 0.8854500000000001,'c': 1.204155,'p': 0.72317},
                            'v_at_nucleus': {'o': -615.1726,'n': -663.2573,'s': -623.1650500000001,'c': -700.5221750000001,'p': -664.5396999999999},
                            }

    elif level_of_theory=="r2scan3c": 
        standard_charge=   {'nbo_charges': {'o': 0.52994,'n': 0.41253,'s': 0.203135,'c': 0.25393,'p': 0.038856},
                            'nbo_wiberg_bond_orders': {'o': 0.7201,'n': 0.8302,'s': 0.9587,'c': 0.9344,'p': 0.9944},
                            'nbo_nbi_bond_orders': {'o': 0.8486,'n': 0.9112,'s': 0.9791,'c': 0.9666,'p': 0.9972},
                            'chg_hirshfeld': {'o': 0.17501733419999999,'n': 0.1030776979,'s': 0.07420372435,'c': 0.03614708185,'p': 0.0011362277666666669},
                            'chg_voronoy': {'o': 0.17665978485,'n': 0.08659040693333332,'s': 0.062288534150000005,'c': 0.018099560125,'p': -0.010174435833333334},
                            'chg_mulliken': {'o': 0.3598106532,'n': 0.3018137650333334,'s': 0.2116424059,'c': 0.15696562735,'p': 0.09856875859999999},
                            'chg_lowdin': {'o': 0.3234688241,'n': 0.2869278436666666,'s': 0.20320992385,'c': 0.18673573470000002,'p': 0.14980201866666668},
                            'chg_becke': {'o': 0.42307506615,'n': 0.3576609553666667,'s': 0.1637756515,'c': 0.08848301642499999,'p': 0.11126772316666667},
                            'chg_ADCH': {'o': 0.4230702012,'n': 0.35764690343333333,'s': 0.16378834009999999,'c': 0.1022323847,'p': 0.11128463076666667},
                            'chg_CHELPG': {'o': 0.4309546073,'n': 0.36116613940000003,'s': 0.1778839307,'c': 0.116656376275,'p': 0.09889692656666667},
                            'chg_MK': {'o': 0.4283754597,'n':0.35569485196666667 ,'s': 0.182123512,'c': 0.1499247789,'p': 0.102447941},
                            'chg_CM5': {'o':0.34025586475 ,'n': 0.28281864556666664,'s': 0.1372838306,'c': 0.084257293925,'p': 0.08014546766666665},
                            'chg_12CM5': {'o': 0.4083068953,'n': 0.33938194816666667,'s': 0.16474064965,'c': 0.101108641375,'p': 0.09617459290000001},
                            'chg_RESP': {'o': 0.4282719404,'n': 0.3555464571,'s': 0.18206142625,'c': 0.1462833787,'p': 0.10238090996666666},
                            'chg_PEOE': {'o': 0.2057302845,'n': 0.1146232457,'s': 0.09844511,'c': 0.0194090064,'p': 0.0511157583},
                            'bo_mayer': {'o': 0.879583845,'n': 0.9008711733333333,'s': 0.9471364499999999,'c': 0.96854546,'p': 0.9756898566666666},
                            'bo_wiberg': {'o': 0.962210655,'n': 0.9481929966666667,'s': 0.9945117400000001,'c': 0.9660152550000001,'p': 0.9914074333333334},
                            'bo_mulliken': {'o': 0.63766731,'n': 0.6853113,'s': 0.653575775,'c': 0.7942201124999999,'p': 0.7117326733333332},
                            'bo_fuzzy': {'o': 0.945368225,'n':0.9736232833333333 ,'s': 1.09300325,'c': 0.95812326,'p': 1.0348678833333331},
                            'bo_laplacian': {'o': 0.55823106,'n': 0.79727295,'s': 0.7468226149999999,'c': 0.9102260125,'p': 0.7525360366666667},
                            'bo_IBSI': {'o': 1.67542,'n': 1.4881766666666667,'s': 0.866495,'c':1.20523 ,'p': 0.7147233333333333},
                            'v_at_nucleus': {'o': -614.50195,'n': -661.8159333333333,'s': -626.47225,'c': -703.370275,'p':-668.9968 },
                            }
        
    elif level_of_theory=="wb97x3c": 
        standard_charge=   {'nbo_charges': {'o': 0.51566,'n': 0.3828,'s': 0.20734,'c': 0.217105,'p': 0.04161},
                            'nbo_wiberg_bond_orders': {'o': 0.7359,'n': 0.8545,'s': 0.9578,'c': 0.9524,'p': 0.09934},
                            'nbo_nbi_bond_orders': {'o': 0.8578,'n': 0.9244,'s': 0.9787,'c': 0.9759,'p': 0.9967},
                            'chg_hirshfeld': {'o': 0.17473751825,'n': 0.10329593826666666,'s': 0.0726495078,'c': 0.034780045100000004,'p': -0.004547128333333333},
                            'chg_voronoy': {'o': 0.1767633931,'n': 0.0867188289,'s': 0.059544269149999995,'c': 0.01619503326666667,'p': -0.0168581133},
                            'chg_mulliken': {'o': 0.1878759293,'n': 0.10119374716666667,'s': 0.081586198,'c': 0.036039024175,'p': -0.024330636933333333},
                            'chg_lowdin': {'o': 0.15449935595,'n': 0.11738489270000001,'s': 0.01879406805,'c': 0.09929541682500001,'p': -0.0034173678},
                            'chg_becke': {'o': 0.4381940065,'n': 0.38026620876666667,'s': 0.17102061505,'c': 0.086487391075,'p': 0.09519356776666665},
                            'chg_ADCH': {'o': 0.43808665835,'n': 0.38012732400000004,'s': 0.1710529309,'c': 0.10024663635,'p': 0.09527818536666666},
                            'chg_CHELPG': {'o': 0.4416169433,'n': 0.37709467513333333,'s': 0.1831113003,'c': 0.107320675,'p':0.08414238596666668 },
                            'chg_MK': {'o': 0.4402911664,'n': 0.375670541,'s': 0.18959444445,'c': 0.138371843025,'p': 0.08950905046666668},
                            'chg_CM5': {'o': 0.3399774673,'n': 0.28303929636666664,'s': 0.1357300386,'c': 0.08289077495,'p': 0.0744645047},
                            'chg_12CM5': {'o': 0.4079735189,'n': 0.3396476398333334,'s': 0.1628763178,'c': 0.09946903455,'p': 0.08935797643333333},
                            'chg_RESP': {'o': 0.44018761,'n': 0.3755220791,'s': 0.18953218975,'c': 0.1347403059,'p': 0.08944299423333334},
                            'chg_PEOE': {'o': 0.2057302845,'n': 0.1146232457,'s': 0.09844511,'c': 0.0194090064,'p': 0.0511157583},
                            'bo_mayer': {'o': 1.04007384,'n': 1.0230898166666667,'s': 1.03315449,'c': 0.9940787275,'p': 1.0044232133333335},
                            'bo_wiberg': {'o': 1.22123846,'n': 1.1147673933333333,'s': 1.19453484,'c': 0.9879965975,'p': 1.0776919466666666},
                            'bo_mulliken': {'o': 0.73301719,'n': 0.7919044466666666,'s': 0.72211152,'c': 0.8298811425,'p': 0.7430875166666667},
                            'bo_fuzzy': {'o': 0.94462322,'n': 0.9739612766666665,'s': 1.093753035,'c': 0.9596483775,'p': 1.0350417833333332},
                            'bo_laplacian': {'o': 0.524868095,'n': 0.7713855033333333,'s': 0.7321695500000001,'c': 0.9120541275,'p': 0.7492871033333334},
                            'bo_IBSI': {'o': 1.67527,'n': 1.4881766666666667,'s': 0.866495,'c': 1.20523,'p': 0.7147233333333333},
                            'v_at_nucleus': {'o': -611.5980500000001,'n': -662.1311,'s':-627.5749000000001 ,'c': -706.4158,'p': -672.2800333333333},
                            }

    if type(prop)==list and type(prop[0])==float:
        relative_charge=[]
        for i,p in enumerate(prop):
            if molecule.atom_list[i].symbol.lower()!="h": relative_charge.append(0.0)
            else:
                nearest_atom=molecule.nearest_atom(molecule.atom_list[i],exclude_H=True)
                if nearest_atom.symbol.lower() in standard_charge[chg].keys():
                    if False: #chg!="list_of_chgs_on_which_is_better_to_use_difference_instead_of_ratio":
                        relative_charge.append(p/standard_charge[chg][nearest_atom.symbol.lower()])
                    else: #better use_difference_instead_of_ratio
                        relative_charge.append(p-standard_charge[chg][nearest_atom.symbol.lower()])
                else: relative_charge.append(0.0)
        return relative_charge

    elif type(prop)==list and type(prop[0])==list:
        new_matrix=np.zeros([len(prop),len(prop)])
        for i in range(len(prop)):
            if molecule.atom_list[i].symbol.lower()=="h":
                nearest_atom=molecule.nearest_atom(molecule.atom_list[i],exclude_H=True)
                if nearest_atom.symbol.lower() in standard_charge[chg].keys():
                    new_matrix[i,nearest_atom.atom_number-1]=prop[i][nearest_atom.atom_number-1]/standard_charge[chg][nearest_atom.symbol.lower()]
                    new_matrix[nearest_atom.atom_number-1,i]=prop[i][nearest_atom.atom_number-1]/standard_charge[chg][nearest_atom.symbol.lower()]
        return new_matrix

def get_NBO_charges(filename):
    with open (filename,"r") as f: lines=f.readlines()
    #charges_section_start_line=lines.index(" Summary of Natural Population Analysis:                  \n")+6
    charges_section_start_line=lines.index(" Summary of Natural Population Analysis:\n")+6
    charges=[]
    for i in range(charges_section_start_line,len(lines)):
        if "==========" in lines[i]: break
        charges.append(float(lines[i].split()[2]))
    return charges

def get_NBO_atom_list(filename):
    with open (filename,"r") as f: lines=f.readlines()
    #charges_section_start_line=lines.index(" Summary of Natural Population Analysis:                  \n")+6
    charges_section_start_line=lines.index(" Summary of Natural Population Analysis:\n")+6
    symbols_list=[]
    for i in range(charges_section_start_line,len(lines)):
        if "==========" in lines[i]: break
        symbols_list.append((lines[i].split()[0]))
    return symbols_list

def read_NBO_BO_matrix(start_text,stop_text,nbo_file):

    if not start_text.endswith("\n"): start_text+="\n"
    if not stop_text.endswith("\n"): stop_text+="\n"
    with open (nbo_file,"r") as f: nbo_lines=f.readlines()
    nbo_lines=nbo_lines[nbo_lines.index(start_text):nbo_lines.index(stop_text)]

    #find out the number of atoms:
    shift=0
    for i in range(0,len(nbo_lines)):
        if nbo_lines[i][4:6]==". ": n_atoms=int(nbo_lines[i][0:4])
        if "Atom" in nbo_lines[i]:
            if shift==0:shift=i #the first time "Atom" is read, it is used for defining "shift"
            else: break #next time, do not keep on reading
        
    #define the matrix
    BO=[ [0.0 for _ in range(0,n_atoms)] for _ in range(0,n_atoms)  ]

    for i in range(shift,len(nbo_lines),n_atoms+3):
        column_indexes=[int(w)  for w in nbo_lines[i].split()[1:]]
        for line in nbo_lines[i+2:i+n_atoms+2]:
            row_index=int(line.split(".")[0])
            for w,column_index in zip(line.split()[2:],column_indexes)   :
                BO[row_index-1][column_index-1]=float(w)
    return BO

def get_nbo_wiberg_matrix(nbo_file):
    start=" Wiberg bond index matrix in the NAO basis:\n"
    finish=" Wiberg bond index, Totals by atom:\n"
    NBO_BO_matrix=np.array(read_NBO_BO_matrix(start,finish,nbo_file))

    ref_values_swb97xd={"O":(0.7376+0.7379)/2, "N":0.84877, "S":0.9605, "C":0.9471, "P":0.9938}
    ref_values_wb97xd={"O":0.752,"N":0.86403,"S":0.9724,"C":0.9525,"P":0.9935}
    ref_values_pbeh3c={"O":0.7417,"N":0.8472,"S":0.9649,"C":0.94615,"P":0.9941}
    ref_values_m06={"O":0.7482,"N":0.8634,"S":0.9735,"C":0.9531,"P":0.9938}
    ref_values_sm06={"O":0.7346,"N":0.8480,"S":0.9614,"C":0.9468,"P":0.9944}
    ref_values_r2scan3c={"O":0.7201,"N":0.8302,"S":0.9603,"C":0.9344,"P":0.9972}
    ref_values_wb97x3c={"O":0.7201,"N":0.8545,"S":0.9578,"C":0.9538,"P":0.9934}
    if level_of_theory=="M06": ref_values= ref_values_m06
    elif level_of_theory=="sM06": ref_values= ref_values_sm06
    elif level_of_theory=="pbeh3c": ref_values= ref_values_pbeh3c
    elif level_of_theory=="swb97xd": ref_values= ref_values_swb97xd
    elif level_of_theory=="wb97xd": ref_values= ref_values_wb97xd
    elif level_of_theory=="r2scan3c": ref_values=ref_values_r2scan3c
    elif level_of_theory=="wb97x3c": ref_values=ref_values_wb97x3c

    NBO_rel_BO_matrix=np.zeros_like(NBO_BO_matrix)
    NBO_atom_list=get_NBO_atom_list(nbo_file)

    for i in range (len(NBO_BO_matrix)):
        for j in range (len(NBO_BO_matrix)):
            if NBO_atom_list[i]=="H" and NBO_atom_list[j] in   ref_values.keys():
                NBO_rel_BO_matrix[i,j]=NBO_BO_matrix[i,j]/ref_values[NBO_atom_list[j]]
            if NBO_atom_list[j]=="H" and NBO_atom_list[i] in   ref_values.keys():
                NBO_rel_BO_matrix[i,j]=NBO_BO_matrix[i,j]/ref_values[NBO_atom_list[i]]
    return [NBO_BO_matrix, NBO_rel_BO_matrix ]

def get_nbo_nbi_matrix(nbo_file):
    start=" NBI: Natural Binding Index (NCU strength parameters)\n"
    finish=" NATURAL BOND ORBITAL ANALYSIS:\n"
    NBO_BO_matrix=np.array(read_NBO_BO_matrix(start,finish,nbo_file))

    ref_values_swb97xd={"O":(0.8590+0.8588)/2, "N":0.9213, "S":0.9801, "C":0.9732, "P":0.9969}
    ref_values_wb97xd={"O":0.86765,"N":0.9295,"S":0.9861,"C":0.97595,"P":0.9968}
    ref_values_pbeh3c={"O":0.86125,"N":0.9204,"S":0.9823,"C":0.9727,"P":0.9970}
    ref_values_m06={"O":0.8649,"N":0.9292,"S":0.9867,"C":0.9763,"P":0.9969}
    ref_values_sm06={"O":0.8571,"N":0.9209,"S":0.9805,"C":0.9730,"P":0.9972}
    ref_values_r2scan3c={"O":0.8486,"N":0.9112,"S":0.9791,"C":0.9666,"P":0.9972}
    ref_values_wb97x3c={"O":0.8578,"N":0.9244,"S":0.9787,"C":0.9759,"P":0.9967}   
    if level_of_theory=="M06": ref_values= ref_values_m06
    elif level_of_theory=="sM06": ref_values= ref_values_sm06
    elif level_of_theory=="pbeh3c": ref_values= ref_values_pbeh3c
    elif level_of_theory=="swb97xd": ref_values= ref_values_swb97xd
    elif level_of_theory=="wb97xd": ref_values= ref_values_wb97xd
    elif level_of_theory=="r2scan3c": ref_values=ref_values_r2scan3c
    elif level_of_theory=="wb97x3c": ref_values=ref_values_wb97x3c

    NBO_rel_BO_matrix=np.zeros_like(NBO_BO_matrix)
    NBO_atom_list=get_NBO_atom_list(nbo_file)
    for i in range (len(NBO_BO_matrix)):
        for j in range (len(NBO_BO_matrix)):
            if NBO_atom_list[i]=="H" and NBO_atom_list[j] in   ref_values.keys():
                NBO_rel_BO_matrix[i,j]=NBO_BO_matrix[i,j]/ref_values[NBO_atom_list[j]]
            if NBO_atom_list[j]=="H" and NBO_atom_list[i] in   ref_values.keys():
                NBO_rel_BO_matrix[i,j]=NBO_BO_matrix[i,j]/ref_values[NBO_atom_list[i]]

    return [NBO_BO_matrix, NBO_rel_BO_matrix ]

def get_nbo_nlmonpa_matrix(nbo_file):
    start=" Atom-Atom Net Linear NLMO/NPA Bond Orders:\n"
    finish=" Linear NLMO/NPA Bond Orders, Totals by Atom:\n"
    NBO_BO_matrix=np.array(read_NBO_BO_matrix(start,finish,nbo_file))

    ref_values_wb97xd={"O":0.5012,"N":0.6280,"S":0.82415,"C":0.7834,"P":0.9771}
    ref_values_swb97xd={"O":(0.4859+0.4856)/2, "N":0.6055, "S":0.79865, "C":0.7713, "P":0.95613}
    ref_values_pbeh3c={"O":0.49155,"N":0.6086,"S":0.8163,"C":0.77025,"P":0.97387}
    ref_values_m06={"O":0.49665,"N":0.6272,"S":0.8281,"C":.78475,"P":0.9794}
    ref_values_sm06={"O":0.4828,"N":0.60457,"S":0.8015,"C":0.77077,"P":0.96136}
    ref_values_r2scan3c={"O":0.4701,"N":0.5875,"S":0.7956,"C":0.7461,"P":0.9576}
    ref_values_wb97x3c={"O":0.4843,"N":0.6172,"S":0.78995,"C":0.7829,"P":0.95176666} 

    if level_of_theory=="M06": ref_values= ref_values_m06
    elif level_of_theory=="sM06": ref_values= ref_values_sm06
    elif level_of_theory=="pbeh3c": ref_values= ref_values_pbeh3c
    elif level_of_theory=="swb97xd": ref_values= ref_values_swb97xd
    elif level_of_theory=="wb97xd": ref_values= ref_values_wb97xd
    elif level_of_theory=="r2scan3c": ref_values=ref_values_r2scan3c
    elif level_of_theory=="wb97x3c": ref_values=ref_values_wb97x3c

    NBO_rel_BO_matrix=np.zeros_like(NBO_BO_matrix)
    NBO_atom_list=get_NBO_atom_list(nbo_file)
    for i in range (len(NBO_BO_matrix)):
        for j in range (len(NBO_BO_matrix)):
            if NBO_atom_list[i]=="H" and NBO_atom_list[j] in   ref_values.keys():
                NBO_rel_BO_matrix[i,j]=NBO_BO_matrix[i,j]/ref_values[NBO_atom_list[j]]
            if NBO_atom_list[j]=="H" and NBO_atom_list[i] in   ref_values.keys():
                NBO_rel_BO_matrix[i,j]=NBO_BO_matrix[i,j]/ref_values[NBO_atom_list[i]]

    return [NBO_BO_matrix, NBO_rel_BO_matrix ]
 
#read the repeated_molecules file to get the number of equivalent structures for modifiying populations 
def get_n_identical_structures(filename):
    filename=filename.split("/")[-1]

    if filename[-4:]==".out":filename=filename.split(".out")[0]

    if "-cation" in filename: compn=filename.split("-cation")[0]+"-cation*"
    elif "-2cation" in filename: compn=filename.split("-2cation")[0]+"-2cation*"
    elif "-3cation" in filename: compn=filename.split("-3cation")[0]+"-3cation*"
    elif "-4cation" in filename: compn=filename.split("-4cation")[0]+"-4cation*"
    elif "-5cation" in filename: compn=filename.split("-5cation")[0]+"-5cation*"
    elif "-6cation" in filename: compn=filename.split("-6cation")[0]+"-6cation*"
    elif "-neut" in filename: compn=filename.split("-neut")[0]+"-neut*" 
    elif "-an"in filename: compn=filename.split("-an")[0]+"-an*"
    elif "-2an"in filename: compn=filename.split("-2an")[0]+"-2an*"
    elif "-3an"in filename: compn=filename.split("-3an")[0]+"-3an*"
    elif "-4an"in filename: compn=filename.split("-4an")[0]+"-4an*"
    elif "-5an"in filename: compn=filename.split("-5an")[0]+"-5an*"
    elif "-6an"in filename: compn=filename.split("-6an")[0]+"-6an*"


    if filename in repeated_molecules.repeated_molecules.keys(): return repeated_molecules.repeated_molecules[filename]
    elif compn in repeated_molecules.repeated_molecules.keys(): return repeated_molecules.repeated_molecules[compn]
    else: return 1

#returns a list of lists, each of it consisting in the list of equilvalent atoms 
# equivalence is stablished by comparing Molecular_structre fingerprints (without geometrical information)
def get_groups_of_equivalent_atoms(molecule):
    groups_of_eq_atoms=[]
    molecule.set_fingerprints(refine=False)
    for i in range(0,len(molecule.atom_list)):
        if not any(i+1 in r for r in groups_of_eq_atoms):
            gp=[i+1]
            for j in range(i+1,len(molecule.atom_list)):
                if molecule.atom_list[j].fingerprint==molecule.atom_list[i].fingerprint: gp.append(j+1)
            groups_of_eq_atoms.append(sorted(gp))
    fingerprints=[]
    for g in groups_of_eq_atoms:
        fingerprints.append(molecule.atom_list[g[0]-1].fingerprint)

    return groups_of_eq_atoms,fingerprints            

#returns the connections between groups of equivalent atoms.
#if a is in group A and b in group B and they are connected, [a,b] is added to the list
def get_connections(molecule,groups_of_equivalent_atoms,diag_zero=False):
    connections=[]
    for i in range(0,len(groups_of_equivalent_atoms)):
        for j in range(i,len(groups_of_equivalent_atoms)):
            for c in molecule.atom(groups_of_equivalent_atoms[i][0]).connection:
                if (c[0]) in groups_of_equivalent_atoms[j]: connections.append([i,j])
    if diag_zero:
        connections=[c for c in connections if c[0]!=c[1]]

    return connections

#returns the groups of equivalent atoms (list of lists), the connections between groups (list of lists) and a nx graph object
def equivalent_atoms_from_molecule(m):
    m.generate_connections_from_distance(allow_fract_orders=False)
    
    #equivalent (by 2D symmetry) atoms are grouped together:
    groups_of_equivalent_atoms,fingerprints=get_groups_of_equivalent_atoms(m)   #indexes start at 1, not 0!!!
    connections_between_groups=get_connections(m,groups_of_equivalent_atoms)
    #build a graph whose nodes are each group of equilvalent atoms and connected depending on connections_between_groups
    G=nx.Graph()
    #to ensure that all nodes have a different identifier, the fingerprint is added to the nodes. It must be cast to str because list are not hashable
    #print ([(i,{"fingerprint":  fingerprints[i]    })     for i,g in enumerate(groups_of_equivalent_atoms)])
    G.add_nodes_from([(i,{"fingerprint":  str(fingerprints[i])    })     for i,g in enumerate(groups_of_equivalent_atoms)])
    G.add_edges_from(connections_between_groups)

    return groups_of_equivalent_atoms,connections_between_groups,G

"""
#returns a networkX graph from a molecule. Each node in the graph is added "labels" dictionary 
# if use_symmetry, only one node of each equivalent atom is kept, and their features are calculated from the features of equivalent atoms
# using symmetry_method ("sum" or "mean")
def graph_from_molecule(molecule,labels=""):

    G=nx.Graph()
     
    if labels=="":
        nodes=[(k,{"symbol":molecule.atom_list[k].symbol.lower()}) for k in range(0,len(molecule.atom_list))]
    elif type(labels)==list and type(labels[0])==dict:
        nodes=[(k,l) for k,l in enumerate(labels)]
    G.add_nodes_from(nodes)
    for a in molecule.atom_list:
        for aa in a.connection: G.add_edge(a.atom_number-1,aa[0]-1)   
        
    return G
            

def get_feature_names_from_nx_graph(nx_graph):
    labels=nx_graph.nodes()
    feature_keys=[k for k in labels[0].keys() if type(labels[0][k])==np.float64 or type(labels[0][k])==float]
    vector_keys=[k for k in labels[0].keys() if type(labels[0][k])==np.ndarray or type(labels[0][k])==list]
    return {"feature_keys":feature_keys,"vector_keys":vector_keys}

# creates a json object from a graph that can be read to generate a spektral graph
def nx_graph_to_json(nx_graph,name):
    
    a=list(nx.adjacency_matrix(nx_graph).todense())
    keys=get_feature_names_from_nx_graph(nx_graph)
    feature_keys,vector_keys=keys["feature_keys"],keys["vector_keys"]


    #keys=list(nx_graph.nodes.data()[0].keys())
    #feature_keys,vector_keys=[],[]
    #for k in keys:
    #    if type(nx_graph.nodes.data()[0][k])==np.float64 or type(nx_graph.nodes.data()[0][k])==float: feature_keys.append(k)
    #    elif type(nx_graph.nodes.data()[0][k])==np.ndarray or type(nx_graph.nodes.data()[0][k])==list: vector_keys.append(k)

    #create the matrix with the features
    x=[]
    for d in nx_graph.nodes.data():
        xx=[]
        for k in feature_keys:
            xx.append(d[1][k])
        x.append(xx)

    #create the matrices with the bond orders (the edge matrices)
    e=[]

    for k in vector_keys:
        ee=[]
        for d in nx_graph.nodes.data():
            #ee is a matrix in which element in row i and column j indicates how the bond order between atom i and j changes upon protonation
            #for _original matrices, diagonal elements are 0 (atom is not bound to itself)
            #for the _added bond orders, it indicates how the bond between united atom j and atom i (not united atom i... it is not symmetrical) changes upon protonation
            #diagonal elements indicates how the bond between each atom and H atoms bound to it changes upon protonation -it is very large for the H atoms lost in the deprotonation-.
            #ith element of: np.sum(ee,axis=1) is the change of bond orders of united atom i 
            ee.append(d[1][k])
            # for softmax:
            #ee=np.array([np.exp(np.abs(b))/np.sum(np.exp(np.abs(b))) for b in ee] )
        e.append(ee)
    e=np.array(e)

    #e=np.array(e).transpose(2,1,0) #this is what spektra is waiting for e: (n_nodes)x(n_nodes)x(n_edge_features) ????

    y=name

    return json.dumps({"label":name,"x":x,"a":a,"e":e,"y":y},cls=NpEncoder)


# reads a spektral graph from json object generated by nx_graph_to_json
def spektral_graph_from_json(json_text):
    
    d=json.loads(json_text)
    x=np.array(d["x"])
    a=np.array(d["a"])
    e=np.array(d["e"])
    y=d["y"] #dictionary that contains the names of the features and the name of the compounds 
    return spektral.data.graph.Graph(x=x,a=a,y=y,e=e)

#creates a spektral graph from networkx graph
def spektral_graph_from_nx_graph(nx_graph,name):
    json_line=nx_graph_to_json(nx_graph,name)
    return spektral_graph_from_json(json_line)
"""





#returns a list of dictionaries for the atomic properties, removing H and adding their properties to the nearest atom, reducing the molecule according to the symmetry
#and ensuring that the order of the atoms is equivalent to the reference molecule passed
#it requires both HL and geometry optimization molecules so chemical shifts and fuerza´s bond order can be extracted, and the HL file name to read multifwn files and NBO stuff.
#the order of the dictionaries is consistent with the oder of the list of molecules and filenames (which must match)
def get_lists_of_properties_for_symmetry_reduced_graph(molecules_HL,molecules,HL_molecules_file_names,ref_molecule):

    props=[]
    #read NBO charges, bond orders ans ei berg bond orders from NBO files
    NBO_charges=[get_NBO_charges(nbo_route+filename[:-4].split("/")[-1]+".nbo.out") for filename in HL_molecules_file_names ]
    NBO_nbi_bond_orders=[get_nbo_nbi_matrix(nbo_route+filename[:-4].split("/")[-1]+".nbo.out") for filename in HL_molecules_file_names ]
    NBO_wiberg_bond_orders=[get_nbo_wiberg_matrix(nbo_route+filename[:-4].split("/")[-1]+".nbo.out") for filename in HL_molecules_file_names ]
    NBO_NLMO_NPA_bond_orders=[get_nbo_nlmonpa_matrix(nbo_route+filename[:-4].split("/")[-1]+".nbo.out") for filename in HL_molecules_file_names ]

    #read multwfn json files
    multwfn_props=[]
    for pmn in HL_molecules_file_names:
        with open(pmn[:-4]+".multwfn.json","r") as f: l=f.read()
        multwfn_props.append(json.loads(l))

    reference_molecule=copy.deepcopy(ref_molecule)
    reference_molecule.remove_atoms("h")
    reference_molecule.generate_connections_from_distance(allow_fract_orders=False)#do this
    reference_groups_of_equivalent_atoms, reference_connections_between_groups, reference_graph=equivalent_atoms_from_molecule(reference_molecule)
    for pm,m,pmp,pNBOchgs,pNBOnbibo,pNBOwibergbo,NBO_NLMO_NPA_bo in zip(molecules_HL,molecules,multwfn_props,NBO_charges,NBO_nbi_bond_orders,NBO_wiberg_bond_orders,NBO_NLMO_NPA_bond_orders):
        np.set_printoptions(precision=2,linewidth=400)
        #add properties: force constants, nbo charges, nbi bond orders and wiberg bond orders to atoms
        m.read_hess()

        pmp["isC"]=[float("c"==a.symbol.lower()) for a in m.atom_list]    #expressed as float instead of bool so it can be operated
        pmp["isN"]=[float("n"==a.symbol.lower()) for a in m.atom_list]
        pmp["isO"]=[float("o"==a.symbol.lower()) for a in m.atom_list]
        pmp["isS"]=[float("s"==a.symbol.lower()) for a in m.atom_list]

        #pmp["force_constants"]=np.sum(get_force_constant_matrix(m),axis=0)
        pmp["force_constants"]=get_force_constant_matrix(m)
        pmp["rel_force_constants"]=get_force_constant_matrix(m,relative=True)
        pmp["inv_distances"]=m.inv_distance_matrix()
        distances=m.distance_matrix()
        rel_distances=m.distance_matrix(relative=True)
        #only distances smaler than 1.20 times the sum of covalent radius are kept; 
        #although distances were only calculated for bound atoms, this criteria is even more restrictive 
        bound_mask=rel_distances<1.20
        pmp["bond_distances"]=distances*bound_mask   
        pmp["rel_bond_distances"]=rel_distances*bound_mask #only rel-distances smaler than 1.20 are kept

        pmp["nbo_charges"]=pNBOchgs
        pmp["nbo_nbi_bond_orders"]=pNBOnbibo[0]
        pmp["rel_nbo_nbi_bond_orders"]=pNBOnbibo[1]
        pmp["nbo_wiberg_bond_orders"]=pNBOwibergbo[0]  
        pmp["rel_nbo_wiberg_bond_orders"]=pNBOwibergbo[1]   
        pmp["nbo_nlmonpa_bond_orders"]= NBO_NLMO_NPA_bo[0]  
        pmp["rel_nbo_nlmonpa_bond_orders"]= NBO_NLMO_NPA_bo[1]
        pmp["chemical_isotropic_shield"]=get_chemical_shifts(pm,relative=False) 
        pmp["rel_chemical_isotropic_shield"]=get_chemical_shifts(pm,relative=True)
        for k in ['chg_hirshfeld', 'chg_voronoy', 'chg_mulliken', 'chg_lowdin', 'chg_becke', 'chg_ADCH',
                  'chg_CHELPG', 'chg_MK', 'chg_CM5', 'chg_12CM5', 'chg_RESP', 'chg_PEOE', 'bo_mayer', 
                  'bo_wiberg', 'bo_mulliken', 'bo_fuzzy', 'bo_laplacian', 'bo_IBSI','v_at_nucleus','nbo_charges']:   
            pmp["rel_"+k]=get_relative_chg(pmp[k],pm,chg=k)


        #mu_proj_bonds,alpa_wr2_bonds_proj_on_bonds,e_quad_proj_bonds, tot_quad_proj_bonds are matrices (number of atoms x number of atoms)
        #they are masked so they are only kept for atoms whose distance is smaller than 1.25 times the sum of the cov. radius
        mu_proj_bonds,alpa_wr2_bonds_proj_on_bonds, alpa_wr2_bonds_with_H_proj_on_bonds  , e_quad_proj_bonds, tot_quad_proj_bonds=get_projected_molecular_dipole(pm)
        pmp["molecular_dipole_projected_on_bonds"]=mu_proj_bonds*bound_mask
        pmp["polarizability_wr2_bonds_projected_on_bonds"]=alpa_wr2_bonds_proj_on_bonds*bound_mask
        pmp["e_spatial_extent_projected_on_bonds"]=e_quad_proj_bonds*bound_mask
        pmp["tot_spatial_extent_projected_on_bonds"]=tot_quad_proj_bonds*bound_mask
        #alpa_wr2_bonds_with_H_proj_on_bonds is a tensor (number of atoms x number of atoms x number of atoms). 
        #to apply the mask ,first it must be broadcasted from 2d to 3d mask, repeating in the 3rd dimension the values of the 2d matrix
        #broadcasted_bound_mask=np.repeat(bound_mask[:,:,np.newaxis],bound_mask.shape[0]  ,axis=2)
        broadcasted_bound_mask=np.repeat(bound_mask[np.newaxis,:,:],len(alpa_wr2_bonds_with_H_proj_on_bonds)  ,axis=0)
        pmp["polarizability_wr2_bonds_with_H_projected_on_bonds"]=alpa_wr2_bonds_with_H_proj_on_bonds*broadcasted_bound_mask


        pmp["number_of_H"]=[0.0]*len(m.atom_list) #number of H atoms bound to each atom; initialized as 0.0 

        #properties that will not be included:
        exclude_keys=["FASA_mol_electronic_spatial_extent_components","FASA_molecular_dipole_moment","FASA_atomic_dipole_moments",
                    "FASA_atomic_dipole_moments_contributions","FASA_Components_of_<r^2>"]    
        for k in exclude_keys: del(pmp[k])
        
        #scalar vector and tensor properties will be treated differently, so lets find out which property is scalar and which is vectorial
        atom_scalar_properties=[p for p in pmp.keys() if (type(pmp[p]) in [list,np.ndarray]) 
                                                        and (type(pmp[p][0]) in  [float,np.float32,np.float64]) and p not in exclude_keys]  
        
        atom_vector_properties=[p for p in pmp.keys() if (type(pmp[p]) in [list,np.ndarray]) and (type(pmp[p][0]) in  [list,np.ndarray])  
                                                        and (type(pmp[p][0][0]) in [float,np.float32,np.float64])  and p not in exclude_keys]
        
        atom_tensor_properties=[p for p in pmp.keys() if (type(pmp[p]) in [list,np.ndarray]) and (type(pmp[p][0]) in  [list,np.ndarray]) 
                                                        and (type(pmp[p][0][0]) in  [list,np.ndarray]) and p not in exclude_keys]
                                

        to_remove=[]
        for k in pmp.keys():
            if k not in atom_scalar_properties and k not in atom_vector_properties and k not in atom_tensor_properties: to_remove.append(k)
        for k in to_remove: del(pmp[k])
            
        #delete diagonal elements of vector properties (bond order of an atom with itself must be 0)
        for k in atom_vector_properties:
            if k!="inv_distances":
                # np.fill_diagonal(pmp[k],0.0)
                for i in range(len(pmp[k])): pmp[k][i][i]=0.0
            else: # inv distances is changed to a large number; later the smallest will be selected
                #np.fill_diagonal(pmp[k],999.9)
                for i in range(len(pmp[k])): pmp[k][i][i]=999.9
        
        #delete diagonal elements of tensor properties (maybe it is not needed)
        for k in atom_tensor_properties:
            for i in range(len(pmp[k])):
                for j in range(len(pmp[k][i])): pmp[k][j][i][i]=0.0
        


        #add properties of H atoms to the nearest non-H atom 
        #create the properties:
        for k in atom_scalar_properties: pmp[k+"_H"]=[0.0]*len(pm.atom_list)   
        for k in atom_vector_properties: pmp[k+"_H"]=[[0.0]*len(pm.atom_list)]*len(pm.atom_list)
        #for the polarizability wr2 XH bonds, all matrices for H atoms are 0, so this is not doing anything; uncomment for other cases...
        #for k in atom_tensor_properties: pmp[k+"_H"]=[[[0.0]*len(pm.atom_list)]*len(pm.atom_list)]*len(pm.atom_list)

        #find out nearest atom to each H and add values to its features    
        for a in pm.atom_list:
            if a.symbol.lower()=="h":
                nearest=pm.nearest_atom(a,exclude_H=True)
                nearest_index=nearest.atom_number-1
                a_index=a.atom_number-1
                for k in atom_scalar_properties:
                    pmp[k+"_H"][nearest_index]+=pmp[k][a_index]
                for k in atom_vector_properties:
                    pmp[k+"_H"][nearest_index]+=np.array(pmp[k][a_index])
                #for the polarizability wr2 XH bonds, all matrices for H atoms are 0, so this is not doing anything but adding zeros; uncomment for other cases...
                #for k in atom_tensor_properties:
                #    pmp[k+"_H"][nearest_index]+=np.array(pmp[k][a_index])
                pmp["number_of_H"][nearest_index]+=1

        #create a copy of the keys of pm dictionary with the preffix: "additive_"; during symmetry reduction of graphs, 
        #the properties of equivalent atoms corresponding to "additive" keys will be added, instead of calculating the mean value
        all_pmp_keys=[k for k in pmp.keys()]   #it has to be done this way because we need values, not references
        for k in all_pmp_keys:
                if k not in ["isC","isN","isO","isS","inv_distances"]:  #additive properties will not be calculated for these keys
                    pmp["additive_"+k]=pmp[k]  #copy the same values for "additive_" entries


        #update scalar, vector and tensor properties lists to include also "_H" and "additive_" properties
        atom_scalar_properties=[p for p in pmp.keys() if (type(pmp[p]) in [list,np.ndarray]) 
                                                        and (type(pmp[p][0]) in  [float,np.float32,np.float64]) and p not in exclude_keys ]#and p not in ["isC","isN","isO","isS"]] 

        atom_vector_properties=[p for p in pmp.keys() if (type(pmp[p]) in [list,np.ndarray]) and (type(pmp[p][0]) in  [list,np.ndarray])  
                                                        and (type(pmp[p][0][0]) in [float,np.float32,np.float64])  and p not in exclude_keys]
        
        atom_tensor_properties=[p for p in pmp.keys() if (type(pmp[p]) in [list,np.ndarray]) and (type(pmp[p][0]) in  [list,np.ndarray]) 
                                                        and (type(pmp[p][0][0]) in  [list,np.ndarray]) and p not in exclude_keys]



        #remove H atoms:
        #find out the numbers of the h atoms and non-h atoms
        h_numbers=[i for i in range(len(pm.atom_list)) if pm.atom_list[i].symbol.lower()=="h"]
        no_h_numbers=[i for i in range(len(pm.atom_list)) if pm.atom_list[i].symbol.lower()!="h"]

        #remove properties of H atoms, including rows corresponding of h atoms in vector properties and whole matrices of tensor properties
        for k in atom_scalar_properties+atom_vector_properties+atom_tensor_properties:
            new_pmp=[pmp[k][i] for i in range(len(pm.atom_list)) if i not in h_numbers]
            pmp[k]=new_pmp

        #remove columns corresponding to h atoms in atom_vector_properties and rows corresponding to tensor properties:
        for k in atom_vector_properties:
            new_pmp=[]
            for p in pmp[k]:
                new_pmp_vector=[p[i] for i in range(len(pm.atom_list)) if i not in h_numbers]
                new_pmp.append(new_pmp_vector)
            pmp[k]=new_pmp

        #remove rows and colums of tensor properties:
        for k in atom_tensor_properties:
            new_pmp_tensor=[]
            for p in pmp[k]:
                new_pmp_matrix=np.zeros((len(no_h_numbers),len(no_h_numbers)))
                for i,a in enumerate(no_h_numbers):
                    for j,b in enumerate(no_h_numbers):
                        new_pmp_matrix[i,j]=p[a][b]
                new_pmp_tensor.append(new_pmp_matrix)
            pmp[k]=np.array(new_pmp_tensor)


        #remove H atoms from the molecule object
        pm.remove_atoms("h")
        pm.generate_connections_from_distance(allow_fract_orders=False)
        

        #get the groups of equivalent atoms, their connections, and a graph with equivalent atoms removed (that will be used for overlapping and maping atom numbers) 
        groups_of_equivalent_atoms, connections_between_groups, G=equivalent_atoms_from_molecule(pm)
        #use vf2pp_isomorphism aglrithm in networkx to find a map of each molecular structure graph to the reference molecule 
        #(a posible choice of the reference molecule is the first protonated molecule (the same must be used for protonated and deprotonated!!!!)
        #this map will be used to renumber the atoms in a consistent way.
        map=nx.vf2pp_isomorphism(reference_graph,G,node_label="fingerprint") 
        map=dict(sorted(map.items()))
        #print ("map after sorting"+str(map))
        #print ("group of eq. atoms"+str(groups_of_equivalent_atoms))


        #add the properties of equivalent atoms, depending if they correspond to scalar, vector or tensor properties
        symmetry_method="sum"
        #scalar:
        for key in atom_scalar_properties:
            new_pmp_value=[]
            for eq_atoms in groups_of_equivalent_atoms: 
                prop= [ pmp[key][eqa-1] for eqa in eq_atoms  ] # -1 because eqa starts at 1, not 0    
                if key.startswith("additive_"):    
                    new_pmp_value.append( float(np.sum (prop))  )  
                else:                     
                    new_pmp_value.append( float(np.mean(prop))  )                           
            pmp[key]=new_pmp_value

        #vector
        #sum columns
        for key in atom_vector_properties:
            new_p=np.zeros([len(groups_of_equivalent_atoms),len(groups_of_equivalent_atoms)])
            for i in range(len(groups_of_equivalent_atoms)):
                for j in range(len(groups_of_equivalent_atoms)):
                    #all vector properties (except inv_distances) are very small for not-bound atoms, 
                    #so there is not a big error if they are summed -the number added will be very small for not bound atoms.
                    #bond distances, projected mu, etc, are 0 for not-bound atoms because the distance matrix was zeroed for all atoms whose distance was not smaller than 1.25 times the sum of covalent radius.
                    prop=[ [ pmp[key][atom_i-1][atom_j-1] for atom_j in groups_of_equivalent_atoms[j]   ]   for atom_i in groups_of_equivalent_atoms[i] ] 
                    if  key.startswith("additive_"):
                        new_p[i,j]=np.sum(prop) #sum columns and rows
                    elif "inv_distances"==key:     
                        new_p[i,j]=np.min(prop) #for "inv_distances", instead of the sum use minimum value
                    else:                          
                        new_p[i,j]=np.mean(prop) #sum columns and rows
            pmp[key]=new_p


        
        #tensor properties
        
        atom_tensor_properties=['polarizability_wr2_bonds_with_H_projected_on_bonds',
                                'additive_polarizability_wr2_bonds_with_H_projected_on_bonds'] #detect them automatically when there are more than one tensor properties?
        for key in atom_tensor_properties:
            new_tensor=np.zeros([np.shape(pmp[key])[0],len(groups_of_equivalent_atoms),len(groups_of_equivalent_atoms)])
            #first, iterate through all atoms, next, trhough groups of equivalent atoms
            for k in range(np.shape(pmp[key])[0]):
                for i in range(len(groups_of_equivalent_atoms)):
                    for j in range(len(groups_of_equivalent_atoms)):
                        prop=[ [ pmp[key][k][atom_i-1][atom_j-1] for atom_j in groups_of_equivalent_atoms[j]   ]   for atom_i in groups_of_equivalent_atoms[i] ]
                        if key.startswith("additive_"): 
                            new_tensor[k,i,j]= np.sum(prop)
                        else:                  
                            new_tensor[k,i,j]= np.mean(prop)
            #new_tensor has dimmensions: n_atoms x number_of_groups x number_of_groups  (still, the first dimmension has not been reduced with the symmetry)
            new_reduced_tensor=[]
            for g in groups_of_equivalent_atoms:
                #                                     new_tensor[a-1] selects the n_groups x n_groups matrix in new_tensor corresponding to index=a-1
                #                                   new_tensor matrices of atoms in the same group of equivalent atoms are combined in a list
                #                          and the mean is calculated on axis=0 (dimensions of the resulting matrix: n_groups x n_groups)
                #                          note that the mean, and not the sum, is used.
                #                  the new_reduced_tensor list is built from n_groups matrices
                new_reduced_tensor.append( np.mean( [ new_tensor[a-1] for a in g ],axis=0 )  )
            pmp[key]=np.array(new_reduced_tensor)

            """
            #old code
            new_p=[]
            for eq_atoms in groups_of_equivalent_atoms:
                #calculate average over matrices:
                #each of these matrices represente the response to an electric field applied on the direction of each X-H bond projected over the bonds in the molecule
                #the matrix with most relevant information is the one that correspond to the electric field applied in the direction of the X-H bond that is broken during 
                #deprotonation. It will be recovered by multiplying with the "mask" (0 for all atoms except for the atom X bound to the H that is cleaved, when it is 1)
                #Therefore, in those cases in which there are several symmetry equivalent X-H atoms, it seems more reasonable to calculate the mean and not the sum.
                new_matrix=np.zeros_like(pmp[key][0])
                for eqa in eq_atoms:
                    new_matrix+=pmp[key][eqa-1]
                #instead of the sum, this uses the average so when the value is recovered applying the mask, integers and not fractinal numbers are used
                new_p.append(new_matrix/float(len(eq_atoms))) 

            #sum rows of matrices; this is consistent with the treatment to atom vector properties (sum in one dimmension and deleting redundant info in the other)
            new_pp=[]
            for m in new_p:
                new_ppp=[] 
                for eq_atoms in groups_of_equivalent_atoms:
                    new_row=np.zeros_like(m[0])
                    for eqa in eq_atoms:
                        new_row+=m[eqa-1]
                    new_ppp.append(new_row)
                new_pp.append(np.array(new_ppp))
                    
            #eliminate columns of matrices
            new_ppp=[]
            for m in new_pp:
                new_pppp=np.zeros((len(groups_of_equivalent_atoms),len(groups_of_equivalent_atoms)))
                for i in range(len(m)):
                    for j,eqa in enumerate(groups_of_equivalent_atoms):
                        new_pppp[i,j]=m[i,eqa[0]-1]
                new_ppp.append(new_pppp)
            
            pmp[key]=np.array(new_ppp)
            """


        
        #sort the list elements according to map so all list are equally ordered 
        for key in atom_scalar_properties:
            pmp_sort=np.zeros_like(pmp[key])
            for k,v in map.items(): pmp_sort[k]=pmp[key][v]
            pmp[key]=list(pmp_sort)
        for key in atom_vector_properties:
            pmp_sort_col=np.zeros_like(pmp[key])
            for k,v in map.items(): pmp_sort_col[k,:]=np.array(pmp[key])[v,:]
            pmp_sort_rows=np.zeros_like(pmp_sort_col)
            for k,v in map.items(): pmp_sort_rows[:,k]=pmp_sort_col[:,v]
            pmp[key]=pmp_sort_rows.tolist()


        #remove empty properties
        properties_to_remove=[k for k in pmp.keys() if k.startswith("rel_") and not k.endswith("_H")]
        for k in properties_to_remove: del(pmp[k])

        props.append(pmp)
        
    return props,nx.adjacency_matrix(reference_graph).todense()    


#flag to determine wether to save or not the lists of scalar and vector propertie's names
write_keys_list=True
#lists of the scalar and vector properties' names
atom_prop_scalar_keys,atom_prop_vector_keys=[],[]
#to know how much is done
counter=0
for compn in rows:
    counter+=1
    if compn in already_done or compn in exclude:continue 

    #get the suffixes of the protonated and deprotonated files
    if str(compn.split("_")[1]).startswith("cation"):   protonated_str="-cation";  deprotonated_str="-neut"
    if str(compn.split("_")[1]).startswith("2cation"):  protonated_str="-2cation"; deprotonated_str="-cation"
    if str(compn.split("_")[1]).startswith("3cation"):  protonated_str="-3cation"; deprotonated_str="-2cation"
    if str(compn.split("_")[1]).startswith("4cation"):  protonated_str="-4cation"; deprotonated_str="-3cation"
    if str(compn.split("_")[1]).startswith("5cation"):  protonated_str="-5cation"; deprotonated_str="-4cation"
    if str(compn.split("_")[1]).startswith("6cation"):  protonated_str="-6cation"; deprotonated_str="-5cation"
    if str(compn.split("_")[1]).startswith("neut"):     protonated_str="-neut";    deprotonated_str="-an"
    if str(compn.split("_")[1]).startswith("an"):       protonated_str="-an";      deprotonated_str="-2an"
    if str(compn.split("_")[1]).startswith("2an"):      protonated_str="-2an";     deprotonated_str="-3an"
    if str(compn.split("_")[1]).startswith("3an"):      protonated_str="-3an";     deprotonated_str="-4an"
    if str(compn.split("_")[1]).startswith("4an"):      protonated_str="-4an";     deprotonated_str="-5an"
    if str(compn.split("_")[1]).startswith("5an"):      protonated_str="-5an";     deprotonated_str="-6an"
    protonated_molecules_file_names,protonated_HL_molecules_file_names=[],[]
    deprotonated_molecules_file_names,deprotonated_HL_molecules_file_names=[],[]


    #get name of files
    optimization_files=[  ff for ff in os.listdir(optimization_files_route) if (("hess" not in ff) and ("chrg" not in ff) and ("m06" not in ff) and ("fake" not in ff) and ("cpcm" not in ff) and "_nbo" not in ff)]
    for f in  optimization_files: #[  ff for ff in os.listdir(optimization_files_route) if (("hess" not in ff) and ("chrg" not in ff) and ("m06" not in ff) and ("fake" not in ff) and ("cpcm" not in ff) and "_nbo" not in ff)]:
        if f.startswith( str(compn).split("_")[0]+protonated_str): 
            protonated_molecules_file_names.append(optimization_files_route+f)
            protonated_HL_molecules_file_names.append(HL_route+f.split(".out")[0]+HL_text+".out")
        if f.startswith( str(compn).split("_")[0]+deprotonated_str): 
            deprotonated_molecules_file_names.append(optimization_files_route+f)
            deprotonated_HL_molecules_file_names.append(HL_route+f.split(".out")[0]+HL_text+".out")

    #print what we are doing
    print (" "*80,end="\n")
    print (str(counter)+"/"+str(len(rows))+"  loading "+str(len(protonated_molecules_file_names))+" protonated molecules and "+str(len(deprotonated_molecules_file_names))+"deprotonated molecules for:"+ str(compn)+"                                            ", end="\n")

    #additional property that will be read from HL molecules: polarizability matrix

    if orca_version=="5":
        quadrupole_moment=Molecular_structure.Property(name="quadrupole_moment", 
                                text_before="                XX           YY           ZZ           XY           XZ           YZ",#orca5
                                text_after="(a.u.)")
    elif orca_version=="6":
        quadrupole_moment=Molecular_structure.Property(name="quadrupole_moment", 
                                text_before="                XX              YY              ZZ              XY              XZ              YZ",#orca6
                                text_after="(a.u.)")

    polarizability_matrix=Molecular_structure.Property(name="polarizability_matrix",
                                                    text_before="The raw cartesian tensor (atomic units):",
                                                    text_after="diagonalized tensor:",format="float")
    
    extra_props=[polarizability_matrix,quadrupole_moment]
                            

    #get list of Molecular_structure objects for HL and geometry optimization and for protonated and deprotonated
    protonated_molecules=[Molecular_structure.Molecular_structure(f,"last") for f in protonated_molecules_file_names]
    protonated_molecules_HL=[Molecular_structure.Molecular_structure(f,"last",extra_props) for f in protonated_HL_molecules_file_names]
    deprotonated_molecules=[Molecular_structure.Molecular_structure(f,"last") for f in deprotonated_molecules_file_names]
    deprotonated_molecules_HL=[Molecular_structure.Molecular_structure(f,"last",extra_props) for f in deprotonated_HL_molecules_file_names]

    #this is important: we do not want weak interactions in the adjancency matrix:
    for m in protonated_molecules_HL+deprotonated_molecules_HL: 
        m.generate_connections_from_distance(allow_fract_orders=False)
        m.remove_non_covalent_connections()


    #read the number of equivalent structures from the annotated_atoms.py file (used for calculating populations)
    protonated_identical_molecules=[get_n_identical_structures(filename) for filename in protonated_molecules_file_names]
    deprotonated_identical_molecules=[get_n_identical_structures(filename) for filename in deprotonated_molecules_file_names]

    #calculate energies for MB populations and populations
    protonated_gibbs_free_energies_optz=   np.array([m.gibbs_free_energy for m in protonated_molecules])             * hartrees_to_kal_mol
    protonated_zero_point_energies_optz=   np.array([m.zero_point_energy for m in protonated_molecules])             * hartrees_to_kal_mol
    protonated_sp_energies_optz=           np.array([m.electronic_energy for m in protonated_molecules])             * hartrees_to_kal_mol
    protonated_sp_energies=                np.array([m.electronic_energy for m in protonated_molecules_HL])          * hartrees_to_kal_mol
    protonated_gibbs_free_energies=        protonated_gibbs_free_energies_optz-protonated_sp_energies_optz+protonated_sp_energies
    protonated_zero_point_energies=        protonated_zero_point_energies_optz-protonated_sp_energies_optz+protonated_sp_energies
    deprotonated_gibbs_free_energies_optz=   np.array([m.gibbs_free_energy for m in deprotonated_molecules])             * hartrees_to_kal_mol
    deprotonated_zero_point_energies_optz=   np.array([m.zero_point_energy for m in deprotonated_molecules])             * hartrees_to_kal_mol
    deprotonated_sp_energies_optz=           np.array([m.electronic_energy for m in deprotonated_molecules])             * hartrees_to_kal_mol
    deprotonated_sp_energies=                np.array([m.electronic_energy for m in deprotonated_molecules_HL])          * hartrees_to_kal_mol
    deprotonated_gibbs_free_energies=        deprotonated_gibbs_free_energies_optz-deprotonated_sp_energies_optz+deprotonated_sp_energies
    deprotonated_zero_point_energies=        deprotonated_zero_point_energies_optz-deprotonated_sp_energies_optz+deprotonated_sp_energies
    #to prevent overflow, use relative values:
    protonated_gibbs_free_energies=protonated_gibbs_free_energies-np.min(protonated_gibbs_free_energies)
    protonated_zero_point_energies=protonated_zero_point_energies-np.min(protonated_zero_point_energies)
    protonated_sp_energies=protonated_sp_energies-np.min(protonated_sp_energies)
    deprotonated_gibbs_free_energies=deprotonated_gibbs_free_energies-np.min(deprotonated_gibbs_free_energies)
    deprotonated_zero_point_energies=deprotonated_zero_point_energies-np.min(deprotonated_zero_point_energies)
    deprotonated_sp_energies=deprotonated_sp_energies-np.min(deprotonated_sp_energies)
    if   weighting=="gibbs":
        protonated_molecules_populations= np.exp( -protonated_gibbs_free_energies/RT )*(protonated_identical_molecules)
        deprotonated_molecules_populations= np.exp( -deprotonated_gibbs_free_energies/RT )*(deprotonated_identical_molecules)            
    elif weighting=="zero":
        protonated_molecules_populations= np.exp( -protonated_zero_point_energies/RT )*(protonated_identical_molecules)
        deprotonated_molecules_populations= np.exp( -deprotonated_zero_point_energies/RT )*(deprotonated_identical_molecules)
    elif weighting=="sp":
        protonated_molecules_populations= np.exp( -protonated_sp_energies/RT )*(protonated_identical_molecules)
        deprotonated_molecules_populations= np.exp( -deprotonated_sp_energies/RT )*(deprotonated_identical_molecules)
    sum_protonated_molecules_populations=np.sum(protonated_molecules_populations)
    protonated_molecules_populations= protonated_molecules_populations / sum_protonated_molecules_populations 
    sum_deprotonated_molecules_populations=np.sum(deprotonated_molecules_populations)
    deprotonated_molecules_populations= deprotonated_molecules_populations / sum_deprotonated_molecules_populations 





    # for protonated molecules:
    pmp,adj_matrix=get_lists_of_properties_for_symmetry_reduced_graph(protonated_molecules_HL,protonated_molecules,protonated_HL_molecules_file_names,protonated_molecules_HL[0])
    #for deprotonated_molecules: (note that reference molecule is the protonated, not the deprotonated, because it must be the same in both calls)
    dmp,_=get_lists_of_properties_for_symmetry_reduced_graph(deprotonated_molecules_HL,deprotonated_molecules,deprotonated_HL_molecules_file_names,protonated_molecules_HL[0])

    #Maxwell-Boltzmann average properties of each atom and difference between protonated and deprotonated 
    #returns a list (one ELEMENT for each atom) of dictionaries with the properties
    averaged_props=[]

    #build all tautomer pairs involving one protonated and one deprotonated structure, calculate their population, keep only those contributing to 99% of population
    #and group them according to the atom that loose the proton (using the difference in the number_of_H between protonated and deprotonated)
    pairs=[]  
    for p_tautomer in range(0,len(protonated_molecules_HL)):
        for d_tautomer in range(0,len(deprotonated_molecules_HL)):
            tautomer_population=protonated_molecules_populations[p_tautomer]*deprotonated_molecules_populations[d_tautomer]
            tautomer_H_position=[numb_h_prot-numb_h_deprot for numb_h_prot,numb_h_deprot in  zip(pmp[p_tautomer]["number_of_H"],dmp[d_tautomer]["number_of_H"])]
            pairs.append([p_tautomer,d_tautomer,tautomer_population,tautomer_H_position])
            #sort pairs according to their "population"(=contribution) and get rid off those whose contribution aggregates 1% 
    pairs_sorted=sorted(pairs, key=lambda x: x[2],reverse=True)
    pairs_valid=[pairs_sorted[0]]; sum=pairs_sorted[0][2]; p=0
    while sum<0.99:
        p+=1
        pairs_valid.append(pairs_sorted[p])
        sum+=pairs_sorted[p][2]
    #group surviving pairs according to "tautomer_H_position": all groups with the same "tautomer_H_position" vector are "conformers"
    #and their properties can be aggregated separately without loss of information since all atoms have the same "role" (alpha, beta, etc atom with respect to the position of deprotonateion, etc)            
    groups_of_tautomer_pairs=[]
    while (len(pairs_valid))>0:
        #initialize current group with the first pair in pairs_valid
        group=[pairs_valid[0]]
        delete_indexes=[0]
        for i in range(1,len(pairs_valid)):
            #append all pairs with the same "tautomer_H_position"
            if group[0][-1]==pairs_valid[i][-1]: group+=[pairs_valid[i]];delete_indexes.append(i)
        #delete pairs already added to a group (so the first element of pairs_valid can be used to start a new group in the next cycle of the "while" bucle)
        pairs_valid=[p for i,p in enumerate(pairs_valid) if i not in delete_indexes]
        #store current group in the groups of tautomers
        groups_of_tautomer_pairs.append(group)



    #list of properties dictionaries, each is the weighted average property of each group of pairs
    prop_tautomers_groups=[]
    for group_of_tautomer_pair in groups_of_tautomer_pairs:

        prop_group_tautomer_pair={}
        #relative population of this group, calculating by adding the sum of the populations of all of their pairs:
        group_population=np.sum([tp[2] for tp in group_of_tautomer_pair] )

        #note that this transformation may lead to the same masks from different "H_change" vectors. For example, for 123triazole_cation->neut, there are two groups:
        # cation2->neut2  with 0,-1,2 -the 2 is created by joining 2 atoms by symmetry- and cation2->neut with 0,0,1. In both cases the mask will be 0,0,1 (this is what one wants to 
        # project the properties of alpha atom), so it is good that this change is not made before so both possibilities are in two different groups
        H_change=[1.0 if i>0 else 0.0 for i in group_of_tautomer_pair[0][-1]]
        sum_H_change=np.sum(H_change)
        prop_group_tautomer_pair["number_of_H_difference"]=[i/sum_H_change for i in H_change]
        group_of_tautomer_pair[0][-1]=[i/sum_H_change for i in H_change] #new??

        #initialize values:
        for k in pmp[0].keys():
            if k!="number_of_H" and k.startswith("additive_")==False:
                prop_group_tautomer_pair[k+"_protonated"],prop_group_tautomer_pair[k+"_deprotonated"],prop_group_tautomer_pair[k+"_difference"]=np.zeros_like(pmp[0][k]),np.zeros_like(pmp[0][k]),np.zeros_like(pmp[0][k])                   
            elif k!="number_of_H" and k.startswith("additive_"):
                #difference properties are calculated from keys with prefix "additive_"
                prop_group_tautomer_pair[k.replace("additive_","")+"_difference"]=np.zeros_like(pmp[0][k])

        #assign values, weighting with the contribution of the tautomer pair
        #currently, the only tensor property that is processed is this; automatize this?
        tensor_properties=['polarizability_wr2_bonds_with_H_projected_on_bonds']#,'polarizability_wr2_bonds_with_H_projected_on_bonds_H']
        
        for k in [key for key in pmp[0].keys() if key.startswith("additive_")==False]: #only iterate through keys without the "additive_" prefix 
            if k!="number_of_H" and k not in tensor_properties:
                for tautomer_pair in group_of_tautomer_pair:
                    #tautomer_pair[2] is the contribution, and it is used to weight the node (atomic) features:
                    if  type(pmp[0][k][0]) in [float,np.float32,np.float64]: w_f=tautomer_pair[2]
                    #to facilitate its use by GNNs, edge features are not weighted by the group population
                    #(similarly as the adjacency matrix is not weighted by the group population)
                    #to correct this effect, the weighting factor is divided by the sum of the weighting factors for all the pairs in this group (the "group population")
                    #note that group_population<1.0, so dividing by it implies scaling up w_f
                    #If it is neccessary to extract features of a particular bond for treating it as a conventional feature (not as an edge feature in a GNN)
                    #the weighted_mask can be used instead of the conventional mask.
                    elif (type(pmp[0][k][0]) in [list,np.ndarray]) and (type(pmp[0][k][0][0]) in [float,np.float32,np.float64]):   w_f=tautomer_pair[2]/group_population          


                    prop_group_tautomer_pair[k+"_protonated"]+=w_f*np.array(pmp[tautomer_pair[0]][k])    #tautomer_pair[2] is the contribution; tautomer_pair[0] is the property of the protonated
                    prop_group_tautomer_pair[k+"_deprotonated"]+=w_f*np.array(dmp[tautomer_pair[1]][k])
                    if "additive_"+k in pmp[0].keys(): # if the additive_ property exists:
                        prop_group_tautomer_pair[k+"_difference"]+=w_f*(    np.array(pmp[tautomer_pair[0]]["additive_"+k])-np.array(dmp[tautomer_pair[1]]["additive_"+k])   )
                    else: #if preffixed "additive_key" does not exist, use the values without the prefix
                        prop_group_tautomer_pair[k+"_difference"]+=w_f*(    np.array(pmp[tautomer_pair[0]][k])-np.array(dmp[tautomer_pair[1]][k])   )

            elif k in tensor_properties:
                for tautomer_pair in group_of_tautomer_pair:
                    #prop_group_tautomer_pair[k+"_protonated"]= np.array().dot (  pmp[tautomer_pair[0]][k] )
                    prop_group_tautomer_pair[k+"_protonated"]=np.zeros_like(pmp[tautomer_pair[0]][k][0])
                    prop_group_tautomer_pair[k+"_deprotonated"]=np.zeros_like(dmp[tautomer_pair[1]][k][0])  #note: this should be zero if the deprotonated form has no more H atoms
                    prop_group_tautomer_pair[k+"_difference"]=np.zeros_like(dmp[tautomer_pair[1]][k][0])
                    #weighting factor is calculated as for other edge features:
                    w_f=tautomer_pair[2]/group_population  
                    #use the "mask" to reduce the first tensor dimension; a matrix will be obtained
                    for w,m in zip(tautomer_pair[-1],pmp[tautomer_pair[0]][k]): prop_group_tautomer_pair[k+"_protonated"]+=w*m*w_f
                    for w,m in zip(tautomer_pair[-1],dmp[tautomer_pair[1]][k]): prop_group_tautomer_pair[k+"_deprotonated"]+=w*m*w_f
                    if "additive_"+k in pmp[0].keys(): # if the additive_ property exists:
                        for w,m,mm in zip(tautomer_pair[-1],pmp[tautomer_pair[0]]["additive_"+k],dmp[tautomer_pair[1]]["additive_"+k]):
                            prop_group_tautomer_pair[k+"_difference"]+=w*w_f*(m-mm)
                    else: 
                        for w,m,mm in zip(tautomer_pair[-1],pmp[tautomer_pair[0]][k],dmp[tautomer_pair[1]][k]):
                            prop_group_tautomer_pair[k+"_difference"]+=w*w_f*(m-mm)
                    #old
                    #prop_group_tautomer_pair[k+"_difference"]=prop_group_tautomer_pair[k+"_protonated"]-prop_group_tautomer_pair[k+"_deprotonated"]


        
        prop_tautomers_groups.append(prop_group_tautomer_pair)

    #build the disjoint equilibrium graph from the properties of each group of tautomers
    number_of_subgraphs=len(prop_tautomers_groups)
    number_of_atoms=adj_matrix.shape[0]
    
    G={}
    G["name"]=compn
    G["y"]=compn
    #adjacency matrix is a block-diagonal repetition of the subgraphs adjacency matrices (that turn out to be identical)
    G["a"]=np.kron(np.eye(number_of_subgraphs),adj_matrix)  #https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array


    #some atomic features were extracted but are useless, so they are deleted 
    #in an ideal world they should not even been extracted, but this is simpler, and also in an ideal world Kamala won, so...

    all_features=prop_tautomers_groups[0].keys()
    #delete all relative features not involving H atoms:
    delete_features=[f for f in all_features if ( f.startswith("rel_") and "_H" not in f )]
    #delete isC, isN... features for deprotonated and for the difference
    delete_features+=[f for f in all_features if (f.startswith("is") and (f.endswith("deprotonated") or f.endswith("difference")))]
    delete_features+=[ "lea_atom_overall_surf_area_protonated", "lea_atom_overall_surf_area_deprotonated", "lea_atom_overall_surf_area_difference",# it is the same that was calculated for ESP 
                        "lea_atom_pos_surf_area_protonated", "lea_atom_pos_surf_area_deprotonated", "lea_atom_pos_surf_area_difference",# it is zero everywhere
                        "lea_atom_neg_surf_area_protonated", "lea_atom_neg_surf_area_deprotonated", "lea_atom_neg_surf_area_difference",# it is the same that overall surf area calculated for ESP
                        "lea_atom_pos_avg_protonated","lea_atom_pos_avg_deprotonated","lea_atom_pos_avg_difference",
                        "lea_atom_neg_avg_protonated","lea_atom_neg_avg_deprotonated","lea_atom_neg_avg_difference",
                        #"lea_atom_neg_variance_protonated","lea_atom_neg_variance_deprotonated","lea_atom_neg_variance_difference",
                        "lea_atom_overall_variance_protonated", "lea_atom_overall_variance_deprotonated", "lea_atom_overall_variance_difference", #this should includ: "lea_atom_neg_variance" instead of , “lea_atom_overall_variance”, but a bug makes multwfn plot NaN in lea_atom_overall_variance, 
                        "lea_atom_pos_variance_protonated", "lea_atom_pos_variance_deprotonated", "lea_atom_pos_variance_difference",#so the information is taken from lea_atom_neg_variance instead and attributed to "atom-var-LEA"
                        "number_of_H_H_protonated", "number_of_H_H_deprotonated", "number_of_H_H_difference",   #the same that number_of_H
                        "lea_atom_overall_surf_area_H_protonated", "lea_atom_overall_surf_area_H_deprotonated", "lea_atom_overall_surf_area_H_difference",
                        "lea_atom_pos_surf_area_H_protonated", "lea_atom_pos_surf_area_H_deprotonated", "lea_atom_pos_surf_area_H_difference",
                        "lea_atom_neg_surf_area_H_protonated", "lea_atom_neg_surf_area_H_deprotonated", "lea_atom_neg_surf_area_H_difference",
                        "lea_atom_overall_variance_H_protonated", "lea_atom_overall_variance_H_deprotonated", "lea_atom_overall_variance_H_difference", 
                        "lea_atom_pos_variance_H_protonated", "lea_atom_pos_variance_H_deprotonated", "lea_atom_pos_variance_H_difference"
                        "lea_atom_pos_avg_H_protonated","lea_atom_pos_avg_H_deprotonated","lea_atom_pos_avg_H_difference",
                        "lea_atom_neg_avg_H_protonated","lea_atom_neg_avg_H_deprotonated","lea_atom_neg_avg_H_difference",
                        "lea_atom_neg_variance_H_protonated","lea_atom_neg_H_variance_deprotonated","lea_atom_neg_variance_H_difference",
                        ]                 

    for g in range(number_of_subgraphs):
        for k in delete_features: 
            if k in prop_tautomers_groups[g].keys(): del(prop_tautomers_groups[g][k])

    #get the scalar and vector properties list: to save some space in the (huge) json file, the keys are stored in the first line and for each compn
    #the values are listed in the same order that the keys are given (this must be taken into account to read the json file!!!!)
    #therefore, it is important that the order in consistent along all compns!!!. 
    #the lists of scalar and vector keys is read from the json file if it already existed when the script was run, or determined if it has not been done before.
    reset_key_list=True # flag to determine wether the keys are going to be determined
    if atom_prop_scalar_keys==[] or atom_prop_vector_keys==[]: # if there is not yet a list of atom or vector properties keys:
        if graphs_extracted_file in os.listdir():  #if there is already a json file that will be continued:
            with open (graphs_extracted_file,"r") as f: keys=json.loads(f.readlines()[0]) #read the keys
            atom_keys_match_pub=all( [   k  in inv_atom_features_publication_names.keys() for k in keys["feature_keys"]  ]   )
            vector_keys_match_pub=all( [   k  in inv_vector_features_publication_names.keys() for k in keys["vector_keys"]  ]   )
            if atom_keys_match_pub and vector_keys_match_pub:
                atom_prop_scalar_keys=[inv_atom_features_publication_names[k] for k in keys["feature_keys"]]  #translate the names of the properties in the publication to the names found here.
                atom_prop_vector_keys=[inv_vector_features_publication_names[k] for k in keys["vector_keys"]] #translate the names of the properties in the publication to the names found here.
                #atom_prop_scalar_keys,atom_prop_vector_keys= keys["feature_keys"],keys["vector_keys"]
                reset_key_list=False
        if reset_key_list:
            write_keys_list=True
            atom_prop_scalar_keys=[k for k in prop_tautomers_groups[0].keys() if (type(prop_tautomers_groups[0][k][0]) in [float,np.float32,np.float64])  ]
            atom_prop_vector_keys=[k for k in prop_tautomers_groups[0].keys() if (type(prop_tautomers_groups[0][k][0]) in [list,np.ndarray])  and 
                                                                        (type(prop_tautomers_groups[0][k][0][0]) in [float,np.float32,np.float64]) ]
    #check if there is any difference in the number of properties determined and the expected list of properties 
    current_keys= prop_tautomers_groups[0].keys()
    if any([k not in current_keys for k in atom_prop_scalar_keys+atom_prop_vector_keys]) or  any([k not in atom_prop_scalar_keys+atom_prop_vector_keys for k in current_keys]):
        s= "possible error: number of atomic properties determined for: "+compn+"are different than for the rest of data"
        print (s)
        with open("report_error.txt","a") as f: f.write(s)




    #build the feature vectors X; usually X is number_of_atoms x number of features, but to deal with different tautomers, it will correspond to the feature vectors
    #of a disjoint graph composed of N "subgraphs", one for each of the N tautomer combinations; each subgraph's X is number_of_atoms x number of features, and therefore
    #the graph X is (N * number_of_atoms) x number of features
    X=[]
    for k in atom_prop_scalar_keys: 
        x=[]
        for g in range(number_of_subgraphs):
            x+=[ prop_tautomers_groups[g][k][a].tolist() for a in range(number_of_atoms)  ]
        X.append(x)
    G["x"]=np.array(X)

    #build the edge matrices; usually e dimensions are  number_of_atoms x number_of_atoms, but to deal with different tautomers, it will correspond to edge matrices of
    #a disjoint graph composed of N "subgraphs" corresponding to each tautomer pair combination        
    E_matrices=[]
    from scipy.linalg import block_diag
    for k in atom_prop_vector_keys:
        e_subgraphs_matrices=[]
        for g in range(number_of_subgraphs):
            ee=[]
            for a in range(number_of_atoms):  ee.append(prop_tautomers_groups[g][k][a])
            e_subgraphs_matrices.append(ee)
        e_for_each_key=e_subgraphs_matrices[0]
        for e_subgraph_matrix in e_subgraphs_matrices[1:]: e_for_each_key=block_diag(e_for_each_key,e_subgraph_matrix)
        E_matrices.append(e_for_each_key)
        #if k=='polarizability_wr2_bonds_with_H_projected_on_bonds_difference': print (e_for_each_key)
            
    G["e"]=E_matrices



    #write json file
    #dictionary containing the publication name of the features (translates the script names of the features)
    features_dict= {"feature_keys":[atom_features_publication_names[k] if k in atom_features_publication_names.keys() else k for k in atom_prop_scalar_keys  ] ,
                    "vector_keys": [vector_features_publication_names[k] if k in vector_features_publication_names.keys() else k for k in atom_prop_vector_keys]}  
    
    #calculate the mask and the weighted_mask
    mask=np.concatenate([ np.array(grps[0][-1]) for grps in groups_of_tautomer_pairs])
    weighted_mask=np.concatenate([ np.sum([ g[2]  for g in grps   ])*np.array(grps[0][-1]) for grps in groups_of_tautomer_pairs])
    #weighted_mask=np.concatenate([ grps[0][2]*np.array(grps[0][-1]) for grps in groups_of_tautomer_pairs])
    #print ("mask"+str(mask))
    #print ("weighted_mask"+str(weighted_mask))
    G["mask"]=mask
    G["weighted_mask"]=weighted_mask

    if  write_keys_list:
        line=json.dumps(features_dict)
        with open(graphs_extracted_file,"w") as f: f.write(line);f.write("\n")
        write_keys_list=False


    #generate a jasonized dictionary to write in file:
    line= json.dumps(G,cls=NpEncoder)
    with open(graphs_extracted_file,"a") as f: f.write(line);f.write("\n")
    



