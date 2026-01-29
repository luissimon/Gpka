#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
sys.path.append('../import')
import Molecular_structure
import numpy as np
import pandas as pd
import json


#for controlling time of execution of different tasks:
import time
time_total=0.0
start_time=time.time()
time_for_loading_molecules=0.0

time_for_homo_lumo_gap=0.0
time_for_reading_mol_properties=0.0
time_for_filling_pd_series=0.0


#the text in the file names for HL output files:
# CHANGE THIS FOR DIFFERENT LEVELS OF THEORY!!!! 

if "-weighting" in sys.argv:  weighting=sys.argv[sys.argv.index("-weighting")+1]
else: weighting="gibbs" # "gibbs", "sp", or "zero-point"
if "-lot" in sys.argv:  level_of_theory=sys.argv[sys.argv.index("-lot")+1]
else: level_of_theory="sM06" # "M06","sM06","swb97xd","wb97xd","pbeh3c"

if weighting not in ["gibbs","sp","zero"]: print ("weighting should be gibbs, sp, or zero"); sys.exit()
if level_of_theory not in ["M06","sM06","swb97xd","wb97xd","pbeh3c"]: print ("lot must be M06, sM06, swb97xd, wb97xd, or pbeh3c"); sys.exit()

if level_of_theory=="wb97xd": HL_text="_wb97xd_chrg"
elif level_of_theory=="swb97xd": HL_text="_swb97xd_chrg"
elif level_of_theory=="sM06": HL_text="_sm06_chrg"
elif level_of_theory=="M06": HL_text="_m06_chrg"
elif level_of_theory=="pbeh3c": HL_text="_pbeh3c_chrg"
root_route="/Users/luissimon/Documents/proyecto2017/pka/"  #in the mac
root_route="/home/lsimon/jobs/pka/"                        #in the server          

#NAME OF FILES:
#the file from where the names and pka values are obtained:
#labels_csv_file_name="labels-refined-extra2.csv"
labels_csv_file_name="labels-all-inchy9.csv"
#the csv file where results will be deposited:
if "-out" in sys.argv: output_suffix=sys.argv[sys.argv.index("-out")+1]  
else: ouptut_suffix="22"
values_extracted_file="values_extracted-"+weighting+"-"+level_of_theory+"."+output_suffix+".csv"
start_index=0
stop_index=1000000

#the place where .csv file with pka values and names lives 
labels_route= root_route
#labels_route= "/Users/luissimon/Desktop/finalcamgrid/cpcm/nuevos/n/"
#the place where hess and cpcm files live
routes=[root_route+"/output_files/optmization/"]
#        ,
#        "/Users/luissimon/Documents/proyecto2017/pka/output_files/all/crest/",
#        "/Users/luissimon/Documents/proyecto2017/pka/output_files/all/crest_tautomer/"]


#HL_routes=["/Users/luissimon/Documents/proyecto2017/pka/output_files/SP-m06/"]

#the place where sp calculation output files, multiwfn json and cpcm files live
HL_route=root_route+"/output_files/SP-"+level_of_theory+"/"
#the place where CREST-gfn2xtb with one (two) explicit water(s) files live
one_water_route=labels_route+"output_files/1water/"
#two_water_route=labels_route+"output_files/2water/"
#three_water_route=labels_route+"output_files/3water/"
#four_water_route=labels_route+"output_files/4water/"
#the place where nbo output files live
nbo_route=labels_route+"output_files/nbo/"+level_of_theory+"/"


#the csv file where results will be deposited:



#the csv file containing the experimental pka values
labels=pd.read_csv(labels_route+labels_csv_file_name,encoding='unicode_escape')
labels.set_index("compn",inplace=True)
#labels.drop(['alternative pka','alternative ref.','is Lange?'],axis=1,inplace=True)
#labels.drop(['is Lange?'],axis=1,inplace=True)
labels.dropna(how='all', axis=1, inplace=True)
print (labels.head())
#labels.drop(['from','to',"Unnamed: 6","Unnamed: 7"],axis=1,inplace=True)
#labels.drop(['from','to'],axis=1,inplace=True)


#HL_text="_swb97xd_chrg"
#HL_text="_wb97xd_chrg"
#HL_text="_sm06_chrg"

nbo_text="_swb97xd_nbo"
one_water_text="_1wat.conformers.gfn2_gfnff_gbsa.xyz"
#two_water_text="_2wat.conformers.gfn2_gfnff_gbsa.xyz"
#three_water_text="_3wat.conformers.gfn2_gfnff_gbsa.xyz"
#four_water_text="_4wat.conformers.gfn2_gfnff_gbsa.xyz"


#some constants and conversing fators
# CHANGE THIS FOR DIFFERENT LEVELS OF THEORY!!!! 
ref_DG= (-76.67717317--76.26868648)   #= 0.40848669 hartrees; deltaG of H2O  --> H3O+; adding G(protonated)-G(desprotonated) yields deltaG for  protonated + H2O --> deprotonated + H3O+
ref_SP= (-76.694104982182--76.272189453193)  #deltaSP of H2O  --> H3O+ ; 
ref_zpE=ref_SP+(0.03602916-0.02183427)  #deltazPE of H2O  --> H3O+ ; 
if HL_text=="_m06_chrg":
    ref_SP_HL= (-76.857719138307 --76.443164963397)  #deltaSP of H2O  --> H3O+ , M062x/Def2-TZVPPD level of theory
elif HL_text=="_sm06_chrg":
    ref_SP_HL= (-76.766723280548 --76.353627644811)  #deltaSP of H2O  --> H3O+ , M062x/Def2-SVPD level of theory
elif HL_text=="_pbeh3c_chrg":
    ref_SP_HL= (-76.694108328376--76.272230386853)   #deltaSP of H2O  --> H3O+ ; PBEh3c level of theory
elif HL_text=="_swb97xd_chrg":
    ref_SP_HL= (-76.784414858073 --76.368594609196)  #deltaSP of H2O  --> H3O+ , wb97xd/Def2-SVPD level of theory
elif HL_text=="_wb97xd_chrg":
    ref_SP_HL= (-76.875298892792 --76.456645992415)  #deltaSP of H2O  --> H3O+ , wb97xd/Def2-TZVPPD level of theory


ref_DG_HL=ref_SP_HL+ref_DG-ref_SP
hartrees_to_kal_mol=627.5095
RT=0.594  # cal/kmol(298 K)


             






#method to get quantitative analysis of molecular surface based on cpcm file
#this is an analogue to the qams analysis done by multwfn but using the smd generated cavity, charges and potential
#returns a dictionary analogue to the dictionary read from multwfn calculations
def get_smd_cavity_qams(cpcm_file): 
    cpcm_qams={}
    with open(cpcm_file,"r") as f: lines=f.readlines()  
    n_atoms=int(lines[0].split()[0])
    n_points=int(lines[1].split()[0])
    volume=float(lines[13].split()[0])
    atom_indxs,areas,potentials,charges=[],[],[],[]
    start=lines.index("# SURFACE POINTS (A.U.)    (Hint - charge NOT scaled by FEps)\n")+3
    useful_lines=lines[start:-1]
    for l in useful_lines:
        w=l.split()
        areas.append(float(w[3]))
        potentials.append(float(w[4]))
        charges.append(float(w[5]))
        atom_indxs.append(int(w[9]))
    """
    #pos_potentials=[v for v in potentials if v>0]
    pos_potentials=[v if v>0 else 0.0 for v in potentials]
    #pos_areas=[a for a,v in zip(areas,potentials) if v>0]
    pos_areas=[a if v>0 else 0 for a,v in zip(areas,potentials)]
    #pos_atom_indxs=[i for i,v in zip(atom_indxs,potentials) if v>0]
    pos_atom_indxs=[i if v>0 else 0 for i,v in zip(atom_indxs,potentials)]
    #neg_potentials=[v for v in potentials if v<=0]
    neg_potentials=[v if v<=0 else 0 for v in potentials]
    #neg_areas=[a for a,v in zip(areas,potentials) if v<=0]
    neg_areas=[a if v<=0 else 0 for a,v in zip(areas,potentials)]
    #neg_atom_indxs=[i for i,v in zip(atom_indxs,potentials) if v<=0]
    neg_atom_indxs=[i if v<=0 else 0 for i,v in zip(atom_indxs,potentials) ]
    """
    areas=np.array(areas)
    potentials=np.array(potentials)
    charges=np.array(charges)
    pos_potential_mask=np.array([v>0 for v in potentials])
    neg_potential_mask=np.array([v<0 for v in potentials])

    cpcm_qams["cpcm_qams_volume"]=volume 
    cpcm_qams["cpcm_qams_min_val"]=np.min(potentials) 
    cpcm_qams["cpcm_qams_max_val"]=np.max(potentials)
    cpcm_qams["cpcm_qams_overall_surf_area"]=np.sum(areas)
    cpcm_qams["cpcm_qams_pos_surf_area"]=np.sum(areas[pos_potential_mask])
    cpcm_qams["cpcm_qams_neg_surf_area"]=np.sum(areas[neg_potential_mask])
    cpcm_qams["cpcm_qams_overall_avg_value"]=np.mean(potentials)
    if np.sum(pos_potential_mask)!=0: 
        cpcm_qams["cpcm_qams_pos_avg_value"]=np.mean(potentials[pos_potential_mask]) 
        cpcm_qams["cpcm_qams_pos_variance"]=np.var(potentials[pos_potential_mask])  
    else: 
        cpcm_qams["cpcm_qams_pos_avg_value"]=0
        cpcm_qams["cpcm_qams_pos_variance"]=0
    if np.sum(neg_potential_mask)!=0:
        cpcm_qams["cpcm_qams_neg_avg_value"]=np.mean(potentials[neg_potential_mask])
        cpcm_qams["cpcm_qams_neg_variance"]=np.var(potentials[neg_potential_mask])
    else: 
        cpcm_qams["cpcm_qams_neg_avg_value"]=0
        cpcm_qams["cpcm_qams_neg_variance"]=0
    if np.sum(cpcm_qams["cpcm_qams_neg_variance"])!=0 and np.sum(cpcm_qams["cpcm_qams_pos_variance"])!=0:
        cpcm_qams["cpcm_qams_overall_variance"]=cpcm_qams["cpcm_qams_pos_variance"]+cpcm_qams["cpcm_qams_neg_variance"] # as defined in p.159 of multwfn manual
    else:
        cpcm_qams["cpcm_qams_overall_variance"]=0
    cpcm_qams["cpcm_qams_Pi"]=np.mean(np.abs(potentials-cpcm_qams["cpcm_qams_overall_avg_value"]))  # as defined in p.159 of multwfn manual
    cpcm_qams["cpcm_qams_MPI"]=np.mean( np.abs( potentials ) )   # as defined in p.159 of multwfn manual

    #per atom magnitudes
    cpcm_qams["cpcm_qams_atom_overall_surf_area"],cpcm_qams["cpcm_qams_atom_pos_surf_area"],cpcm_qams["cpcm_qams_atom_neg_surf_area"]=[],[],[]
    cpcm_qams["cpcm_qams_atom_min_val"],cpcm_qams["cpcm_qams_atom_max_val"],cpcm_qams["cpcm_qams_atom_overall_val"]=[],[],[]
    cpcm_qams["cpcm_qams_atom_overall_avg"],cpcm_qams["cpcm_qams_atom_pos_avg"],cpcm_qams["cpcm_qams_atom_neg_avg"]=[],[],[]
    cpcm_qams["cpcm_qams_atom_overall_variance"],cpcm_qams["cpcm_qams_atom_pos_variance"],cpcm_qams["cpcm_qams_atom_neg_variance"]=[],[],[]
    cpcm_qams["cpcm_qams_atom_Pi"],cpcm_qams["cpcm_qams_atom_MPI"]=[],[]
    for i in range(0,n_atoms):
        atom_mask=np.array([a==i for a in atom_indxs])
        #print (i)
        #print (atom_mask)
        if np.sum(atom_mask)!=0:
            cpcm_qams["cpcm_qams_atom_overall_surf_area"].append( np.sum(areas[atom_mask]) )
            if np.sum(atom_mask*pos_potential_mask)!=0:
                cpcm_qams["cpcm_qams_atom_pos_surf_area"].append( np.sum(areas[atom_mask*pos_potential_mask]) )
                cpcm_qams["cpcm_qams_atom_pos_avg"].append(np.mean(potentials[atom_mask*pos_potential_mask]))
                cpcm_qams["cpcm_qams_atom_pos_variance"].append(np.var(potentials[atom_mask*pos_potential_mask]))
            else:
                cpcm_qams["cpcm_qams_atom_pos_surf_area"].append(0.0) 
                cpcm_qams["cpcm_qams_atom_pos_avg"].append(0.0) 
                cpcm_qams["cpcm_qams_atom_pos_variance"].append(0.0) 
            if np.sum(atom_mask*neg_potential_mask)!=0:
                cpcm_qams["cpcm_qams_atom_neg_surf_area"].append( np.sum(areas[atom_mask*neg_potential_mask]) )
                cpcm_qams["cpcm_qams_atom_neg_avg"].append(np.mean(potentials[atom_mask*neg_potential_mask]))
                cpcm_qams["cpcm_qams_atom_neg_variance"].append(np.var(potentials[atom_mask*neg_potential_mask]))
            else:
                cpcm_qams["cpcm_qams_atom_neg_surf_area"].append(0.0)
                cpcm_qams["cpcm_qams_atom_neg_avg"].append(0.0)
                cpcm_qams["cpcm_qams_atom_neg_variance"].append(0.0)

            cpcm_qams["cpcm_qams_atom_min_val"].append(np.min(potentials[atom_mask])   )
            cpcm_qams["cpcm_qams_atom_max_val"].append(np.max(potentials[atom_mask])   )
            cpcm_qams["cpcm_qams_atom_overall_avg"].append(np.mean(potentials[atom_mask]))

            #according to Multwfn manulal (p.159), overall variance= pos variance + neg variance; in case any of them is not defined (no positive or negative potential points)
            #the overall variance is also not defined. For consistence with QAMS,the overall variance cannot be defined simply as:
            #cpcm_qams["cpcm_qams_atom_overall_variance"].append(np.var(potentials[atom_mask]))
            if np.sum(cpcm_qams["cpcm_qams_atom_neg_variance"])!=0 and np.sum(cpcm_qams["cpcm_qams_atom_pos_variance"])!=0:
                cpcm_qams["cpcm_qams_atom_overall_variance"].append(cpcm_qams["cpcm_qams_atom_pos_variance"][-1]+cpcm_qams["cpcm_qams_atom_neg_variance"][-1])
            else: 
                cpcm_qams["cpcm_qams_atom_overall_variance"].append(0.0)


            cpcm_qams["cpcm_qams_atom_Pi"].append(  np.mean(np.abs(potentials[atom_mask]-cpcm_qams["cpcm_qams_atom_overall_avg"][-1])))
        else:
            cpcm_qams["cpcm_qams_atom_overall_surf_area"].append(0.0)
            cpcm_qams["cpcm_qams_atom_pos_surf_area"].append(0.0)
            cpcm_qams["cpcm_qams_atom_neg_surf_area"].append(0.0)
            cpcm_qams["cpcm_qams_atom_min_val"].append(0.0)
            cpcm_qams["cpcm_qams_atom_max_val"].append(0.0)
            cpcm_qams["cpcm_qams_atom_overall_avg"].append(0.0)
            cpcm_qams["cpcm_qams_atom_pos_avg"].append(0.0)
            cpcm_qams["cpcm_qams_atom_neg_avg"].append(0.0)
            cpcm_qams["cpcm_qams_atom_overall_variance"].append(0.0)
            cpcm_qams["cpcm_qams_atom_pos_variance"].append(0.0)
            cpcm_qams["cpcm_qams_atom_neg_variance"].append(0.0)
            cpcm_qams["cpcm_qams_atom_Pi"].append(0.0)          
    
    return cpcm_qams

def get_NBO_HB_2nd_order_energies(filename,molecule,acceptor_number=0,proton_number=0):

    second_order_energies=[]
    with open (filename,"r") as f: lines=f.readlines()
    #nbo_2nd_order_section_start_line=lines.index(" Second Order Perturbation Theory Analysis of Fock Matrix in NBO Basis\n") #in gaussian and nbo3
    nbo_2nd_order_section_start_line=lines.index(" SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS\n")
    #nbo_2nd_order_section_end_line=lines.index(" Natural Bond Orbitals (Summary):\n")-2 #in gaussian and nbo3
    nbo_2nd_order_section_end_line=lines.index(" NATURAL BOND ORBITALS (Summary):\n")-2
    nbo_2nd_order_lines=lines[nbo_2nd_order_section_start_line:nbo_2nd_order_section_end_line]
    
    if acceptor_number!=0:
        #nbo_2nd_order_LP_acceptor=[l for l in nbo_2nd_order_LP_acceptor if int(l[18:21])==acceptor_number ]#in gaussian and nbo3
        nbo_2nd_order_LP_acceptor=[l for l in nbo_2nd_order_LP_acceptor if int(l[17:19])==acceptor_number ]
    else:
        #nbo_2nd_order_LP_acceptor=[l for l in nbo_2nd_order_lines if (l[6:8]=="LP" and  l[16].lower() in ["o","f","s","n","cl"])  ]#in gaussian and nbo3
        nbo_2nd_order_LP_acceptor=[l for l in nbo_2nd_order_lines if (l[7:9]=="LP" and  l[15].lower() in ["o","f","s","n","cl"])  ]

    if proton_number!=0:
        #nbo_2nd_order_HB=[l for l in nbo_2nd_order_HB  if (int(l[56:58])==proton_number or int(l[64:66])==proton_number)   ]#in gaussian and nbo3
        nbo_2nd_order_HB=[l for l in nbo_2nd_order_HB  if (int(l[45:47])==proton_number or int(l[51:53])==proton_number)   ]
    else:
        #nbo_2nd_order_HB=[l for l in nbo_2nd_order_LP_acceptor if ((l[53].lower() in ["o","f","s","n","cl"] and l[61].lower() in ["h"]) or (l[61].lower() in ["o","f","s","n","cl"] and l[53].lower() in ["h"])   )]
        nbo_2nd_order_HB=[l for l in nbo_2nd_order_LP_acceptor if ((l[43].lower() in ["o","f","s","n","cl"] and l[49].lower() in ["h"]) or (l[49].lower() in ["o","f","s","n","cl"] and l[43].lower() in ["h"])   )]
        #nbo_2nd_order_HB+=[l for l in nbo_2nd_order_LP_acceptor if ((l[53].lower() in ["h"]) and l[43:46]=="LP*")]#in gaussian and nbo3
        nbo_2nd_order_HB+=[l for l in nbo_2nd_order_LP_acceptor if ((l[43].lower() in ["h"]) and l[35:38]=="LP*")]

    for l in nbo_2nd_order_HB:
        #donnor_atom=int(l[18:22])#in gaussian and nbo3
        donnor_atom=int(l[17:19])
        #if l[43:46]=="BD*": acceptor_atom=int(l[63:68]) #in gaussian and nbo3
        if l[35:38]=="BD*": acceptor_atom=int(l[51:54]) 
        #elif l[43:46]=="LP*": acceptor_atom=int(l[55:58])#in gaussian and nbo3
        elif l[35:38]=="LP*": acceptor_atom=int(l[45:47])
        if molecule.distance_rcov_ratio([donnor_atom,acceptor_atom])>1.2: 
            #second_order_energies.append(float(l[76:83])) #in gaussian and nbo3
            second_order_energies.append(float(l[56:64]))

    #for l in nbo_2nd_order_HB: print (l)

    return np.sum(second_order_energies)
    

  
#method to calculate the HOMO-LUMO gap given a molecule     
def get_HOMO_LUMO_gap(molecule):

    ctime=time.time()

    #print (molecule.properties["orbitals"])
    #print (molecule.QM_output.outfile)
    for i in range(0,len(molecule.properties["orbitals"])):
        gap=float(molecule.properties["orbitals"][i][2])-float(molecule.properties["orbitals"][i+1][2])
        if molecule.properties["orbitals"][i][1]=="2.0000" and molecule.properties["orbitals"][i+1][1]=="0.0000": break
    global time_for_homo_lumo_gap
    time_for_homo_lumo_gap+=time.time()-ctime
    return gap


#method to get the stabilization by one (or more) explicit water molecules (using CREST, at gfn2/gfnff level of theory)
def get_water_stabilization_energy(file):
    with open (file,"r") as f: lines=f.readlines()
    for l in lines:
        if l.find("(averaged energy")>-1: e_with_water=float(l.split("(averaged energy:")[1].split()[0])
        if len(l)>10 and len(l)<40:e_water_at_infinite=float(l.strip())
    return hartrees_to_kal_mol*(e_with_water-e_water_at_infinite)



import repeated_molecules
repeated_molecules=repeated_molecules.repeated_molecules

def get_n_identical_structures(filename):
    filename=filename.split("/")[-1]

    if filename[-4:]==".out":filename=filename.split(".out")[0]

    number=""
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

    if filename in repeated_molecules.keys(): return repeated_molecules[filename]
    elif compn in repeated_molecules.keys(): return repeated_molecules[compn]
    else: return 1



def get_file_names(compn,routes):
    charges_str=["6cation","5cation","4cation","3cation","2cation","cation","neut","an","2an","3an","4an","5an","6an"]
    charges=[6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6]

    deprotonated_strs=charges_str[charges_str.index(compn.split("_")[1].split("->")[0])+1:]
    protonated_strs=charges_str[:charges_str.index(compn.split("_")[1].split("->")[0])+1]

    protonated_charge=charges[charges_str.index(compn.split("_")[1].split("->")[0])]
    deprotonated_charge=charges[charges_str.index(compn.split("_")[1].split("->")[0])+1]

    protonated_molecules_file_names,protonated_HL_molecules_file_names=[],[]
    deprotonated_molecules_file_names,deprotonated_HL_molecules_file_names=[],[]
    for route in routes:
        for f in  [  ff for ff in os.listdir(route) if (("hess" not in ff) and ("chrg" not in ff) and ("m06" not in ff) and ("fake" not in ff) and ("cpcm" not in ff) and "_nbo" not in ff)]:
            for protonated_str in  protonated_strs: 
                if f.startswith( str(compn).split("_")[0]+"-"+protonated_str): 
                    protonated_molecules_file_names.append(route+f)
                    protonated_HL_molecules_file_names.append(HL_route+f.split(".out")[0]+HL_text+".out")     
            for deprotonated_str in  deprotonated_strs:
                if f.startswith( str(compn).split("_")[0]+"-"+deprotonated_str): 
                    deprotonated_molecules_file_names.append(route+f)
                    deprotonated_HL_molecules_file_names.append(HL_route+f.split(".out")[0]+HL_text+".out")


    return protonated_charge,deprotonated_charge,protonated_molecules_file_names,protonated_HL_molecules_file_names,deprotonated_molecules_file_names,deprotonated_HL_molecules_file_names


def load_molecules(file_names):
    molecules=[Molecular_structure.Molecular_structure(f,"last",properties) for f in file_names]
    #change energies according to protonation state
    charges_str=["-6cation","-5cation","-4cation","-3cation","-2cation","-cation","-neut","-an","-2an","-3an","-4an","-5an","-6an"]
    factors=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
    for m,file in zip(molecules,file_names):
        factor=factors[ [c in file for c in charges_str].index(True)    ]
        m.gibbs_free_energy=   (  m.gibbs_free_energy  +  factor*ref_DG   ) * hartrees_to_kal_mol
        m.zero_point_energy=   (  m.zero_point_energy  +  factor*ref_zpE  ) * hartrees_to_kal_mol
        m.electronic_energy=   (  m.electronic_energy  +  factor*ref_SP   ) * hartrees_to_kal_mol
    return molecules




###############################################          ADDITIONAL PROPERTIES READ WHEN LOADING MOLECULES        ##############################################################
#not needed: use cpcm_qams["cpcm_qams_volume"] and cpcm_qams["cpcm_qams_overall_surf_area"] instead
#GEPOL_volume=Molecular_structure.Property(name="GEPOL_volume",text_before="GEPOL Volume                                      ...",text_after="",separators=[],drop_words=[],format="float")
#GEPOL_surface=Molecular_structure.Property(name="GEPOL_surface",text_before="GEPOL Surface-area                                ...",text_after="",separators=[],drop_words=[],format="float")

orbitals=Molecular_structure.Property(name="orbitals",text_before="  NO   OCC          E(Eh)            E(eV)",text_after="********************************",separators=["\n",""],length=2000)
SMD_energy=Molecular_structure.Property(name="SMD_energy",text_before="Free-energy (cav+disp)  :",text_after="Eh",format="float",separators=[""])
#properties=[GEPOL_volume,GEPOL_surface,orbitals,SMD_energy]
properties=[orbitals,SMD_energy]

########################################         D E F I N I T I O N    O F    P A N D A    S E R I E S         ################################################################

#file names:
protonated_file_names,deprotonated_file_names=pd.Series(dtype="string"),pd.Series(dtype="string")
print (protonated_file_names)#borrame

#molecular charges
protonated_charge,deprotonated_charge=pd.Series(dtype="int"),pd.Series(dtype="int")  

#energies:
protonated_zero_point_energies,deprotonated_zero_point_energies=pd.Series(dtype="float"),pd.Series(dtype="float")
protonated_energies,deprotonated_energies=pd.Series(dtype="float"),pd.Series(dtype="float")

#solvation
SMD_energy_protonated,SMD_energy_deprotonated,SMD_energy_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
one_water_stabilization_energy_protonated,one_water_stabilization_energy_deprotonated,one_water_stabilization_energy_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
two_water_stabilization_energy_protonated,two_water_stabilization_energy_deprotonated,two_water_stabilization_energy_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
three_water_stabilization_energy_protonated,three_water_stabilization_energy_deprotonated,three_water_stabilization_energy_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
four_water_stabilization_energy_protonated,four_water_stabilization_energy_deprotonated,four_water_stabilization_energy_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")


# delta energies 
delta_zero_point_energy=pd.Series(dtype="float")
delta_electronic_energy=pd.Series(dtype="float")
delta_energy=pd.Series(dtype="float")
delta_zero_point_energy=pd.Series(dtype="float")


#MOLECULAR PROPERTIES:

#dipole moment modulus 
dipole_moment_protonated,dipole_moment_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
dipole_moment_difference=pd.Series(dtype="float")

electronic_spatial_extent_protonated,electronic_spatial_extent_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
electronic_spatial_extent_difference=pd.Series(dtype="float")



#polarizability
polarizability_protonated,polarizability_deprotonated,polarizability_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
#HOMO_LUMO gaps
HOMO_LUMO_gap_protonated,HOMO_LUMO_gap_deprotonated,HOMO_LUMO_gap_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

# intramolecular HB energy according to 2nd order perturbation energy of the NBO
NBO_HB_energy_protonated,NBO_HB_energy_deprotonated,NBO_HB_energy_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

# fraction of reduced density gradient points with sign(lambda2)*density in range for H-bonds, vdw interactions, and steric repulsion
RDG_HB_fraction_protonated,RDG_vdw_fraction_protonated,RDG_steric_fraction_protonated=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
RDG_HB_fraction_deprotonated,RDG_vdw_fraction_deprotonated,RDG_steric_fraction_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
RDG_HB_fraction_difference,RDG_vdw_fraction_difference,RDG_steric_fraction_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

RDG_promol_HB_fraction_protonated,RDG_promol_vdw_fraction_protonated,RDG_promol_steric_fraction_protonated=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
RDG_promol_HB_fraction_deprotonated,RDG_promol_vdw_fraction_deprotonated,RDG_promol_steric_fraction_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
RDG_promol_HB_fraction_difference,RDG_promol_vdw_fraction_difference,RDG_promol_steric_fraction_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")


#SMD molecular properties (analogue to QAMS)
SMD_volume_protonated,SMD_volume_deprotonated,SMD_volume_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_surface_protonated,SMD_surface_deprotonated,SMD_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_min_potential_protonated,SMD_min_potential_deprotonated,SMD_min_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_max_potential_protonated,SMD_max_potential_deprotonated,SMD_max_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_pos_surface_protonated,SMD_pos_surface_deprotonated,SMD_pos_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_neg_surface_protonated,SMD_neg_surface_deprotonated,SMD_neg_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_avg_potential_protonated,SMD_avg_potential_deprotonated,SMD_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_pos_avg_potential_protonated,SMD_pos_avg_potential_deprotonated,SMD_pos_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_neg_avg_potential_protonated,SMD_neg_avg_potential_deprotonated,SMD_neg_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_var_potential_protonated,SMD_var_potential_deprotonated,SMD_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_pos_var_potential_protonated,SMD_pos_var_potential_deprotonated,SMD_pos_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_neg_var_potential_protonated,SMD_neg_var_potential_deprotonated,SMD_neg_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_PI_protonated,SMD_PI_deprotonated,SMD_PI_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
SMD_MPI_protonated,SMD_MPI_deprotonated,SMD_MPI_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

#QAMS molecular properties
QAMS_volume_protonated,QAMS_volume_deprotonated,QAMS_volume_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_surface_protonated,QAMS_surface_deprotonated,QAMS_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_min_potential_protonated,QAMS_min_potential_deprotonated,QAMS_min_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_max_potential_protonated,QAMS_max_potential_deprotonated,QAMS_max_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_pos_surface_protonated,QAMS_pos_surface_deprotonated,QAMS_pos_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_neg_surface_protonated,QAMS_neg_surface_deprotonated,QAMS_neg_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_avg_potential_protonated,QAMS_avg_potential_deprotonated,QAMS_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_pos_avg_potential_protonated,QAMS_pos_avg_potential_deprotonated,QAMS_pos_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_neg_avg_potential_protonated,QAMS_neg_avg_potential_deprotonated,QAMS_neg_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_var_potential_protonated,QAMS_var_potential_deprotonated,QAMS_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_pos_var_potential_protonated,QAMS_pos_var_potential_deprotonated,QAMS_pos_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_neg_var_potential_protonated,QAMS_neg_var_potential_deprotonated,QAMS_neg_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_PI_protonated,QAMS_PI_deprotonated,QAMS_PI_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
QAMS_MPI_protonated,QAMS_MPI_deprotonated,QAMS_MPI_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")


#LEA molecular properties
LEA_min_potential_protonated,LEA_min_potential_deprotonated,LEA_min_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_max_potential_protonated,LEA_max_potential_deprotonated,LEA_max_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_pos_surface_protonated,LEA_pos_surface_deprotonated,LEA_pos_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_neg_surface_protonated,LEA_neg_surface_deprotonated,LEA_neg_surface_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_avg_potential_protonated,LEA_avg_potential_deprotonated,LEA_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_pos_avg_potential_protonated,LEA_pos_avg_potential_deprotonated,LEA_pos_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_neg_avg_potential_protonated,LEA_neg_avg_potential_deprotonated,LEA_neg_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_var_potential_protonated,LEA_var_potential_deprotonated,LEA_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_pos_var_potential_protonated,LEA_pos_var_potential_deprotonated,LEA_pos_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
LEA_neg_var_potential_protonated,LEA_neg_var_potential_deprotonated,LEA_neg_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

#ALIE molecular properties
ALIE_min_potential_protonated,ALIE_min_potential_deprotonated,ALIE_min_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
ALIE_max_potential_protonated,ALIE_max_potential_deprotonated,ALIE_max_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
ALIE_avg_potential_protonated,ALIE_avg_potential_deprotonated,ALIE_avg_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")
ALIE_var_potential_protonated,ALIE_var_potential_deprotonated,ALIE_var_potential_difference=pd.Series(dtype="float"),pd.Series(dtype="float"),pd.Series(dtype="float")

V_grad_average_protonated,V_grad_average_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_average_protonated,V_grad_tang_average_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_average_protonated,V_grad_norm_average_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_95_protonated,V_grad_95_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_95_protonated,V_grad_tang_95_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_95_protonated,V_grad_norm_95_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_90_protonated,V_grad_90_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_90_protonated,V_grad_tang_90_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_90_protonated,V_grad_norm_90_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_75_protonated,V_grad_75_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_75_protonated,V_grad_tang_75_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_75_protonated,V_grad_norm_75_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_50_protonated,V_grad_50_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_50_protonated,V_grad_tang_50_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_50_protonated,V_grad_norm_50_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_ang_average_protonated,V_grad_ang_average_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_ang_95_protonated,V_grad_ang_95_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_ang_90_protonated,V_grad_ang_90_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_ang_75_protonated,V_grad_ang_75_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_ang_50_protonated,V_grad_ang_50_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_red_average_protonated,V_grad_red_average_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_red_average_protonated,V_grad_tang_red_average_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_red_average_protonated,V_grad_norm_red_average_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_red_95_protonated,V_grad_red_95_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_red_95_protonated,V_grad_tang_red_95_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_red_95_protonated,V_grad_norm_red_95_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_red_90_protonated,V_grad_red_90_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_red_90_protonated,V_grad_tang_red_90_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_red_90_protonated,V_grad_norm_red_90_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_red_75_protonated,V_grad_red_75_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_red_75_protonated,V_grad_tang_red_75_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_red_75_protonated,V_grad_norm_red_75_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_red_50_protonated,V_grad_red_50_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_tang_red_50_protonated,V_grad_tang_red_50_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
V_grad_norm_red_50_protonated,V_grad_norm_red_50_deprotonated=pd.Series(dtype="float"),pd.Series(dtype="float")
#ATOMIC CHARGES:





counter=0
if not any([":" in s for s in sys.argv]): rows=labels.index
else:   
        arg_index=[":" in s for s in sys.argv].index(True) 
        if sys.argv[arg_index].startswith(":"): start=0;end=sys.argv[arg_index].split(":")[1]
        elif sys.argv[arg_index].endswith(":"): end=len(labels.index)+1; start=sys.argv[arg_index].split(":")[0]
        else: start,end=sys.argv[arg_index].split(":")[0],sys.argv[arg_index].split(":")[1]
        rows=labels.index[int(start):int(end)]
        print (start)
        print (end)

########################################         R U N   F O R   E V E R Y   E N T R Y    I N   T H E   D A T A B A S E         ################################################################
for compn in rows:  
    counter+=1 
    if counter<start_index or counter>stop_index: continue 


    protonated_charge[compn],deprotonated_charge[compn],protonated_molecules_file_names,protonated_HL_molecules_file_names,deprotonated_molecules_file_names,deprotonated_HL_molecules_file_names=get_file_names(compn,routes)

    print ("protonated_molecules_file_names"); print (protonated_molecules_file_names)
    print ("deprotonated_HL_molecules_file_names"); print(deprotonated_molecules_file_names)
    ######################################       F O R      B L A N K      E N T R I E S     O R     M I S S I N G     F I L E S      ###########################################################
    if len(protonated_molecules_file_names)==0:
        pass
        


    ##################################       F O R    N O T     B L A N K      E N T R I E S    A N D      E X I S T I N G     F I L E S      ###################################################
    else:
        ctime=time.time()
        #list of file names for each conformer/tautomer for the protonated(deprotonated) for the entry in labels file compn.
        protonated_file_names[compn]=protonated_molecules_file_names
        deprotonated_file_names[compn]=deprotonated_molecules_file_names    
        print (" "*80,end="\r")
        print (str(counter)+"/"+str(len(rows))+"  loading "+str(len(protonated_molecules_file_names))+" protonated molecules and "+str(len(deprotonated_molecules_file_names))+"deprotonated molecules for:"+ str(compn)+"                                            ", end="\r")



        #######################################################              L I S T S       O F      M O L E C U L E S               ###########################################################
        #list of molecule objects of each conformer/tautomer for the protonated(deprotonated) corresponding to the entry in labels file compn.
        #with the level of theory used in the optimization

        protonated_molecules=load_molecules(protonated_molecules_file_names)
        protonated_molecules_HL=load_molecules(protonated_HL_molecules_file_names)
        deprotonated_molecules=load_molecules(deprotonated_molecules_file_names)
        deprotonated_molecules_HL=load_molecules(deprotonated_HL_molecules_file_names)


        #list with the number of equivalent molecules considering the information in "repeated_molecules" dictionary
        protonated_identical_molecules=[get_n_identical_structures(filename) for filename in protonated_molecules_file_names]
        deprotonated_identical_molecules=[get_n_identical_structures(filename) for filename in deprotonated_molecules_file_names]
        #list of molecule objects of each conformer/tautomer for the protonated(deprotonated) corresponding to the entry in labels file compn.
        #with the high level of theory


        #list of dictionaries with the properties stored in multwfn.json files
        protonated_multwfn_props,deprotonated_multwfn_props=[],[]
        protonated_cpcm_props,deprotonated_cpcm_props=[],[]
        for pmn in protonated_HL_molecules_file_names:
            with open(pmn[:-4]+".multwfn.json","r") as f: l=f.read()
            protonated_multwfn_props.append(json.loads(l))
            protonated_cpcm_props.append(get_smd_cavity_qams(pmn[:-4]+".cpcm"))
        for dmn in deprotonated_HL_molecules_file_names:
            with open(dmn[:-4]+".multwfn.json","r") as f: l=f.read()
            deprotonated_multwfn_props.append(json.loads(l))  
            deprotonated_cpcm_props.append(get_smd_cavity_qams(dmn[:-4]+".cpcm")) 



        

        #list of intramolecular HB energies according to 2nd order NBO analysis
        protonated_HB_NBO_energies=[get_NBO_HB_2nd_order_energies(nbo_route+filename[:-4].split("/")[-1]+".nbo.out",pm_HL) for filename,pm_HL in zip(protonated_HL_molecules_file_names,protonated_molecules_HL)]
        deprotonated_HB_NBO_energies=[get_NBO_HB_2nd_order_energies(nbo_route+filename[:-4].split("/")[-1]+".nbo.out",dm_HL) for filename,dm_HL in zip(deprotonated_HL_molecules_file_names,deprotonated_molecules_HL)]

        #list of the gfn2/gfnff stabilization energy when one explicit water molecule is added, according to CREST-QRC 
        protonated_one_water_stabilization_energies=[get_water_stabilization_energy(one_water_route+filename[:-4].split("/")[-1]+one_water_text) for filename in protonated_molecules_file_names]  
        deprotonated_one_water_stabilization_energies=[get_water_stabilization_energy(one_water_route+filename[:-4].split("/")[-1]+one_water_text) for filename in deprotonated_molecules_file_names]

        #list of the gfn2/gfnff stabilization energy when two explicit water molecule is added, according to CREST-QRC 
        #protonated_two_water_stabilization_energies=[get_water_stabilization_energy(two_water_route+filename[:-4].split("/")[-1]+two_water_text) for filename in protonated_molecules_file_names]  
        #deprotonated_two_water_stabilization_energies=[get_water_stabilization_energy(two_water_route+filename[:-4].split("/")[-1]+two_water_text) for filename in deprotonated_molecules_file_names]

        #list of the gfn2/gfnff stabilization energy when three explicit water molecule is added, according to CREST-QRC 
        #protonated_three_water_stabilization_energies=[get_water_stabilization_energy(three_water_route+filename[:-4].split("/")[-1]+three_water_text) for filename in protonated_molecules_file_names]  
        #deprotonated_three_water_stabilization_energies=[get_water_stabilization_energy(three_water_route+filename[:-4].split("/")[-1]+three_water_text) for filename in deprotonated_molecules_file_names]

        #list of the gfn2/gfnff stabilization energy when four explicit water molecule is added, according to CREST-QRC 
        #protonated_four_water_stabilization_energies=[get_water_stabilization_energy(four_water_route+filename[:-4].split("/")[-1]+four_water_text) for filename in protonated_molecules_file_names]  
        #deprotonated_four_water_stabilization_energies=[get_water_stabilization_energy(four_water_route+filename[:-4].split("/")[-1]+four_water_text) for filename in deprotonated_molecules_file_names]

        time_for_loading_molecules+=time.time()-ctime
        ctime=time.time()

        #################################             E Q U I L I B R I U M        D E P E N D E N T      P R O P E R T I E  S              ########################################################
        ##############################################################              E N E R G I E S               ####################################################################################
        # the goal is to find an energy difference for the equilibrium:    A-H   +   H2O   <--->  A(-)  + H3O(+)   such that if there were only one conformer/tautomer of A-H and of A(-), their energy
        # difference would reproduce the populations of A-H and A(-): 
        # Population(A-H)= Sum_over_protonated_conformers(exp(-DeltaGi/RT)) / Z                Z= Sum_over_protonated_conformers(exp(-DeltaGi/RT)) + Sum_over_deprotonated_conformers(exp(-DeltaGi/RT))
        # Population(A(-))= Sum_over_deprotonated_conformers(exp(-DeltaGi/RT)) / Z         
        # Population(A-H)/Population(A(-)) =  Sum_over_protonated_conformers(exp(-DeltaGi/RT)) /  Sum_over_deprotonated_conformers(exp(-DeltaGi/RT))  =   exp(-DeltaG/RT)
        # DeltaG = -RT*ln(  Sum_over_protonated_conformers(exp(-DeltaGi/RT)) /  Sum_over_deprotonated_conformers(exp(-DeltaGi/RT))    )
        protonated_gibbs_free_energies_optz=   np.array([m.gibbs_free_energy for m in protonated_molecules])
        deprotonated_gibbs_free_energies_optz= np.array([m.gibbs_free_energy for m in deprotonated_molecules])
        protonated_zero_point_energies_optz=   np.array([m.zero_point_energy for m in protonated_molecules])
        deprotonated_zero_point_energies_optz= np.array([m.zero_point_energy for m in deprotonated_molecules])
        protonated_sp_energies_optz=           np.array([m.electronic_energy for m in protonated_molecules])
        deprotonated_sp_energies_optz=         np.array([m.electronic_energy for m in deprotonated_molecules])
        protonated_sp_energies=                np.array([m.electronic_energy for m in protonated_molecules_HL])
        deprotonated_sp_energies=              np.array([m.electronic_energy for m in deprotonated_molecules_HL])
        protonated_gibbs_free_energies=   protonated_gibbs_free_energies_optz-protonated_sp_energies_optz+protonated_sp_energies
        deprotonated_gibbs_free_energies= deprotonated_gibbs_free_energies_optz-deprotonated_sp_energies_optz+deprotonated_sp_energies
        protonated_zero_point_energies=   protonated_zero_point_energies_optz-protonated_sp_energies_optz+protonated_sp_energies
        deprotonated_zero_point_energies= deprotonated_zero_point_energies_optz-deprotonated_sp_energies_optz+deprotonated_sp_energies

        #to prevent overflow, use relative values:
        protonated_gibbs_free_energies=protonated_gibbs_free_energies-np.min(deprotonated_gibbs_free_energies)
        deprotonated_gibbs_free_energies=deprotonated_gibbs_free_energies-np.min(deprotonated_gibbs_free_energies)
        protonated_zero_point_energies=protonated_zero_point_energies-np.min(deprotonated_zero_point_energies)
        deprotonated_zero_point_energies=deprotonated_zero_point_energies-np.min(deprotonated_zero_point_energies)
        protonated_sp_energies=protonated_sp_energies-np.min(deprotonated_sp_energies)
        deprotonated_sp_energies=deprotonated_sp_energies-np.min(deprotonated_sp_energies)


        delta_zero_point_energy[compn] = -RT*np.log( ( np.sum( np.exp( -protonated_zero_point_energies/RT )*(protonated_identical_molecules)   )  / np.sum( np.exp( -deprotonated_zero_point_energies/RT )*(deprotonated_identical_molecules)   ) ) )
        delta_electronic_energy[compn] = -RT*np.log( ( np.sum( np.exp( -protonated_sp_energies/RT )*(protonated_identical_molecules)   )  / np.sum( np.exp( -deprotonated_sp_energies/RT )*(deprotonated_identical_molecules)   ) ) )
        delta_energy[compn]            = -RT*np.log( ( np.sum( np.exp( -protonated_gibbs_free_energies/RT )*(protonated_identical_molecules)   )  / np.sum( np.exp( -deprotonated_gibbs_free_energies/RT )*(deprotonated_identical_molecules)   ) ) )
        #delta_zero_point_energy[compn] = -RT*np.log( ( np.sum( np.exp( -protonated_zero_point_energies/RT )   )  / np.sum( np.exp( -deprotonated_sp_energies/RT )   ) ) )
 
        #these populations will be used later:
        #Note that the sp at the high level of theory + deltaG correction at the optimization level of theory is used. Maybe this have to be changed to use other energies
        if   weighting=="gibbs":
            protonated_molecules_populations= np.exp( -protonated_gibbs_free_energies/RT )*(protonated_identical_molecules)
            deprotonated_molecules_populations= np.exp( -deprotonated_gibbs_free_energies/RT )*(deprotonated_identical_molecules)
        elif weighting=="zero":
            protonated_molecules_populations= np.exp( -protonated_zero_point_energies/RT )*(protonated_identical_molecules)
            deprotonated_molecules_populations= np.exp( -deprotonated_zero_point_energies/RT )*(deprotonated_identical_molecules)
        elif weighting=="sp":
            protonated_molecules_populations= np.exp( -protonated_sp_energies/RT )*(protonated_identical_molecules)
            deprotonated_molecules_populations= np.exp( -deprotonated_sp_energies/RT )*(deprotonated_identical_molecules)           
        #delta_energy[compn]            = -RT*np.log(sum_protonated_molecules_populations/sum_deprotonated_molecules_populations)
        #renormalize internally for future calculation of contributions of different properties:
        sum_protonated_molecules_populations=np.sum(protonated_molecules_populations)
        sum_deprotonated_molecules_populations=np.sum(deprotonated_molecules_populations)
        protonated_molecules_populations= protonated_molecules_populations / sum_protonated_molecules_populations 
        deprotonated_molecules_populations= deprotonated_molecules_populations / sum_deprotonated_molecules_populations

    
        #########################################################       M O L E C U L A R        P R O P E R T I E S       ################################################################### 
        #these properties are read from each molecule and averaged according to the Maxwell-Boltzman populations 


        NBO_HB_energy_protonated[compn]   = np.array( protonated_HB_NBO_energies  ).dot(protonated_molecules_populations)
        NBO_HB_energy_deprotonated[compn] = np.array( deprotonated_HB_NBO_energies).dot(deprotonated_molecules_populations)
        NBO_HB_energy_difference[compn]   = NBO_HB_energy_protonated[compn] - NBO_HB_energy_deprotonated[compn]

        one_water_stabilization_energy_protonated[compn]   = np.array( protonated_one_water_stabilization_energies    ).dot(protonated_molecules_populations)
        one_water_stabilization_energy_deprotonated[compn] = np.array( deprotonated_one_water_stabilization_energies  ).dot(deprotonated_molecules_populations)
        one_water_stabilization_energy_difference[compn]   = one_water_stabilization_energy_protonated[compn] - one_water_stabilization_energy_deprotonated[compn]

        #two_water_stabilization_energy_protonated[compn]   = np.array( protonated_two_water_stabilization_energies    ).dot(protonated_molecules_populations)
        #two_water_stabilization_energy_deprotonated[compn] = np.array( deprotonated_two_water_stabilization_energies  ).dot(deprotonated_molecules_populations)
        #two_water_stabilization_energy_difference[compn]   = two_water_stabilization_energy_protonated[compn] - two_water_stabilization_energy_deprotonated[compn]

        #three_water_stabilization_energy_protonated[compn]   = np.array( protonated_three_water_stabilization_energies    ).dot(protonated_molecules_populations)
        #three_water_stabilization_energy_deprotonated[compn] = np.array( deprotonated_three_water_stabilization_energies  ).dot(deprotonated_molecules_populations)
        #three_water_stabilization_energy_difference[compn]   = three_water_stabilization_energy_protonated[compn] - three_water_stabilization_energy_deprotonated[compn]

        #four_water_stabilization_energy_protonated[compn]   = np.array( protonated_four_water_stabilization_energies    ).dot(protonated_molecules_populations)
        #four_water_stabilization_energy_deprotonated[compn] = np.array( deprotonated_four_water_stabilization_energies  ).dot(deprotonated_molecules_populations)
        #four_water_stabilization_energy_difference[compn]   = four_water_stabilization_energy_protonated[compn] - four_water_stabilization_energy_deprotonated[compn]

        RDG_HB_fraction_protonated[compn]        = np.array([float(pmp["fraction_hb_in_rdg_plot"]     ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        RDG_vdw_fraction_protonated[compn]       = np.array([float(pmp["fraction_vdw_in_rdg_plot"]    ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        RDG_steric_fraction_protonated[compn]    = np.array([float(pmp["fraction_steric_in_rdg_plot"] ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)

        RDG_HB_fraction_deprotonated[compn]        = np.array([float(dmp["fraction_hb_in_rdg_plot"]     ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        RDG_vdw_fraction_deprotonated[compn]       = np.array([float(dmp["fraction_vdw_in_rdg_plot"]    ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        RDG_steric_fraction_deprotonated[compn]    = np.array([float(dmp["fraction_steric_in_rdg_plot"] ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)

        RDG_HB_fraction_difference[compn]          = RDG_HB_fraction_protonated[compn] - RDG_HB_fraction_deprotonated[compn]
        RDG_vdw_fraction_difference[compn]         = RDG_vdw_fraction_protonated[compn] - RDG_vdw_fraction_deprotonated[compn]
        RDG_steric_fraction_difference[compn]      = RDG_steric_fraction_protonated[compn] - RDG_steric_fraction_deprotonated[compn]

        RDG_promol_HB_fraction_protonated[compn]        = np.array([float(pmp["fraction_hb_in_promolecular_rdg_plot"]    ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        RDG_promol_vdw_fraction_protonated[compn]       = np.array([float(pmp["fraction_vdw_in_promolecular_rdg_plot"]    ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        RDG_promol_steric_fraction_protonated[compn]    = np.array([float(pmp["fraction_steric_in_promolecular_rdg_plot"]    ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)

        RDG_promol_HB_fraction_deprotonated[compn]      = np.array([float(dmp["fraction_hb_in_promolecular_rdg_plot"]    ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        RDG_promol_vdw_fraction_deprotonated[compn]     = np.array([float(dmp["fraction_vdw_in_promolecular_rdg_plot"]    ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        RDG_promol_steric_fraction_deprotonated[compn]  = np.array([float(dmp["fraction_steric_in_promolecular_rdg_plot"]    ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)

        RDG_promol_HB_fraction_difference[compn]        = RDG_promol_HB_fraction_protonated[compn] - RDG_promol_HB_fraction_deprotonated[compn]
        RDG_promol_vdw_fraction_difference[compn]       = RDG_promol_vdw_fraction_protonated[compn] - RDG_promol_vdw_fraction_deprotonated[compn]
        RDG_promol_steric_fraction_difference[compn]    = RDG_promol_steric_fraction_protonated[compn] - RDG_promol_steric_fraction_deprotonated[compn]

        dipole_moment_protonated[compn]   = np.array([float(pmp["FASA_mod_Molecular_dipole_moment"]    ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        dipole_moment_deprotonated[compn] = np.array([float(dmp["FASA_mod_Molecular_dipole_moment"] ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        #dipole_moment_protonated[compn]   = np.array([float(np.linalg.norm(pm.dipole_moment)) for pm in protonated_molecules_HL]).dot(protonated_molecules_populations)
        #dipole_moment_deprotonated[compn] = np.array([float(np.linalg.norm(dm.dipole_moment)) for dm in deprotonated_molecules_HL]).dot(deprotonated_molecules_populations)
        dipole_moment_difference[compn]      = dipole_moment_protonated[compn] - dipole_moment_deprotonated[compn] 

        electronic_spatial_extent_protonated[compn]   = np.array([float(pmp["FASA_mol_electronic_spatial_extent"]    ) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        electronic_spatial_extent_deprotonated[compn] = np.array([float(dmp["FASA_mol_electronic_spatial_extent"] ) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        electronic_spatial_extent_difference[compn]   = electronic_spatial_extent_protonated[compn] - electronic_spatial_extent_deprotonated[compn]

        polarizability_protonated[compn]  = np.array([float(np.linalg.norm(pm.polarizability)) for pm in protonated_molecules_HL]).dot(protonated_molecules_populations)
        polarizability_deprotonated[compn]= np.array([float(np.linalg.norm(dm.polarizability)) for dm in deprotonated_molecules_HL]).dot(deprotonated_molecules_populations)
        polarizability_difference[compn]     = polarizability_protonated[compn]-polarizability_deprotonated[compn]

        SMD_energy_protonated[compn]      = np.array([float(pm.properties["SMD_energy"]) for pm in protonated_molecules_HL]).dot(protonated_molecules_populations)
        SMD_energy_deprotonated[compn]    = np.array([float(dm.properties["SMD_energy"]) for dm in deprotonated_molecules_HL]).dot(deprotonated_molecules_populations)
        SMD_energy_difference[compn]         = SMD_energy_protonated[compn] - SMD_energy_deprotonated[compn]

        HOMO_LUMO_gap_protonated[compn]   = np.array([get_HOMO_LUMO_gap(pm) for pm in protonated_molecules_HL]).dot(protonated_molecules_populations)
        HOMO_LUMO_gap_deprotonated[compn] = np.array([get_HOMO_LUMO_gap(dm) for dm in deprotonated_molecules_HL]).dot(deprotonated_molecules_populations)
        HOMO_LUMO_gap_difference[compn]      = HOMO_LUMO_gap_protonated[compn]-HOMO_LUMO_gap_deprotonated[compn]

        QAMS_volume_protonated[compn]              = np.array([float(pmp['qams_volume']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_surface_protonated[compn]             = np.array([float(pmp['qams_overall_surf_area']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_min_potential_protonated[compn]       = np.array([float(pmp['qams_min_val']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_max_potential_protonated[compn]       = np.array([float(pmp['qams_max_val']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_pos_surface_protonated[compn]         = np.array([float(pmp['qams_pos_surf_area']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_neg_surface_protonated[compn]         = np.array([float(pmp['qams_neg_surf_area']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_avg_potential_protonated[compn]       = np.array([float(pmp['qams_overall_avg_value']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_pos_avg_potential_protonated[compn]   = np.array([float(pmp['qams_pos_avg_value']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_neg_avg_potential_protonated[compn]   = np.array([float(pmp['qams_neg_avg_value']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_var_potential_protonated[compn]       = np.array([float(pmp['qams_overall_variance']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_pos_var_potential_protonated[compn]   = np.array([float(pmp['qams_pos_variance']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        QAMS_neg_var_potential_protonated[compn]   = np.array([float(pmp['qams_neg_variance']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)        
        QAMS_PI_protonated[compn]                  = np.array([float(pmp['qams_Pi']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations) 
        QAMS_MPI_protonated[compn]                 = np.array([float(pmp['qams_MPI']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations) 

        QAMS_volume_deprotonated[compn]              = np.array([float(dmp['qams_volume']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_surface_deprotonated[compn]             = np.array([float(dmp['qams_overall_surf_area']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_min_potential_deprotonated[compn]       = np.array([float(dmp['qams_min_val']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_max_potential_deprotonated[compn]       = np.array([float(dmp['qams_max_val']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_pos_surface_deprotonated[compn]         = np.array([float(dmp['qams_pos_surf_area']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_neg_surface_deprotonated[compn]         = np.array([float(dmp['qams_neg_surf_area']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_avg_potential_deprotonated[compn]       = np.array([float(dmp['qams_overall_avg_value']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_pos_avg_potential_deprotonated[compn]   = np.array([float(dmp['qams_pos_avg_value']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_neg_avg_potential_deprotonated[compn]   = np.array([float(dmp['qams_neg_avg_value']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_var_potential_deprotonated[compn]       = np.array([float(dmp['qams_overall_variance']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_pos_var_potential_deprotonated[compn]   = np.array([float(dmp['qams_pos_variance']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        QAMS_neg_var_potential_deprotonated[compn]   = np.array([float(dmp['qams_neg_variance']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)        
        QAMS_PI_deprotonated[compn]                  = np.array([float(dmp['qams_Pi']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations) 
        QAMS_MPI_deprotonated[compn]                 = np.array([float(dmp['qams_MPI']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations) 

        QAMS_volume_difference[compn]                = QAMS_volume_protonated[compn]-QAMS_volume_deprotonated[compn]
        QAMS_surface_difference[compn]               = QAMS_surface_protonated[compn]-QAMS_surface_deprotonated[compn]
        QAMS_min_potential_difference[compn]         = QAMS_min_potential_protonated[compn]-QAMS_min_potential_deprotonated[compn]
        QAMS_max_potential_difference[compn]         = QAMS_max_potential_protonated[compn]-QAMS_max_potential_deprotonated[compn]
        QAMS_pos_surface_difference[compn]           = QAMS_pos_surface_protonated[compn]-QAMS_pos_surface_deprotonated[compn]
        QAMS_neg_surface_difference[compn]           = QAMS_neg_surface_protonated[compn]-QAMS_neg_surface_deprotonated[compn]
        QAMS_avg_potential_difference[compn]         = QAMS_avg_potential_protonated[compn]-QAMS_avg_potential_deprotonated[compn]
        QAMS_pos_avg_potential_difference[compn]     = QAMS_pos_avg_potential_protonated[compn]-QAMS_pos_avg_potential_deprotonated[compn]
        QAMS_neg_avg_potential_difference[compn]     = QAMS_neg_avg_potential_protonated[compn]-QAMS_pos_avg_potential_deprotonated[compn]
        QAMS_var_potential_difference[compn]         = QAMS_var_potential_protonated[compn]-QAMS_var_potential_deprotonated[compn]
        QAMS_pos_var_potential_difference[compn]     = QAMS_pos_var_potential_protonated[compn]-QAMS_pos_var_potential_deprotonated[compn]
        QAMS_neg_var_potential_difference[compn]     = QAMS_neg_var_potential_protonated[compn]-QAMS_neg_var_potential_deprotonated[compn]
        QAMS_PI_difference[compn]                    = QAMS_PI_protonated[compn]-QAMS_PI_deprotonated[compn] 
        QAMS_MPI_difference[compn]                   = QAMS_MPI_protonated[compn]-QAMS_MPI_deprotonated[compn]

        SMD_volume_protonated[compn]              = np.array([float(pmp['cpcm_qams_volume']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_surface_protonated[compn]             = np.array([float(pmp['cpcm_qams_overall_surf_area']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_min_potential_protonated[compn]       = np.array([float(pmp['cpcm_qams_min_val']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_max_potential_protonated[compn]       = np.array([float(pmp['cpcm_qams_max_val']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_pos_surface_protonated[compn]         = np.array([float(pmp['cpcm_qams_pos_surf_area']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_neg_surface_protonated[compn]         = np.array([float(pmp['cpcm_qams_neg_surf_area']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_avg_potential_protonated[compn]       = np.array([float(pmp['cpcm_qams_overall_avg_value']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_pos_avg_potential_protonated[compn]   = np.array([float(pmp['cpcm_qams_pos_avg_value']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_neg_avg_potential_protonated[compn]   = np.array([float(pmp['cpcm_qams_neg_avg_value']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_var_potential_protonated[compn]       = np.array([float(pmp['cpcm_qams_overall_variance']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_pos_var_potential_protonated[compn]   = np.array([float(pmp['cpcm_qams_pos_variance']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)
        SMD_neg_var_potential_protonated[compn]   = np.array([float(pmp['cpcm_qams_neg_variance']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations)        
        SMD_PI_protonated[compn]                  = np.array([float(pmp['cpcm_qams_Pi']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations) 
        SMD_MPI_protonated[compn]                 = np.array([float(pmp['cpcm_qams_MPI']) for pmp in protonated_cpcm_props]).dot(protonated_molecules_populations) 

        SMD_volume_deprotonated[compn]              = np.array([float(dmp['cpcm_qams_volume']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_surface_deprotonated[compn]             = np.array([float(dmp['cpcm_qams_overall_surf_area']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_min_potential_deprotonated[compn]       = np.array([float(dmp['cpcm_qams_min_val']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_max_potential_deprotonated[compn]       = np.array([float(dmp['cpcm_qams_max_val']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_pos_surface_deprotonated[compn]         = np.array([float(dmp['cpcm_qams_pos_surf_area']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_neg_surface_deprotonated[compn]         = np.array([float(dmp['cpcm_qams_neg_surf_area']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_avg_potential_deprotonated[compn]       = np.array([float(dmp['cpcm_qams_overall_avg_value']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_pos_avg_potential_deprotonated[compn]   = np.array([float(dmp['cpcm_qams_pos_avg_value']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_neg_avg_potential_deprotonated[compn]   = np.array([float(dmp['cpcm_qams_neg_avg_value']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_var_potential_deprotonated[compn]       = np.array([float(dmp['cpcm_qams_overall_variance']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_pos_var_potential_deprotonated[compn]   = np.array([float(dmp['cpcm_qams_pos_variance']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)
        SMD_neg_var_potential_deprotonated[compn]   = np.array([float(dmp['cpcm_qams_neg_variance']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations)        
        SMD_PI_deprotonated[compn]                  = np.array([float(dmp['cpcm_qams_Pi']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations) 
        SMD_MPI_deprotonated[compn]                 = np.array([float(dmp['cpcm_qams_MPI']) for dmp in deprotonated_cpcm_props]).dot(deprotonated_molecules_populations) 

        SMD_volume_difference[compn]                = SMD_volume_protonated[compn]-SMD_volume_deprotonated[compn]
        SMD_surface_difference[compn]               = SMD_surface_protonated[compn]-SMD_surface_deprotonated[compn]
        SMD_min_potential_difference[compn]         = SMD_min_potential_protonated[compn]-SMD_min_potential_deprotonated[compn]
        SMD_max_potential_difference[compn]         = SMD_max_potential_protonated[compn]-SMD_max_potential_deprotonated[compn]
        SMD_pos_surface_difference[compn]           = SMD_pos_surface_protonated[compn]-SMD_pos_surface_deprotonated[compn]
        SMD_neg_surface_difference[compn]           = SMD_neg_surface_protonated[compn]-SMD_neg_surface_deprotonated[compn]
        SMD_avg_potential_difference[compn]         = SMD_avg_potential_protonated[compn]-SMD_avg_potential_deprotonated[compn]
        SMD_pos_avg_potential_difference[compn]     = SMD_pos_avg_potential_protonated[compn]-SMD_pos_avg_potential_deprotonated[compn]
        SMD_neg_avg_potential_difference[compn]     = SMD_neg_avg_potential_protonated[compn]-SMD_pos_avg_potential_deprotonated[compn]
        SMD_var_potential_difference[compn]         = SMD_var_potential_protonated[compn]-SMD_var_potential_deprotonated[compn]
        SMD_pos_var_potential_difference[compn]     = SMD_pos_var_potential_protonated[compn]-SMD_pos_var_potential_deprotonated[compn]
        SMD_neg_var_potential_difference[compn]     = SMD_neg_var_potential_protonated[compn]-SMD_neg_var_potential_deprotonated[compn]
        SMD_PI_difference[compn]                    = SMD_PI_protonated[compn]-SMD_PI_deprotonated[compn] 
        SMD_MPI_difference[compn]                   = SMD_MPI_protonated[compn]-SMD_MPI_deprotonated[compn]

        LEA_min_potential_protonated[compn]       = np.array([float(pmp['lea_min_val']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_max_potential_protonated[compn]       = np.array([float(pmp['lea_max_val']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_pos_surface_protonated[compn]         = np.array([float(pmp['lea_pos_surf_area']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_neg_surface_protonated[compn]         = np.array([float(pmp['lea_neg_surf_area']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_avg_potential_protonated[compn]       = np.array([float(pmp['lea_overall_avg_value']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_pos_avg_potential_protonated[compn]   = np.array([float(pmp['lea_pos_avg_value']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_neg_avg_potential_protonated[compn]   = np.array([float(pmp['lea_neg_avg_value']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_var_potential_protonated[compn]       = np.array([float(pmp['lea_overall_variance']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_pos_var_potential_protonated[compn]   = np.array([float(pmp['lea_pos_variance']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        LEA_neg_var_potential_protonated[compn]   = np.array([float(pmp['lea_neg_variance']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)        

        LEA_min_potential_deprotonated[compn]       = np.array([float(dmp['lea_min_val']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_max_potential_deprotonated[compn]       = np.array([float(dmp['lea_max_val']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_pos_surface_deprotonated[compn]         = np.array([float(dmp['lea_pos_surf_area']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_neg_surface_deprotonated[compn]         = np.array([float(dmp['lea_neg_surf_area']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_avg_potential_deprotonated[compn]       = np.array([float(dmp['lea_overall_avg_value']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_pos_avg_potential_deprotonated[compn]   = np.array([float(dmp['lea_pos_avg_value']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_neg_avg_potential_deprotonated[compn]   = np.array([float(dmp['lea_neg_avg_value']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_var_potential_deprotonated[compn]       = np.array([float(dmp['lea_overall_variance']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_pos_var_potential_deprotonated[compn]   = np.array([float(dmp['lea_pos_variance']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        LEA_neg_var_potential_deprotonated[compn]   = np.array([float(dmp['lea_neg_variance']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)        

        LEA_min_potential_difference[compn]         = LEA_min_potential_protonated[compn]-LEA_min_potential_deprotonated[compn]
        LEA_max_potential_difference[compn]         = LEA_max_potential_protonated[compn]-LEA_max_potential_deprotonated[compn]
        LEA_pos_surface_difference[compn]           = LEA_pos_surface_protonated[compn]-LEA_pos_surface_deprotonated[compn]
        LEA_neg_surface_difference[compn]           = LEA_neg_surface_protonated[compn]-LEA_neg_surface_deprotonated[compn]
        LEA_avg_potential_difference[compn]         = LEA_avg_potential_protonated[compn]-LEA_avg_potential_deprotonated[compn]
        LEA_pos_avg_potential_difference[compn]     = LEA_pos_avg_potential_protonated[compn]-LEA_pos_avg_potential_deprotonated[compn]
        LEA_neg_avg_potential_difference[compn]     = LEA_neg_avg_potential_protonated[compn]-LEA_pos_avg_potential_deprotonated[compn]
        LEA_var_potential_difference[compn]         = LEA_var_potential_protonated[compn]-LEA_var_potential_deprotonated[compn]
        LEA_pos_var_potential_difference[compn]     = LEA_pos_var_potential_protonated[compn]-LEA_pos_var_potential_deprotonated[compn]
        LEA_neg_var_potential_difference[compn]     = LEA_neg_var_potential_protonated[compn]-LEA_neg_var_potential_deprotonated[compn]

        ALIE_min_potential_protonated[compn]       = np.array([float(pmp['alie_min_val']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        ALIE_max_potential_protonated[compn]       = np.array([float(pmp['alie_max_val']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        ALIE_avg_potential_protonated[compn]       = np.array([float(pmp['alie_overall_avg_value']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        ALIE_var_potential_protonated[compn]       = np.array([float(pmp['alie_overall_variance']) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)

        ALIE_min_potential_deprotonated[compn]       = np.array([float(dmp['alie_min_val']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        ALIE_max_potential_deprotonated[compn]       = np.array([float(dmp['alie_max_val']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        ALIE_avg_potential_deprotonated[compn]       = np.array([float(dmp['alie_overall_avg_value']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        ALIE_var_potential_deprotonated[compn]       = np.array([float(dmp['alie_overall_variance']) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)

        ALIE_min_potential_difference[compn]         = ALIE_min_potential_protonated[compn]-ALIE_min_potential_deprotonated[compn]
        ALIE_max_potential_difference[compn]         = ALIE_max_potential_protonated[compn]-ALIE_max_potential_deprotonated[compn]
        ALIE_avg_potential_difference[compn]         = ALIE_avg_potential_protonated[compn]-ALIE_avg_potential_deprotonated[compn]
        ALIE_var_potential_difference[compn]         = ALIE_var_potential_protonated[compn]-ALIE_var_potential_deprotonated[compn]


        V_grad_average_protonated[compn]            = np.array([float(pmp["v_gradient_mod_average"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations) 
        V_grad_tang_average_protonated[compn]       = np.array([float(pmp["tangential_v_gradients_mod_average"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations) 
        V_grad_norm_average_protonated[compn]       = np.array([float(pmp["normal_v_gradients_mod_average"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_95_protonated[compn]                 = np.array([float(pmp["v_gradient_0.95_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_95_protonated[compn]            = np.array([float(pmp["tangential_v_gradient_0.95_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_95_protonated[compn]            = np.array([float(pmp["normal_v_gradient_0.95_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_90_protonated[compn]                 = np.array([float(pmp["v_gradient_0.9_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_90_protonated[compn]            = np.array([float(pmp["tangential_v_gradient_0.9_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_90_protonated[compn]            = np.array([float(pmp["normal_v_gradient_0.9_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_75_protonated[compn]                 = np.array([float(pmp["v_gradient_0.75_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_75_protonated[compn]            = np.array([float(pmp["tangential_v_gradient_0.75_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_75_protonated[compn]            = np.array([float(pmp["normal_v_gradient_0.75_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_50_protonated[compn]                 = np.array([float(pmp["v_gradient_0.5_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_50_protonated[compn]            = np.array([float(pmp["tangential_v_gradient_0.5_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_50_protonated[compn]            = np.array([float(pmp["normal_v_gradient_0.5_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)

        V_grad_ang_average_protonated[compn]        = np.array([float(pmp["v_gradient_angle_norm_average"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_ang_95_protonated[compn]             = np.array([float(pmp["v_gradient_angle_norm_0.95_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_ang_90_protonated[compn]             = np.array([float(pmp["v_gradient_angle_norm_0.9_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_ang_75_protonated[compn]             = np.array([float(pmp["v_gradient_angle_norm_0.75_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_ang_50_protonated[compn]             = np.array([float(pmp["v_gradient_angle_norm_0.5_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)

        V_grad_red_average_protonated[compn]        = np.array([float(pmp["v_gradient_mod/v_average"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_red_average_protonated[compn]   = np.array([float(pmp["tangential_v_gradient_mod/v_average"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_red_average_protonated[compn]   = np.array([float(pmp["normal_v_gradient_mod/v_average"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)

        V_grad_red_95_protonated[compn]             = np.array([float(pmp["v_gradient_mod/v_0.95_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_red_95_protonated[compn]        = np.array([float(pmp["tangential_v_gradient_mod/v_0.95_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_red_95_protonated[compn]        = np.array([float(pmp["normal_v_gradient_mod/v_0.95_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_red_90_protonated[compn]             = np.array([float(pmp["v_gradient_mod/v_0.9_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_red_90_protonated[compn]        = np.array([float(pmp["tangential_v_gradient_mod/v_0.9_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_red_90_protonated[compn]        = np.array([float(pmp["normal_v_gradient_mod/v_0.9_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_red_75_protonated[compn]             = np.array([float(pmp["v_gradient_mod/v_0.75_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_red_75_protonated[compn]        = np.array([float(pmp["tangential_v_gradient_mod/v_0.75_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_red_75_protonated[compn]        = np.array([float(pmp["normal_v_gradient_mod/v_0.75_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_red_50_protonated[compn]             = np.array([float(pmp["v_gradient_mod/v_0.5_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_tang_red_50_protonated[compn]        = np.array([float(pmp["tangential_v_gradient_mod/v_0.5_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)
        V_grad_norm_red_50_protonated[compn]        = np.array([float(pmp["normal_v_gradient_mod/v_0.5_quantile"]) for pmp in protonated_multwfn_props]).dot(protonated_molecules_populations)


        V_grad_average_deprotonated[compn]           = np.array([float(dmp["v_gradient_mod_average"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations) 
        V_grad_tang_average_deprotonated[compn]      = np.array([float(dmp["tangential_v_gradients_mod_average"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations) 
        V_grad_norm_average_deprotonated[compn]      = np.array([float(dmp["normal_v_gradients_mod_average"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_95_deprotonated[compn]                = np.array([float(dmp["v_gradient_0.95_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_95_deprotonated[compn]           = np.array([float(dmp["tangential_v_gradient_0.95_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_95_deprotonated[compn]           = np.array([float(dmp["normal_v_gradient_0.95_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_90_deprotonated[compn]                = np.array([float(dmp["v_gradient_0.9_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_90_deprotonated[compn]           = np.array([float(dmp["tangential_v_gradient_0.9_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_90_deprotonated[compn]           = np.array([float(dmp["normal_v_gradient_0.9_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_75_deprotonated[compn]                = np.array([float(dmp["v_gradient_0.75_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_75_deprotonated[compn]           = np.array([float(dmp["tangential_v_gradient_0.75_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_75_deprotonated[compn]           = np.array([float(dmp["normal_v_gradient_0.75_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_50_deprotonated[compn]                = np.array([float(dmp["v_gradient_0.5_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_50_deprotonated[compn]           = np.array([float(dmp["tangential_v_gradient_0.5_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_50_deprotonated[compn]           = np.array([float(dmp["normal_v_gradient_0.5_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)

        V_grad_ang_average_deprotonated[compn]       = np.array([float(dmp["v_gradient_angle_norm_average"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_ang_95_deprotonated[compn]            = np.array([float(dmp["v_gradient_angle_norm_0.95_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_ang_90_deprotonated[compn]            = np.array([float(dmp["v_gradient_angle_norm_0.9_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_ang_75_deprotonated[compn]            = np.array([float(dmp["v_gradient_angle_norm_0.75_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_ang_50_deprotonated[compn]            = np.array([float(dmp["v_gradient_angle_norm_0.5_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)

        V_grad_red_average_deprotonated[compn]       = np.array([float(dmp["v_gradient_mod/v_average"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_red_average_deprotonated[compn]  = np.array([float(dmp["tangential_v_gradient_mod/v_average"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_red_average_deprotonated[compn]  = np.array([float(dmp["normal_v_gradient_mod/v_average"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)

        V_grad_red_95_deprotonated[compn]            = np.array([float(dmp["v_gradient_mod/v_0.95_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_red_95_deprotonated[compn]       = np.array([float(dmp["tangential_v_gradient_mod/v_0.95_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_red_95_deprotonated[compn]       = np.array([float(dmp["normal_v_gradient_mod/v_0.95_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_red_90_deprotonated[compn]            = np.array([float(dmp["v_gradient_mod/v_0.9_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_red_90_deprotonated[compn]       = np.array([float(dmp["tangential_v_gradient_mod/v_0.9_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_red_90_deprotonated[compn]       = np.array([float(dmp["normal_v_gradient_mod/v_0.9_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_red_75_deprotonated[compn]            = np.array([float(dmp["v_gradient_mod/v_0.75_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_red_75_deprotonated[compn]       = np.array([float(dmp["tangential_v_gradient_mod/v_0.75_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_red_75_deprotonated[compn]       = np.array([float(dmp["normal_v_gradient_mod/v_0.75_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_red_50_deprotonated[compn]            = np.array([float(dmp["v_gradient_mod/v_0.5_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_tang_red_50_deprotonated[compn]       = np.array([float(dmp["tangential_v_gradient_mod/v_0.5_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)
        V_grad_norm_red_50_deprotonated[compn]       = np.array([float(dmp["normal_v_gradient_mod/v_0.5_quantile"]) for dmp in deprotonated_multwfn_props]).dot(deprotonated_molecules_populations)



        time_for_reading_mol_properties+=time.time()-ctime
        ctime=time.time()




        if counter%10==0 or counter==len(rows): 

            ##########################################################################################################################################################################################
            #######################################          F I L L     C S V     F I L E      W I T H      P A N D A      S E R I E S                ###############################################
            ##########################################################################################################################################################################################

            labels["protonated charge"]=protonated_charge
            labels["deprotonated charge"]=deprotonated_charge
            labels["delta zero point energy"]=delta_zero_point_energy
            labels["delta electronic energy"]=delta_electronic_energy
            labels["delta gibbs free energy"]=delta_energy

            labels["protonated SMD energy"]=SMD_energy_protonated
            labels["deprotonated SMD energy"]=SMD_energy_deprotonated
            labels["SMD energy difference"]=SMD_energy_difference


            labels["protonated intramolecular HB energies acc. to NBO"]=NBO_HB_energy_protonated
            labels["deprotonated intramolecular HB energies acc. to NBO"]=NBO_HB_energy_deprotonated
            labels["delta intramolecular HB energies acc. to NBO"]=NBO_HB_energy_difference

            labels["protonated gfn2/gfnff CREST-QCG 1 explicit water stabilization"]=one_water_stabilization_energy_protonated
            labels["deprotonated gfn2/gfnff CREST-QCG 1 explicit water stabilization"]=one_water_stabilization_energy_deprotonated
            labels["delta gfn2/gfnff CREST-QCG 1 explicit water stabilization"]=one_water_stabilization_energy_difference

            #labels["protonated gfn2/gfnff CREST-QCG 2 exlicit water stabilization"]=two_water_stabilization_energy_protonated
            #labels["deprotonated gfn2/gfnff CREST-QCG 2 explicit water stabilization"]=two_water_stabilization_energy_deprotonated
            #labels["delta gfn2/gfnff CREST-QCG 2 explicit water stabilization"]=two_water_stabilization_energy_difference

            #labels["protonated gfn2/gfnff CREST-QCG 3 exlicit water stabilization"]=three_water_stabilization_energy_protonated
            #labels["deprotonated gfn2/gfnff CREST-QCG 3 explicit water stabilization"]=three_water_stabilization_energy_deprotonated
            #labels["delta gfn2/gfnff CREST-QCG 3 explicit water stabilization"]=three_water_stabilization_energy_difference

            #labels["protonated gfn2/gfnff CREST-QCG 4 exlicit water stabilization"]=four_water_stabilization_energy_protonated
            #labels["deprotonated gfn2/gfnff CREST-QCG 4 explicit water stabilization"]=four_water_stabilization_energy_deprotonated
            #labels["delta gfn2/gfnff CREST-QCG 4 explicit water stabilization"]=four_water_stabilization_energy_difference

            labels["protonated percentage of points in the RDG plot corresponding to HB"]=RDG_HB_fraction_protonated*100
            labels["deprotonated percentage of points in the RDG plot corresponding to HB"]=RDG_HB_fraction_deprotonated*100
            labels["difference percentage of points in the RDG plot corresponding to HB"]=RDG_HB_fraction_difference*100

            labels["protonated percentage of points in the RDG plot corresponding to vdw"]=RDG_vdw_fraction_protonated*100
            labels["deprotonated percentage of points in the RDG plot corresponding to vdw"]=RDG_vdw_fraction_deprotonated*100
            labels["difference percentage of points in the RDG plot corresponding to vdw"]=RDG_vdw_fraction_difference*100

            labels["protonated percentage of points in the RDG plot corresponding to steric"]=RDG_steric_fraction_protonated*100
            labels["deprotonated percentage of points in the RDG plot corresponding to steric"]=RDG_steric_fraction_deprotonated*100
            labels["difference percentage of points in the RDG plot corresponding to steric"]=RDG_steric_fraction_difference*100

            labels["protonated percentage of points in the promolecular RDG plot corresponding to HB"]=RDG_promol_HB_fraction_protonated*100
            labels["deprotonated percentage of points in the promolecular RDG plot corresponding to HB"]=RDG_promol_HB_fraction_deprotonated*100
            labels["difference percentage of points in the promolecular RDG plot corresponding to HB"]=RDG_promol_HB_fraction_difference*100

            labels["protonated percentage of points in the promolecular RDG plot corresponding to vdw"]=RDG_promol_vdw_fraction_protonated*100
            labels["deprotonated percentage of points in the promolecular RDG plot corresponding to vdw"]=RDG_promol_vdw_fraction_deprotonated*100
            labels["difference percentage of points in the promolecular RDG plot corresponding to vdw"]=RDG_promol_vdw_fraction_difference*100

            labels["protonated percentage of points in the promolecular RDG plot corresponding to steric"]=RDG_promol_steric_fraction_protonated*100
            labels["deprotonated percentage of points in the promolecular RDG plot corresponding to steric"]=RDG_promol_steric_fraction_deprotonated*100
            labels["difference percentage of points in the promolecular RDG plot corresponding to steric"]=RDG_promol_steric_fraction_difference*100

            labels["protonated dipole moment module"],labels["deprotonated dipole moment module"]=   dipole_moment_protonated,dipole_moment_deprotonated
            labels["difference dipole moment module"]=dipole_moment_difference
            labels["protonated molecular electronic spatial extent"],labels["deprotonated molecular electronic spatial extent"]= electronic_spatial_extent_protonated,electronic_spatial_extent_deprotonated
            labels["difference molecular electronic spatial extent"]=electronic_spatial_extent_difference
            labels["protonated isotropic polarizability"],labels["deprotonated isotropic polarizability"]= polarizability_protonated,polarizability_deprotonated
            labels["difference isotropic polarizability"]=polarizability_difference

            labels["protonated HOMO-LUMO gap"],labels["deprotonated HOMO-LUMO gap"]=HOMO_LUMO_gap_protonated,HOMO_LUMO_gap_deprotonated
            labels["difference HOMO-LUMO gap"]=HOMO_LUMO_gap_difference

            labels["protonated QAMS volume"],labels["deprotonated QAMS volume"],labels["difference QAMS volume"]=QAMS_volume_protonated,QAMS_volume_deprotonated,QAMS_volume_difference
            labels["protonated QAMS surface"],labels["deprotonated QAMS surface"],labels["difference QAMS surface"]=QAMS_surface_protonated,QAMS_surface_deprotonated,QAMS_surface_difference
            labels["protonated QAMS pos surface"],labels["deprotonated QAMS pos surface"],labels["difference QAMS pos surface"]=QAMS_pos_surface_protonated,QAMS_pos_surface_deprotonated,QAMS_pos_surface_difference
            labels["protonated QAMS neg surface"],labels["deprotonated QAMS neg surface"],labels["difference QAMS neg surface"]=QAMS_neg_surface_protonated,QAMS_neg_surface_deprotonated,QAMS_neg_surface_difference
            labels["protonated QAMS min potential"],labels["deprotonated QAMS min potential"],labels["difference QAMS min potential"]=QAMS_min_potential_protonated,QAMS_min_potential_deprotonated,QAMS_min_potential_difference            
            labels["protonated QAMS max potential"],labels["deprotonated QAMS max potential"],labels["difference QAMS max potential"]=QAMS_max_potential_protonated,QAMS_max_potential_deprotonated,QAMS_max_potential_difference            
            labels["protonated QAMS potential average"],labels["deprotonated QAMS potential average"],labels["difference QAMS potential average"]=QAMS_avg_potential_protonated,QAMS_avg_potential_deprotonated,QAMS_avg_potential_difference
            labels["protonated QAMS pos potential average"],labels["deprotonated QAMS pos potential average"],labels["difference QAMS pos potential average"]=QAMS_pos_avg_potential_protonated,QAMS_pos_avg_potential_deprotonated,QAMS_pos_avg_potential_difference
            labels["protonated QAMS neg potential average"],labels["deprotonated QAMS neg potential average"],labels["difference QAMS neg potential average"]=QAMS_neg_avg_potential_protonated,QAMS_neg_avg_potential_deprotonated,QAMS_neg_avg_potential_difference
            labels["protonated QAMS potential variance"],labels["deprotonated QAMS potential variance"],labels["difference QAMS potential variance"]=QAMS_var_potential_protonated,QAMS_var_potential_deprotonated,QAMS_var_potential_difference
            labels["protonated QAMS pos potential variance"],labels["deprotonated QAMS pos potential variance"],labels["difference QAMS pos potential variance"]=QAMS_pos_var_potential_protonated,QAMS_pos_var_potential_deprotonated,QAMS_pos_var_potential_difference
            labels["protonated QAMS neg potential variance"],labels["deprotonated QAMS neg potential variance"],labels["difference QAMS neg potential variance"]=QAMS_neg_var_potential_protonated,QAMS_neg_var_potential_deprotonated,QAMS_neg_var_potential_difference
            labels["protonated QAMS PI"],labels["deprotonated QAMS PI"],labels["difference QAMS PI"]=QAMS_PI_protonated,QAMS_PI_deprotonated,QAMS_PI_difference
            labels["protonated QAMS MPI"],labels["deprotonated QAMS MPI"],labels["difference QAMS MPI"]=QAMS_MPI_protonated,QAMS_MPI_deprotonated,QAMS_MPI_difference

            labels["protonated SMD volume"],labels["deprotonated SMD volume"],labels["difference SMD volume"]=SMD_volume_protonated,SMD_volume_deprotonated,SMD_volume_difference
            labels["protonated SMD surface"],labels["deprotonated SMD surface"],labels["difference SMD surface"]=SMD_surface_protonated,SMD_surface_deprotonated,SMD_surface_difference
            labels["protonated SMD pos surface"],labels["deprotonated SMD pos surface"],labels["difference SMD pos surface"]=SMD_pos_surface_protonated,SMD_pos_surface_deprotonated,SMD_pos_surface_difference
            labels["protonated SMD neg surface"],labels["deprotonated SMD neg surface"],labels["difference SMD neg surface"]=SMD_neg_surface_protonated,SMD_neg_surface_deprotonated,SMD_neg_surface_difference
            labels["protonated SMD min potential"],labels["deprotonated SMD min potential"],labels["difference SMD min potential"]=SMD_min_potential_protonated,SMD_min_potential_deprotonated,SMD_min_potential_difference            
            labels["protonated SMD max potential"],labels["deprotonated SMD max potential"],labels["difference SMD max potential"]=SMD_max_potential_protonated,SMD_max_potential_deprotonated,SMD_max_potential_difference            
            labels["protonated SMD potential average"],labels["deprotonated SMD potential average"],labels["difference SMD potential average"]=SMD_avg_potential_protonated,SMD_avg_potential_deprotonated,SMD_avg_potential_difference
            labels["protonated SMD pos potential average"],labels["deprotonated SMD pos avg potential"],labels["difference SMD pos potential average"]=SMD_pos_avg_potential_protonated,SMD_pos_avg_potential_deprotonated,SMD_pos_avg_potential_difference
            labels["protonated SMD neg potential average"],labels["deprotonated SMD neg potential average"],labels["difference SMD neg potential average"]=SMD_neg_avg_potential_protonated,SMD_neg_avg_potential_deprotonated,SMD_neg_avg_potential_difference
            labels["protonated SMD potential variance"],labels["deprotonated SMD potential variance"],labels["difference SMD potential variance"]=SMD_var_potential_protonated,SMD_var_potential_deprotonated,SMD_var_potential_difference
            labels["protonated SMD pos potential variance"],labels["deprotonated SMD pos potential variance"],labels["difference SMD pos potential variance"]=SMD_pos_var_potential_protonated,SMD_pos_var_potential_deprotonated,SMD_pos_var_potential_difference
            labels["protonated SMD neg potential variance"],labels["deprotonated SMD neg potential variance"],labels["difference SMD neg potential variance"]=SMD_neg_var_potential_protonated,SMD_neg_var_potential_deprotonated,SMD_neg_var_potential_difference
            labels["protonated SMD PI"],labels["deprotonated SMD PI"],labels["difference SMD PI"]=SMD_PI_protonated,SMD_PI_deprotonated,SMD_PI_difference
            labels["protonated SMD MPI"],labels["deprotonated SMD MPI"],labels["difference SMD MPI"]=SMD_MPI_protonated,SMD_MPI_deprotonated,SMD_MPI_difference

            labels["protonated LEA pos surface"],labels["deprotonated LEA pos surface"],labels["difference LEA pos surface"]=LEA_pos_surface_protonated,LEA_pos_surface_deprotonated,LEA_pos_surface_difference
            labels["protonated LEA neg surface"],labels["deprotonated LEA neg surface"],labels["difference LEA neg surface"]=LEA_neg_surface_protonated,LEA_neg_surface_deprotonated,LEA_neg_surface_difference
            labels["protonated LEA min"],labels["deprotonated LEA min"],labels["difference LEA min"]=LEA_min_potential_protonated,LEA_min_potential_deprotonated,LEA_min_potential_difference            
            labels["protonated LEA max"],labels["deprotonated LEA max"],labels["difference LEA max"]=LEA_max_potential_protonated,LEA_max_potential_deprotonated,LEA_max_potential_difference            
            labels["protonated LEA average"],labels["deprotonated LEA average"],labels["difference LEA average"]=LEA_avg_potential_protonated,LEA_avg_potential_deprotonated,LEA_avg_potential_difference
            labels["protonated LEA pos average"],labels["deprotonated LEA pos average"],labels["difference LEA pos average"]=LEA_pos_avg_potential_protonated,LEA_pos_avg_potential_deprotonated,LEA_pos_avg_potential_difference
            labels["protonated LEA neg average"],labels["deprotonated LEA neg average"],labels["difference LEA neg average"]=LEA_neg_avg_potential_protonated,LEA_neg_avg_potential_deprotonated,LEA_neg_avg_potential_difference
            labels["protonated LEA variance"],labels["deprotonated LEA variance"],labels["difference LEA variance"]=LEA_var_potential_protonated,LEA_var_potential_deprotonated,LEA_var_potential_difference
            labels["protonated LEA pos variance"],labels["deprotonated LEA pos variance"],labels["difference LEA pos variance"]=LEA_pos_var_potential_protonated,LEA_pos_var_potential_deprotonated,LEA_pos_var_potential_difference
            labels["protonated LEA neg variance"],labels["deprotonated LEA neg variance"],labels["difference LEA neg variance"]=LEA_neg_var_potential_protonated,LEA_neg_var_potential_deprotonated,LEA_neg_var_potential_difference

            labels["protonated ALIE min"],labels["deprotonated ALIE min"],labels["difference ALIE min"]=ALIE_min_potential_protonated,ALIE_min_potential_deprotonated,ALIE_min_potential_difference            
            labels["protonated ALIE max"],labels["deprotonated ALIE max"],labels["difference ALIE max"]=ALIE_max_potential_protonated,ALIE_max_potential_deprotonated,ALIE_max_potential_difference            
            labels["protonated ALIE average"],labels["deprotonated ALIE average"],labels["difference ALIE average"]=ALIE_avg_potential_protonated,ALIE_avg_potential_deprotonated,ALIE_avg_potential_difference
            labels["protonated ALIE variance"],labels["deprotonated ALIE variance"],labels["difference ALIE variance"]=ALIE_var_potential_protonated,ALIE_var_potential_deprotonated,ALIE_var_potential_difference

            labels["protonated average V grad"],labels["deprotonated average V grad"]=V_grad_average_protonated,V_grad_average_deprotonated
            labels["difference average V grad"]=V_grad_average_protonated-V_grad_average_deprotonated
            labels["protonated average V grad tangential"],labels["deprotonated average V grad tangential"]=V_grad_tang_average_protonated,V_grad_tang_average_deprotonated
            labels["difference average V grad tangential"]=V_grad_tang_average_protonated-V_grad_tang_average_deprotonated
            labels["protonated average V grad normal"],labels["deprotonated average V grad normal"]=V_grad_norm_average_protonated,V_grad_norm_average_deprotonated
            labels["difference average V grad normal"]=V_grad_norm_average_protonated-V_grad_norm_average_deprotonated
            labels["protonated 0.95quantile V grad"],labels["deprotonated 0.95quantile V grad"]=V_grad_tang_95_protonated,V_grad_tang_95_deprotonated
            labels["difference 0.95quantile V grad"]=V_grad_95_protonated-V_grad_95_deprotonated
            labels["protonated 0.95quantile V grad tangential"],labels["deprotonated 0.95quantile V grad tangential"]=V_grad_tang_95_protonated,V_grad_tang_95_deprotonated
            labels["difference 0.95quantile V grad tangential"]=V_grad_tang_95_protonated-V_grad_tang_95_deprotonated
            labels["protonated 0.95quantile V grad normal"],labels["deprotonated 0.95quantile V grad normal"]=V_grad_norm_95_protonated,V_grad_norm_95_deprotonated  
            labels["difference 0.95quantile V grad normal"]=V_grad_norm_95_protonated-V_grad_norm_95_deprotonated            
            labels["protonated 0.9quantile V grad"],labels["deprotonated 0.9quantile V grad"]=V_grad_90_protonated,V_grad_90_deprotonated
            labels["difference 0.9quantile V grad"]=V_grad_90_protonated-V_grad_90_deprotonated
            labels["protonated 0.9quantile V grad tangential"],labels["deprotonated 0.9quantile V grad tangential"]=V_grad_tang_90_protonated,V_grad_tang_90_deprotonated
            labels["difference 0.9quantile V grad tangential"]=V_grad_tang_90_protonated-V_grad_tang_90_deprotonated
            labels["protonated 0.9quantile V grad normal"],labels["deprotonated 0.9quantile V grad normal"]=V_grad_norm_90_protonated,V_grad_norm_90_deprotonated 
            labels["difference 0.9quantile V grad normal"]=V_grad_norm_90_protonated-V_grad_norm_90_deprotonated 
            labels["protonated 0.75quantile V grad"],labels["deprotonated 0.75quantile V grad"]=V_grad_75_protonated,V_grad_75_deprotonated
            labels["difference 0.75quantile V grad"]=V_grad_75_protonated-V_grad_75_deprotonated
            labels["protonated 0.75quantile V grad tangential"],labels["deprotonated 0.75quantile V grad tangential"]=V_grad_tang_75_protonated,V_grad_tang_75_deprotonated
            labels["difference 0.75quantile V grad tangential"]=V_grad_tang_75_protonated-V_grad_tang_75_deprotonated
            labels["protonated 0.75quantile V grad normal"],labels["deprotonated 0.75quantile V grad normal"]=V_grad_norm_75_protonated,V_grad_norm_75_deprotonated 
            labels["difference 0.75quantile V grad normal"]=V_grad_norm_75_protonated-V_grad_norm_75_deprotonated
            labels["protonated 0.5quantile V grad"],labels["deprotonated 0.5quantile V grad"]=V_grad_50_protonated,V_grad_50_deprotonated
            labels["difference 0.5quantile V grad"]=V_grad_50_protonated-V_grad_50_deprotonated
            labels["protonated 0.5quantile V grad tangential"],labels["deprotonated 0.5quantile V grad tangential"]=V_grad_tang_50_protonated,V_grad_tang_50_deprotonated
            labels["difference 0.5quantile V grad tangential"]=V_grad_tang_50_protonated-V_grad_tang_50_deprotonated
            labels["protonated 0.5quantile V grad normal"],labels["deprotonated 0.5quantile V grad normal"]=V_grad_norm_50_protonated,V_grad_norm_50_deprotonated  
            labels["difference 0.5quantile V grad normal"]=V_grad_norm_50_protonated-V_grad_norm_50_deprotonated 

            labels["protonated angle average of V grad with normal"],labels["deprotonated angle average of V grad with normal"]=V_grad_ang_average_protonated,V_grad_ang_average_deprotonated
            labels["difference angle average of V grad with normal"]=V_grad_ang_average_protonated-V_grad_ang_average_deprotonated
            labels["protonated angle 0.95quantile of V grad with normal"],labels["deprotonated angle 0.95quantile of V grad with normal"]=V_grad_ang_95_protonated,V_grad_ang_95_deprotonated
            labels["difference angle 0.95quantile of V grad with normal"]=V_grad_ang_95_protonated-V_grad_ang_95_deprotonated
            labels["protonated angle 0.9quantile of V grad with normal"],labels["deprotonated angle 0.9quantile of V grad with normal"]=V_grad_ang_90_protonated,V_grad_ang_90_deprotonated
            labels["difference angle 0.9quantile of V grad with normal"]=V_grad_ang_90_protonated-V_grad_ang_90_deprotonated            
            labels["protonated angle 0.75quantile of V grad with normal"],labels["deprotonated angle 0.75quantile of V grad with normal"]=V_grad_ang_75_protonated,V_grad_ang_75_deprotonated
            labels["difference angle 0.75quantile of V grad with normal"]=V_grad_ang_75_protonated-V_grad_ang_75_deprotonated
            labels["protonated angle 0.5quantile of V grad with normal"],labels["deprotonated angle 0.5quantile of V grad with normal"]=V_grad_ang_50_protonated,V_grad_ang_50_deprotonated
            labels["difference angle 0.5quantile of V grad with normal"]=V_grad_ang_50_protonated-V_grad_ang_50_deprotonated

            labels["protonated average V grad / V ratio"],labels["deprotonated average V grad / V ratio"]=V_grad_red_average_protonated,V_grad_red_average_deprotonated
            labels["difference average V grad / V ratio"]=V_grad_red_average_protonated-V_grad_red_average_deprotonated
            labels["protonated average V grad tangential / V ratio"],labels["deprotonated average V grad tangential / V ratio"]=V_grad_tang_red_average_protonated,V_grad_tang_red_average_deprotonated
            labels["difference average V grad tangential / V ratio"]=V_grad_tang_red_average_protonated-V_grad_tang_red_average_deprotonated
            labels["protonated average V grad normal / V ratio"],labels["deprotonated average V grad normal / V ratio"]=V_grad_norm_red_average_protonated,V_grad_norm_red_average_deprotonated
            labels["difference average V grad normal / V ratio"]=V_grad_norm_red_average_protonated-V_grad_norm_red_average_deprotonated
            labels["protonated 0.95quantile V grad / V ratio"],labels["deprotonated 0.95quantile V grad / V ratio"]=V_grad_red_95_protonated,V_grad_red_95_deprotonated
            labels["difference 0.95quantile V grad / V ratio"]=V_grad_red_95_protonated-V_grad_red_95_deprotonated
            labels["protonated 0.95quantile V grad tangential / V ratio"],labels["deprotonated 0.95quantile V grad tangential / V ratio"]=V_grad_tang_red_95_protonated,V_grad_tang_red_95_deprotonated
            labels["difference 0.95quantile V grad tangential / V ratio"]=V_grad_tang_red_95_protonated-V_grad_tang_red_95_deprotonated
            labels["protonated 0.95quantile V grad normal / V ratio"],labels["deprotonated 0.95quantile V grad normal / V ratio"]=V_grad_norm_red_95_protonated,V_grad_norm_red_95_deprotonated
            labels["difference 0.95quantile V grad normal / V ratio"]=V_grad_norm_red_95_protonated-V_grad_norm_red_95_deprotonated
            labels["protonated 0.9quantile V grad / V ratio"],labels["deprotonated 0.9quantile V grad / V ratio"]=V_grad_red_90_protonated,V_grad_red_90_deprotonated
            labels["difference 0.9quantile V grad / V ratio"]=V_grad_red_90_protonated-V_grad_red_90_deprotonated
            labels["protonated 0.9quantile V grad tangential / V ratio"],labels["deprotonated 0.9quantile V grad tangential / V ratio"]=V_grad_tang_red_90_protonated,V_grad_tang_red_90_deprotonated
            labels["difference 0.9quantile V grad tangential / V ratio"]=V_grad_tang_red_90_protonated-V_grad_tang_red_90_deprotonated
            labels["protonated 0.9quantile V grad normal / V ratio"],labels["deprotonated 0.9quantile V grad normal / V ratio"]=V_grad_norm_red_90_protonated,V_grad_norm_red_90_deprotonated
            labels["difference 0.9quantile V grad normal / V ratio"]=V_grad_tang_red_90_protonated-V_grad_tang_red_90_deprotonated
            labels["protonated 0.75quantile V grad / V ratio"],labels["deprotonated 0.75quantile V grad / V ratio"]=V_grad_red_75_protonated,V_grad_red_75_deprotonated
            labels["difference 0.75quantile V grad / V ratio"]=V_grad_red_75_protonated-V_grad_red_75_deprotonated
            labels["protonated 0.75quantile V grad tangential / V ratio"],labels["deprotonated 0.75quantile V grad tangential / V ratio"]=V_grad_tang_red_75_protonated,V_grad_tang_red_75_deprotonated
            labels["difference 0.75quantile V grad tangential / V ratio"]=V_grad_tang_red_75_protonated-V_grad_tang_red_75_deprotonated
            labels["protonated 0.75quantile V grad normal / V ratio"],labels["deprotonated 0.75quantile V grad normal / V ratio"]=V_grad_norm_red_75_protonated,V_grad_norm_red_75_deprotonated
            labels["difference 0.75quantile V grad normal / V ratio"]=V_grad_norm_red_75_protonated-V_grad_norm_red_75_deprotonated
            labels["protonated 0.5quantile V grad / V ratio"],labels["deprotonated 0.5quantile V grad / V ratio"]=V_grad_red_50_protonated,V_grad_red_50_deprotonated
            labels["difference 0.5quantile V grad / V ratio"]=V_grad_red_50_protonated-V_grad_red_50_deprotonated
            labels["protonated 0.5quantile V grad tangential / V ratio"],labels["deprotonated 0.5quantile V grad tangential / V ratio"]=V_grad_tang_red_50_protonated,V_grad_tang_red_50_deprotonated
            labels["difference 0.5quantile V grad tangential / V ratio"]=V_grad_tang_red_50_protonated-V_grad_tang_red_50_deprotonated
            labels["protonated 0.5quantile V grad normal / V ratio"],labels["deprotonated 0.5quantile V grad normal / V ratio"]=V_grad_norm_red_50_protonated,V_grad_norm_red_50_deprotonated
            labels["difference 0.5quantile V grad normal / V ratio"]=V_grad_norm_red_50_protonated-V_grad_norm_red_50_deprotonated


            print ("writing to file: "+labels_route+values_extracted_file+"                                     ")
            labels.to_csv(labels_route+values_extracted_file)




#print ("done reading files")
#print (protonated_het_charge)


print ("TIMES:")
time_total=time.time()-start_time
print ("TOTAL:                   "+ str(time_total))
print ("FOR LOADING MOLECULES:   "+ str(time_for_loading_molecules)+" ("+str(100*time_for_loading_molecules/time_total )  +"%)"  )
print ("FOR HOMO/LUMO GAP:       "+ str(time_for_homo_lumo_gap)+" ("+str(100*time_for_homo_lumo_gap/time_total )  +"%)"  )
print ("FOR READING MOL PROPS:   "+ str(time_for_reading_mol_properties)+" ("+str(100*time_for_reading_mol_properties/time_total )  +"%)"  )
print ("FOR FILLING PD SERIES:   "+ str(time_for_filling_pd_series)+" ("+str(100*time_for_filling_pd_series/time_total )  +"%)"  )


sys.exit()



