#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
import time
#import Molecular_structure
import numpy as np
import json
import multiprocessing as mp


import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
from routes import orca5_mkl_exe,orca6_mkl_exe,multiwfn_exe,multiwfn_home

def create_molden_input(filename,orca_mkl_exe="/home/orca5/orca_2mkl"):
    print("in create molden input")
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    command=orca_mkl_exe+" "+filename+" -molden "
    print (command)#borrame
    os.system(command)

def correct_molden_input(filename):
    print("in correct molden input")
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    if filename[-13:]==".molden.input": filename=filename.split(".molden.input")[0]
    with open(filename+".molden.input","r") as f: lines=f.readlines()
    new_lines=[]
    
    atomic_charges={}
    print("pseudo section????????????")
    if "[Pseudo]\n" in lines:
        
        pseudo_section=[lines.index("[Pseudo]\n"),lines.index("[GTO]\n")]      
        for i in range(pseudo_section[0],pseudo_section[1]): 
            if len(lines[i].split())==3: atomic_charges[lines[i].split()[0]]=lines[i].split()[2]
        print (atomic_charges)

    #print(atomic_charges!=None)#borrame
    atoms_section=[lines.index("[Atoms] AU\n"),lines.index("[GTO]\n")]
    for i in range(atoms_section[0],atoms_section[1]):
        if lines[i][0]=="I": 
            lines[i]="".join(lines[i][0:8])+"25"+"".join(lines[i][10:])
        if atomic_charges!={}:
            if lines[i].split()[0] in atomic_charges.keys():
                lines[i]="".join(lines[i][0:8])+str(atomic_charges[lines[i].split()[0]])+"".join(lines[i][10:])

    with open(filename+".molden.input","w") as f: f.write("".join(lines))

def read_number_of_atoms(molden_input_file):
    with open (molden_input_file,"r") as f: lines=f.readlines()
    for i in range(0,len(lines)):
        if lines[i].startswith("[Atoms] AU"):
            n_atoms=0
            while "[GTO]" not in lines[i+n_atoms]: n_atoms+=1
            break
    return n_atoms

def generate_multiwfn_job_files(n_atoms=1):
    with open("get_xyz.multiwfn","w") as f:        f.write("100\n2\n2\n\n\nq\n")    # to: file.molden.xyz
    with open("hirshfeld-chrg.multiwfn","w") as f: f.write("7\n1\n1\ny\nq\n")       # to: file.molden.chg
    with open("voronoy-chrg.multiwfn","w") as f:   f.write("7\n2\n1\ny\nq\n")       # to: file.molden.chg
    with open("mulliken-chrg.multiwfn","w") as f:  f.write("7\n5\n1\ny\nq\n")       # to: file.molden.chg
    with open("lowdin-chrg.multiwfn","w") as f:    f.write("7\n6\n\ny\nq\n")        # to: file.molden.chg
    with open("becke-chrg.multiwfn","w") as f:     f.write("7\n10\n0\ny\nq\n")      # to: file.molden.chg
    with open("ADCH-chrg.multiwfn","w") as f:      f.write("7\n11\n1\ny\nq\n")      # to: file.molden.chg
    with open("CHELPG-chrg.multiwfn","w") as f:    f.write("7\n12\n1\ny\nq\n")      # to: file.molden.chg 
    with open("MK-chrg.multiwfn","w") as f:        f.write("7\n13\n1\ny\nq\n")      # to: file.molden.chg
    with open("CM5-chrg.multiwfn","w") as f:       f.write("7\n16\n1\ny\nq\n")      # to: file.molden.chg
    with open("12CM5-chrg.multiwfn","w") as f:     f.write("7\n-16\n1\ny\nq\n")     # to: file.molden.chg
    with open("RESP-chrg.multiwfn","w") as f:      f.write("7\n18\n1\ny\nq\n")      # to: file.molden.chg
    with open("PEOE-chrg.multiwfn","w") as f:      f.write("7\n19\ny\q")            # to: file.molden.chg
    with open("mayer-BO.multiwfn","w") as f:       f.write("9\n1\ny")               # to: bndmat.txt
    with open("wiberg-BO.multiwfn","w") as f:      f.write("9\n3\ny")               # to: bndmat.txt
    with open("mulliken-BO.multiwfn","w") as f:    f.write("9\n4\ny")               # to: bndmat.txt
    with open("fuzzy-BO.multiwfn","w") as f:       f.write("9\n7\ny")               # to: bndmat.txt
    with open("laplacian-BO.multiwfn","w") as f:   f.write("9\n8\ny")               # to: bndmat.txt
    with open("IBSI-BO.multiwfn","w") as f:        f.write("9\n10\n2\n5\n8.0\n1\n1\nq")   # too slow: "9\n10\n5\n8.0\n1\n4\nq"      
                                                                                    # to: stdout, between: 
                                                                                    # 'Note: "Dist" is distance between the two atoms in Angstrom, Int(dg_pair) is the integral in the numerator of the IBSI formule (atomic pair delta-g index)'
                                                                                    # and:  '---------- Intrinsic bond strength index (IBSI) ----------'
    with open("expensive-IBSI-BO.multiwfn","w") as f:  f.write("9\n10\n5\n8.0\n1\n2\nq")   # too slow: "9\n10\n5\n8.0\n1\n4\nq"      
                                                                                    # to: stdout, between: 
                                                                                    # 'Note: "Dist" is distance between the two atoms in Angstrom, Int(dg_pair) is the integral in the numerator of the IBSI formule (atomic pair delta-g index)'
                                                                                    # and:  '---------- Intrinsic bond strength index (IBSI) ----------'

    with open("QAMS.multiwfn","w") as f:           f.write("12\n0\n11\n\q\n")       # to: stdouw, between:
                                                                                    #'       ================= Summary of surface analysis ================='
                                                                                    # and: ' Surface analysis finished!',
                                                                                    #and between ' Note: The atoms having zero surface area (i.e. buried) are not shown below'
                                                                                    #and ' If outputting the surface facets to locsurf.pdb in current folder? By which you can visualize local surface via third-part visualization program such as VMD (y/n)'
    with open("FASA-atoms.multiwfn","w") as f:     f.write("15\n2\n2\nq\n")         # to: multipole.txt and atom_moment.txt  
    with open("AIM.multiwfn","w") as f:            f.write("17\n1\n1\n2\n8\ny\nq\n")# "17\n1\n1\n3\n8\ny\nq\n" much slower to: multipole.txt and atom_moment.txt  (the charges and dipole, quadrupole...)
    with open("rdg_plot.multiwfn","w") as f:        f.write("20\n1\n2\n2\nq\n")    # to: output.txt
    with open("rdg_promol_plot.multiwfn","w") as f:        f.write("20\n2\n2\n2\nq\n")    # to: file.molden.xyz
    with open("ALIE.multiwfn","w") as f: f.write("12\n2\n2\n0\n11\nn\q")
    with open("LEA.multiwfn","w") as f: f.write("12\n2\n4\n0\n11\nn\q")


    t= "1\n"
    for i in range(0,n_atoms):t+="a"+str(i)+"\n"
    t+="q"
    with open("get_v_at_nucleus.multiwfn","w") as f:f.write(t)


    os.environ["Multiwfnpath"]="/home/multiwfn_3.8"

#print xyz file (needed for some methods)
def get_xyz(filename,multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    os.environ["Multiwfnpath"]="/home/multiwfn_3.8"
    command=multiwfn_exe+" "+filename+".molden.input < get_xyz.multiwfn > null  "
    os.system(command)

#charges
def multwfn_chrg(filename,method,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):  #method: hirshfeld,voronoy,mulliken,lowdin,becke,ADCH,CHELPG,MK,CM5,12CM5,RESP,PEOE
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input < "+method+"-chrg.multiwfn > null  -nt "+str(n_proc)
    print(command)
    os.system(command)
    with open(filename+".molden.chg","r") as f: charges=[float(line.split()[4]) for line in f.readlines()]
    return charges

#bond orders
def multwfn_BO(filename,method,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"): #method: mayer,wiberg,mulliken,fuzzy,laplacian
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input < "+method+"-BO.multiwfn > null  -nt "+str(n_proc)
    print (method)
    print (command)
    os.system(command)
    with open("bndmat.txt","r") as f: lines=f.readlines()[1:]
    if lines[0].startswith(" ****"): lines=lines[1:]
    elif lines[1].startswith(" ****"): lines=lines[2:]
    elif lines[2].startswith(" ****"): lines=lines[3:]

    if lines[-1].strip()!="":n_atoms=int(lines[-1][2:8])
    elif lines[-2].strip()!="":n_atoms=int(lines[-2][2:8])
    elif lines[-3].strip()!="":n_atoms=int(lines[-3][2:8])
    BO=[ [0.0 for _ in range(0,n_atoms)] for _ in range(0,n_atoms)  ]
    for i in range(0,len(lines),n_atoms+1):
        column_indexes=[int(w)  for w in lines[i].split()]
        for line in lines[i+1:i+n_atoms+1]:
            row_index=int(line.split()[0])
            for w,column_index in zip(line.split()[1:],column_indexes)   :
                BO[row_index-1][column_index-1]=float(w)
    return BO

def multwfn_IBSI_BO(filename,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    xyz_file=filename+".molden.xyz"
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input < IBSI-BO.multiwfn > IBSI.txt  -nt "+str(n_proc)
    os.system(command)

    with open(xyz_file,"r") as f: n_atoms=int( f.readlines()[0]   )
    with open("IBSI.txt","r") as f: lines=f.readlines()
    start=lines.index(' Note: "Dist" is distance between the two atoms in Angstrom, Int(dg_pair) is the integral in the numerator of the IBSI formule (atomic pair delta-g index)\n')
    end=len(lines) -1 -lines[::-1].index("            ---------- Intrinsic bond strength index (IBSI) ----------\n")
    useful_lines=lines[start+2:end-1]

    BO=[ [0.0 for _ in range(0,n_atoms)] for _ in range(0,n_atoms)  ]   
    for line in useful_lines:
        w=line.split()
        row_index,column_index=int(line[3:5]),int(line[12:14]) 
        BO[row_index-1][column_index-1]=float(line.split("IBSI:")[-1])
   
    #ensure that matrix is symmetric 
    for i in range(0,n_atoms):
        for j in range(i+1,n_atoms):
            BO[j][i]=BO[i][j]

    return BO

def multwfn_expensive_IBSI_BO(filename,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    xyz_file=filename+".molden.xyz"
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input < expensive_IBSI-BO.multiwfn > IBSI.txt  -nt "+str(n_proc)
    os.system(command)

    with open(xyz_file,"r") as f: n_atoms=int( f.readlines()[0]   )
    with open("IBSI.txt","r") as f: lines=f.readlines()
    start=lines.index(' Note: "Dist" is distance between the two atoms in Angstrom, Int(dg_pair) is the integral in the numerator of the IBSI formule (atomic pair delta-g index)\n')
    end=len(lines) -1 -lines[::-1].index("            ---------- Intrinsic bond strength index (IBSI) ----------\n")
    useful_lines=lines[start+2:end-1]

    BO=[ [0.0 for _ in range(0,n_atoms)] for _ in range(0,n_atoms)  ]   
    for line in useful_lines:
        w=line.split()
        row_index,column_index=int(line[3:5]),int(line[12:14]) 
        BO[row_index-1][column_index-1]=float(line.split("IBSI:")[-1])
   
    #ensure that matrix is symmetric 
    for i in range(0,n_atoms):
        for j in range(i+1,n_atoms):
            BO[j][i]=BO[i][j]

    return BO

#molecular properties:
def multwfn_QAMS(filename,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    xyz_file=filename+".molden.xyz"
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input <QAMS.multiwfn> QAMS.txt  -nt "+str(n_proc)
    os.system(command)
    with open("QAMS.txt","r") as f: lines=f.readlines()
    start=lines.index('       ================= Summary of surface analysis =================\n')
    end=lines.index(" Surface analysis finished!\n")  
    useful_lines=lines[start+1:end]
    QAMS={}

    QAMS["qams_volume"]=float(useful_lines[1].split()[4])
    QAMS["qams_min_val"],QAMS["qams_max_val"]=float(useful_lines[3].split()[2]),float(useful_lines[3].split()[6])
    QAMS["qams_overall_surf_area"]=float(useful_lines[4].split()[6])
    QAMS["qams_pos_surf_area"]=float(useful_lines[5].split()[6])
    QAMS["qams_neg_surf_area"]=float(useful_lines[6].split()[6])
    QAMS["qams_overall_avg_value"]=float(useful_lines[7].split()[6])
    v=useful_lines[8].split()[6]
    if v=="NaN": QAMS["qams_pos_avg_value"]=0.0
    else:  QAMS["qams_pos_avg_value"]=float(v)
    v=useful_lines[9].split()[6]
    if v=="NaN": QAMS["qams_neg_avg_value"]=0.0
    else: QAMS["qams_neg_avg_value"]=float(v)
    QAMS["qams_overall_variance"]=float(useful_lines[10].split()[6])
    QAMS["qams_pos_variance"]=float(useful_lines[11].split()[5])
    QAMS["qams_neg_variance"]=float(useful_lines[12].split()[5])
    QAMS["qams_Pi"]=float(useful_lines[15].split()[7])
    QAMS["qams_MPI"]=float(useful_lines[16].split()[7])

    start=lines.index(' Note: The atoms having zero surface area (i.e. buried) are not shown below\n')
    end=lines.index(' If outputting the surface facets to locsurf.pdb in current folder? By which you can visualize local surface via third-part visualization program such as VMD (y/n)\n')    
    useful_lines=lines[start+1:end]
    with open(xyz_file,"r") as f: n_atoms=int( f.readlines()[0]   )  

    qams_atom_overall_surf_area=[0.0 for _ in range(0,n_atoms)]
    qams_atom_pos_surf_area=[0.0 for _ in range(0,n_atoms)]
    qams_atom_neg_surf_area=[0.0 for _ in range(0,n_atoms)]
    qams_atom_min_val=[0.0 for _ in range(0,n_atoms)]
    qams_atom_max_val=[0.0 for _ in range(0,n_atoms)]
    i=3
    while useful_lines[i].strip()!="":
        w=useful_lines[i].split()
        atom=int(w[0])-1
        qams_atom_overall_surf_area[atom]=float(w[1])
        qams_atom_pos_surf_area[atom]=float(w[2])
        qams_atom_neg_surf_area[atom]=float(w[3])
        qams_atom_min_val[atom]=float(w[4])
        qams_atom_max_val[atom]=float(w[5])
        i+=1
    QAMS["qams_atom_overall_surf_area"]=qams_atom_overall_surf_area
    QAMS["qams_atom_pos_surf_area"]=qams_atom_pos_surf_area
    QAMS["qams_atom_neg_surf_area"]=qams_atom_neg_surf_area
    QAMS["qams_atom_min_val"]=qams_atom_min_val
    QAMS["qams_atom_max_val"]=qams_atom_max_val


    qams_atom_overall_avg=[0.0 for _ in range(0,n_atoms)]
    qams_atom_pos_avg=[0.0 for _ in range(0,n_atoms)]
    qams_atom_neg_avg=[0.0 for _ in range(0,n_atoms)]
    qams_atom_overall_variance=[0.0 for _ in range(0,n_atoms)]
    qams_atom_pos_variance=[0.0 for _ in range(0,n_atoms)]
    qams_atom_neg_variance=[0.0 for _ in range(0,n_atoms)]
    i+=3
    while useful_lines[i].strip()!="":
        w=useful_lines[i].split()
        atom=int(w[0])-1  
        qams_atom_overall_avg[atom]=float(w[1])
        if w[2]=="NaN": qams_atom_pos_avg[atom]=0.0
        else:           qams_atom_pos_avg[atom]=float(w[2])
        if w[3]=="NaN": qams_atom_neg_avg[atom]=0.0
        else:           qams_atom_neg_avg[atom]=float(w[3])
        if w[4]=="NaN": qams_atom_overall_variance[atom]=0.0
        else:           qams_atom_overall_variance[atom]=float(w[4])
        if w[5]=="NaN": qams_atom_pos_variance[atom]=0.0
        else:           qams_atom_pos_variance[atom]=float(w[5])
        if w[6]=="NaN": qams_atom_neg_variance[atom]=0.0
        else:           qams_atom_neg_variance[atom]=float(w[6])
        i+=1 
    QAMS["qams_atom_overall_avg"]=qams_atom_overall_avg
    QAMS["qams_atom_pos_avg"]=qams_atom_pos_avg
    QAMS["qams_atom_neg_avg"]=qams_atom_neg_avg
    QAMS["qams_atom_overall_variance"]=qams_atom_overall_variance
    QAMS["qams_atom_pos_variance"]=qams_atom_pos_variance
    QAMS["qams_atom_neg_variance"]=qams_atom_neg_variance

    qams_atom_Pi=[0.0 for _ in range(0,n_atoms)]
    i+=3
    while i<len(useful_lines) and useful_lines[i].strip()!="":
        w=useful_lines[i].split()
        atom=int(w[0])-1
        qams_atom_Pi[atom]=float(w[1])
        i+=i
    QAMS["qams_atom_Pi"]  = qams_atom_Pi

    return QAMS

def multwfn_FASA(filename,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
        if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
        os.environ["Multiwfnpath"]=multiwfn_home
        command=multiwfn_exe+" "+filename+".molden.input <FASA-atoms.multiwfn> null  -nt "+str(n_proc)
        os.system(command)
        with open("multipole.txt","r") as f: lines=f.readlines()
        FASA={}
        xyz_file=filename+".molden.xyz"
        with open(xyz_file,"r") as f: n_atoms=int( f.readlines()[0]   )
        FASA["FASA_atomic_dipole_moments"]=[_ for _ in range(0,n_atoms) ]
        FASA["FASA_mod_atomic_dipole_moments"]=[_ for _ in range(0,n_atoms)]
        FASA["FASA_atomic_dipole_moments_contributions"]=[_ for _ in range(0,n_atoms)]
        FASA["FASA_mod_atomic_dipole_moments_contributions"]=[_ for _ in range(0,n_atoms)]
        FASA["FASA_Components_of_<r^2>"]=[_ for _ in range(0,n_atoms)]
        FASA["FASA_Atomic_electronic_spatial_extent_<r^2>"]=[_ for _ in range(0,n_atoms)]
        for i in range (0,len(lines)):
            if lines[i].startswith("                           *****  Atom "): atom_index= int(lines[i].split()[2].split("(")[0])-1
            #if lines[i].startswith(" Atomic charge:"): AIM["AIM_charges"][atom_index]=float(lines[i].split()[2])
            if lines[i].startswith(" Atomic dipole moments:"):
                l=lines[i+1]
                FASA["FASA_atomic_dipole_moments"][atom_index]=[float(l[3:16]),float(l[19:30]),float(l[35:48])]
                FASA["FASA_mod_atomic_dipole_moments"][atom_index]=float(lines[i+1][54:])
            if lines[i].startswith(" Contribution to molecular dipole moment:"):
                l=lines[i+1]
                FASA["FASA_atomic_dipole_moments_contributions"][atom_index]=[float(l[3:16]),float(l[19:30]),float(l[35:48])]
                FASA["FASA_mod_atomic_dipole_moments_contributions"][atom_index]=float(lines[i+1][54:])
            if lines[i].startswith(" Atomic electronic spatial extent <r^2>:"):
                FASA["FASA_Atomic_electronic_spatial_extent_<r^2>"][atom_index]=float(lines[i].split("<r^2>:")[1])
                if lines[i+1].startswith(" Components of <r^2>:"):
                    w=lines[i+1].split()
                    FASA["FASA_Components_of_<r^2>"][atom_index]=[float(w[4]),float(w[6]),float(w[8])]
            if lines[i].startswith(" Molecular dipole moment (Debye):"):
                w=lines[i].split()
                FASA["FASA_molecular_dipole_moment"]=[float(w[4]),float(w[5]),float(w[6])]
            if lines[i].startswith(" Magnitude of molecular dipole moment (a.u.&Debye):"):
                w=lines[i].split()
                FASA["FASA_mod_Molecular_dipole_moment"]=float(w[7])
            if lines[i].startswith(" Molecular electronic spatial extent <r^2>:"):
                w=lines[i].split()
                FASA["FASA_mol_electronic_spatial_extent"]=float(w[5])
                if lines[i+1].startswith(" Components of <r^2>:"):
                    w=lines[i+1].split()
                    FASA["FASA_mol_electronic_spatial_extent_components"]=[float(w[4]),float(w[6]),float(w[8])]
            if lines[i].startswith(" Molecular octopole moments (Spherical harmonic form):"):
                w=lines[i+3].split("|Q_3|=")
                FASA["magnitude_Q3"]=float(w[-1])
        return FASA


#atomid dipole moments and AIM charges
def multwfn_AIM(filename,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    xyz_file=filename+".molden.xyz"
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input <AIM.multiwfn> null  -nt "+str(n_proc)
    os.system(command)

    with open(xyz_file,"r") as f: n_atoms=int( f.readlines()[0]   )
    with open("multipole.txt","r") as f: lines=f.readlines()
    AIM={}
    AIM["AIM_charges"]=[0.0 for _ in range(0,n_atoms)]
    AIM["AIM_atomic_dipole_moments"]=[[0.0,0.0,0.0] for _ in range(0,n_atoms)]
    AIM["AIM_mod_atomic_dipole_moments"]=[0.0 for _ in range(0,n_atoms)]
    for i in range (0,len(lines)):
        if lines[i].startswith(" *****  Result of atom"): atom_index= int(lines[i].split()[4])-1
        if lines[i].startswith(" Atomic charge:"): AIM["AIM_charges"][atom_index]=float(lines[i].split()[2])
        if lines[i].startswith(" Basin dipole moments:"):
            w=lines[i+1].split()
            AIM["AIM_atomic_dipole_moments"][atom_index]=[float(w[1]),float(w[3]),float(w[5])]
            AIM["AIM_mod_atomic_dipole_moments"][atom_index]=float(w[7])
    return AIM


def multwfn_v_at_nucleus(filename,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    v_at_nucl={}
    vs=[]
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input < get_v_at_nucleus.multiwfn > v_at_nucleus.txt  -nt "+str(n_proc)
    print (command)
    os.system(command)
    with open("v_at_nucleus.txt","r") as f: lines=f.readlines()
    for i in range(0,len(lines)):
        if " Total ESP without contribution from nuclear charge of atom" in lines[i]:
            vs.append(float(lines[i+1].split("eV,")[1].split("kcal/mol)")[0]))
    v_at_nucl["v_at_nucleus"]=vs
    return v_at_nucl


#methods for obtaining the gradient of the electrostatic potential by numerical differentation

def prepare_surf(filename,out_file="",spacing=0.25,isovalue=0.001,n_proc=28):
    if out_file=="": out_file=filename+"_"+str(spacing).replace(".","")+"_"+str(isovalue).replace(".","")+".vtx.txt"
    with open("prep-surf.multiwfn","w") as f:        f.write("12\n3\n"+str(spacing)+"\n1\n1\n"+str(isovalue)+"\n0\n7\nq")
    command="/home/multiwfn_3.8/Multiwfn "+filename+".molden.input < prep-surf.multiwfn > null  -nt "+str(n_proc)
    print (command)
    os.system(command)
    command="mv vtx.txt "+out_file
    print (command)
    os.system(command)

def gradient(center,points,norm_points=[],rext=0.725,rint=0.4):
    surf_neighbour_points=[center]
    all_points=np.concatenate((points,norm_points))
    for p in points: 
        if abs(p[0]-center[0])<rint and abs(p[1]-center[1])<rint and abs(p[2]-center[2])<rint:   
                d=np.linalg.norm(p[:3]-center[:3])
                if d<rint:
                    surf_neighbour_points.append(p)
    surf_neighbour_points=np.array(surf_neighbour_points)

    neighbour_points=[]
    for p in all_points:
        if abs(p[0]-center[0])<rext and abs(p[1]-center[1])<rext and abs(p[2]-center[2])<rext: 
            d=np.linalg.norm(p[:3]-center[:3])
            if d<rext:
                neighbour_points.append(p)
    neighbour_points=np.array(neighbour_points)
    
    n_surf_points=len(surf_neighbour_points) #for debuggin
    n_points=len(neighbour_points) #for debuggin

    #determine the normal of the surface using the PCA analysis
    ev,eig=np.linalg.eig(np.cov(surf_neighbour_points[:,0:3],rowvar=False))
    norm=eig[:,np.argmin(abs(ev))]
    if norm.dot(np.array(center[:3]))<0: norm=-norm/np.linalg.norm(norm)  #ensure that norm is always pointing otwards (it makes use of the origin of coordinates being inside the surface)
    else: norm=norm/np.linalg.norm(norm)

    #A=np.array([[t[0]-center[0],t[1]-center[1],t[2]-center[2]]  for t in three_points])
    B=np.array([tp[5]-center[5] for tp in neighbour_points])
    A=np.array([[tt-c for tt,c in zip(t[:3],center[:3])]  for t in neighbour_points])
    #if len(B)==3: #very unlikely...
    #    A_inv=  np.linalg.inv(A)
    #    gradient= A_inv.dot(B)
    #else: 
    At=np.transpose(A)
    gradient= np.linalg.inv(At.dot(A)).dot(At).dot(B)
    norm_gradient=gradient.dot(norm)*norm
    surf_gradient=gradient-norm_gradient

    return surf_gradient,norm_gradient,gradient,center[:3],center[5],n_surf_points,n_points  



def calc_gradient_from_molden(filename,spacing1=0.25,spacing2=0.35,spacing3=0.4,n_proc=28,filter="auto"):

    points,e_points,i_points=[],[],[]

    prepare_surf(filename=filename,out_file="o_vtx.txt" ,spacing=spacing1,isovalue=0.001,n_proc=n_proc)
    with open ("o_vtx.txt","r") as f: lines=f.readlines()
    number_of_points=int(lines[0])
    points=np.array([[float(w) for w in line.split()[0:6]] for line in lines[1:]])

    prepare_surf(filename=filename,out_file="e1_vtx.txt" ,spacing=spacing3,isovalue=0.0009,n_proc=n_proc)
    with open ("e1_vtx.txt","r") as f: lines=f.readlines()[1:]
    e_points=np.array([[float(w) for w in line.split()[0:6]] for line in lines])

    prepare_surf(filename=filename,out_file="e2_vtx.txt" ,spacing=spacing2,isovalue=0.00099,n_proc=n_proc)
    with open ("e2_vtx.txt","r") as f: lines=f.readlines()[1:]
    e2_points=np.array([[float(w) for w in line.split()[0:6]] for line in lines])
    e_points=np.concatenate((e_points,e2_points))

    prepare_surf(filename=filename,out_file="i1_vtx.txt" ,spacing=spacing3,isovalue=0.0011,n_proc=n_proc)
    with open ("i1_vtx.txt","r") as f: lines=f.readlines()[1:]
    i_points=np.array([[float(w) for w in line.split()[0:6]] for line in lines])

    prepare_surf(filename=filename,out_file="i2_vtx.txt" ,spacing=spacing2,isovalue=0.00101,n_proc=n_proc)
    with open ("i2_vtx.txt","r") as f: lines=f.readlines()[1:]
    i2_points=np.array([[float(w) for w in line.split()[0:6]] for line in lines])
    i_points=np.concatenate((i_points,i2_points))

    n_points=np.concatenate((e_points,i_points))


    rext=spacing1*2*2**(1/2)
    rint=spacing2*3**(1/2)

    if filter=="auto":
        if number_of_points<10000: filter=0.4
        elif number_of_points>30000: filter=0.8
        else:
            filter = 0.4 + 0.4*(number_of_points-1000)/20000
    """
    # method to calculate the distance between layers, for debuggin
    for p in points:
        distances=[]
        if np.random.random()>0.99:
            dist=[np.linalg.norm(pp-p) for pp in n_points]
            distances.append(np.min(dist))
    print (distances) 
    print (np.average(distances))
    print (np.std(distances))  
    """ 


    pool=mp.Pool(n_proc)
    c_time=time.time()
    results=[]
    for p in points:
        if np.random.random()>filter:
            pool.apply_async(gradient, args=(p,points,n_points,rext,rint), callback=results.append )
            #results.append(gradient(p,points,n_points,rext,rint))
    pool.close()
    pool.join()

    surf_gradients=np.array([r[0] for r in results])/(0.529177)
    norm_gradients=np.array([r[1] for r in results])/(0.529177)
    all_gradients =np.array([r[2] for r in results])/(0.529177)
    surf_points   =np.array([r[3] for r in results])
    potentials    =np.array([r[4] for r in results])
    ev_points_surf=np.array([r[5] for r in results])
    ev_points_norm=np.array([r[6] for r in results])
    gradients_mod=np.array([np.linalg.norm(v) for v in all_gradients])
    norm_gradients_mod=np.array([np.linalg.norm(v) for v in norm_gradients])
    surf_gradients_mod=np.array([np.linalg.norm(v) for v in surf_gradients])
    surf_norm_grad_ratio=surf_gradients_mod/norm_gradients_mod
    angles=np.arctan(surf_norm_grad_ratio)

    features={}
    features["v_gradient_mod_average"]=np.average(gradients_mod)
    features["tangential_v_gradients_mod_average"]=np.average(surf_gradients_mod)
    features["normal_v_gradients_mod_average"]=np.average(norm_gradients_mod)
    features["v_gradient_0.95_quantile"]=np.quantile(gradients_mod,0.95)
    features["tangential_v_gradient_0.95_quantile"]=np.quantile(surf_gradients_mod,0.95)
    features["normal_v_gradient_0.95_quantile"]=np.quantile(norm_gradients_mod,0.95)
    features["v_gradient_0.9_quantile"]=np.quantile(gradients_mod,0.9)
    features["tangential_v_gradient_0.9_quantile"]=np.quantile(surf_gradients_mod,0.9)
    features["normal_v_gradient_0.9_quantile"]=np.quantile(norm_gradients_mod,0.9)
    features["v_gradient_0.75_quantile"]=np.quantile(gradients_mod,0.75)
    features["tangential_v_gradient_0.75_quantile"]=np.quantile(surf_gradients_mod,0.75)
    features["normal_v_gradient_0.75_quantile"]=np.quantile(norm_gradients_mod,0.75)
    features["v_gradient_0.5_quantile"]=np.quantile(gradients_mod,0.5)
    features["tangential_v_gradient_0.5_quantile"]=np.quantile(surf_gradients_mod,0.5)
    features["normal_v_gradient_0.5_quantile"]=np.quantile(norm_gradients_mod,0.5) 
    features["v_gradient_angle_norm_average"]=np.average(angles) 
    features["v_gradient_angle_norm_0.95_quantile"]=np.quantile(angles,0.95)
    features["v_gradient_angle_norm_0.9_quantile"]=np.quantile(angles,0.9)
    features["v_gradient_angle_norm_0.75_quantile"]=np.quantile(angles,0.75)
    features["v_gradient_angle_norm_0.5_quantile"]=np.quantile(angles,0.55)
    features["v_gradient_mod/v_average"]=np.average(gradients_mod/abs(potentials))
    features["tangential_v_gradient_mod/v_average"]=np.average(surf_gradients_mod/abs(potentials))
    features["normal_v_gradient_mod/v_average"]=np.average(norm_gradients_mod/abs(potentials))
    features["v_gradient_mod/v_0.95_quantile"]=np.quantile(gradients_mod/abs(potentials),0.95)
    features["tangential_v_gradient_mod/v_0.95_quantile"]=np.quantile(surf_gradients_mod/abs(potentials),0.95)
    features["normal_v_gradient_mod/v_0.95_quantile"]=np.quantile(norm_gradients_mod/abs(potentials),0.95)
    features["v_gradient_mod/v_0.9_quantile"]=np.quantile(gradients_mod/abs(potentials),0.9)
    features["tangential_v_gradient_mod/v_0.9_quantile"]=np.quantile(surf_gradients_mod/abs(potentials),0.9)
    features["normal_v_gradient_mod/v_0.9_quantile"]=np.quantile(norm_gradients_mod/abs(potentials),0.9)
    features["v_gradient_mod/v_0.75_quantile"]=np.quantile(gradients_mod/abs(potentials),0.75)
    features["tangential_v_gradient_mod/v_0.75_quantile"]=np.quantile(surf_gradients_mod/abs(potentials),0.75)
    features["normal_v_gradient_mod/v_0.75_quantile"]=np.quantile(norm_gradients_mod/abs(potentials),0.75)
    features["v_gradient_mod/v_0.5_quantile"]=np.quantile(gradients_mod/abs(potentials),0.5)
    features["tangential_v_gradient_mod/v_0.5_quantile"]=np.quantile(surf_gradients_mod/abs(potentials),0.5)
    features["normal_v_gradient_mod/v_0.5_quantile"]=np.quantile(norm_gradients_mod/abs(potentials),0.5)
    
    #for debuggin
    """
    print ("details of the num. differenciation algorithm")
    print ("for gradient on the surface")
    print ("average number of points:"+str(np.mean(ev_points_surf)))
    print ("max number of points:" + str(np.max(ev_points_surf)))
    print ("min number of points:" + str(np.min(ev_points_surf)))
    print ("for 3D gradient")
    print ("average number of points:"+str(np.mean(ev_points_surf+ev_points_norm)))
    print ("max number of points:" + str(np.max(ev_points_surf+ev_points_norm)))
    print ("min number of points:" + str(np.min(ev_points_surf+ev_points_norm)))
    """

    """
    #for debbuggin
    print ("results:")
    print ("average modululs value of tantential component of gradient: "+str(np.average(surf_gradients_mod)))
    print ("average modululs value of normal component of gradient: "+str(np.average(norm_gradients_mod)))
    print ("0.5 quantile of modulus of tangential component of gradient: "+str(np.quantile(surf_gradients_mod,0.5)) )
    print ("0.5 quantile of modulus of normal component of gradient: "+str(np.quantile(norm_gradients_mod,0.5)) )
    print ("0.75 quantile of modulus of tangential component of gradient: "+str(np.quantile(surf_gradients_mod,0.75)) )
    print ("0.75 quantile of modulus of normal component of gradient: "+str(np.quantile(norm_gradients_mod,0.75)) )
    print ("0.90 quantile of modulus of tangential component of gradient: "+str(np.quantile(surf_gradients_mod,0.90)) )
    print ("0.90 quantile of modulus of normal component of gradient: "+str(np.quantile(norm_gradients_mod,0.90)) )
    print ("0.95 quantile of modulus of tangential component of gradient: "+str(np.quantile(surf_gradients_mod,0.95)) )
    print ("0.95 quantile of modulus of normal component of gradient: "+str(np.quantile(norm_gradients_mod,0.95)) )
    print ()
    print ("average value of grad_surf/grad_norm"+str( np.average([ gs/gn for gs,gn in zip(surf_gradients_mod,norm_gradients_mod) ])  )    )
    print ("std dev. grad_surf/grad_norm"+str( np.std([ gs/gn for gs,gn in zip(surf_gradients_mod,norm_gradients_mod) ])  )  )
    print ("average value of the angle with the normal:" + str(np.average(angles)))
    print ("std dev of the angle with the normal:" + str(np.average(angles)))
    print ("0.5 quantile value of the angle with the normal surface: "+str(np.quantile(angles,0.5) ))
    print ("0.75 quantile of the angle with the normal surface: "+str(np.quantile(angles,0.75) ))
    print ("0.9 quantile of the angle with the normal surface: "+str(np.quantile(angles,0.90) ))
    print ("0.95 quantile of the angle with the normal surface: "+str(np.quantile(angles,0.95) ))
    print ("fraction of points with angle<45 degrees with the surface: "+str( np.sum([a>np.pi/4 and a<3*np.pi/4 for a in angles])/len(angles)))
    print ("fraction of points with angle<30 degrees with the surface: "+str( np.sum([a>2*np.pi/6 and a<4*np.pi/6 for a in angles])/len(angles)))
    print ("fraction of points with angle<15 degrees with the surface: "+str( np.sum([a>5*np.pi/12 and a<7*np.pi/12 for a in angles])/len(angles))) 
    """

    """
    import matplotlib.pyplot as plt 
    plt.scatter(potentials,parameter, c=[surf_gradients_mod])
    plt.savefig(filename+"-parameter"+".png")
    plt.clf()
    plt.scatter(potentials,surf_norm_grad_ratio, c=[surf_gradients_mod])
    plt.savefig(filename+"surf_norm"+".png")
    plt.clf()
    plt.scatter(potentials,surf_gradients_mod,c=[surf_norm_grad_ratio])
    plt.savefig(filename+"surf.png")
    plt.clf()
    plt.scatter(potentials,norm_gradients_mod,c=[surf_norm_grad_ratio])
    plt.savefig(filename+"norm.png")
    plt.clf()
    plt.scatter(potentials,angles,c=[((surf_gradients_mod)**2+(norm_gradients_mod)**2)**1/2])
    plt.savefig(filename+"ang.png")
    plt.clf()
    plt.scatter(norm_gradients_mod,surf_gradients_mod)
    plt.savefig(filename+"surf_vs_norm.png")
    """

    return features


def rdg_analysis_weak_inter(filename,lim_weak=0.5,lim_h_bonds=-0.01,lim_steric=0.01,promolecular=False,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]

    if promolecular==False:
        command=multiwfn_exe+" "+filename+".molden.input < rdg_plot.multiwfn > null  -nt "+str(n_proc)
    else: 
        command=multiwfn_exe+" "+filename+".molden.input < rdg_promol_plot.multiwfn > null  -nt "+str(n_proc)

    print (command)
    os.system(command)

    rdg_analysis={}
    with open ("output.txt") as f: lines=f.readlines()
    h_bonds,vdw,steric=0,0,0
    for l in lines:
        w=[float(e) for e in l.split()]
        if w[4]<lim_weak:
            if w[3]<lim_h_bonds: h_bonds+=1
            elif w[3]>lim_h_bonds and w[3]<lim_steric: vdw+=1
            else: steric+=1
    if promolecular==False:
        rdg_analysis["fraction_hb_in_rdg_plot"]=h_bonds/len(lines)
        rdg_analysis["fraction_vdw_in_rdg_plot"]=vdw/len(lines)
        rdg_analysis["fraction_steric_in_rdg_plot"]=steric/len(lines)
    else:
        rdg_analysis["fraction_hb_in_promolecular_rdg_plot"]=h_bonds/len(lines)
        rdg_analysis["fraction_vdw_in_promolecular_rdg_plot"]=vdw/len(lines)
        rdg_analysis["fraction_steric_in_promolecular_rdg_plot"]=steric/len(lines)
    return rdg_analysis


#molecular properties:
def multwfn_ALIE(filename,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    xyz_file=filename+".molden.xyz"
    os.environ["Multiwfnpath"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input <ALIE.multiwfn> ALIE.txt  -nt "+str(n_proc)
    os.system(command)
    with open("ALIE.txt","r") as f: lines=f.readlines()
    start=lines.index('       ================= Summary of surface analysis =================\n')
    end=lines.index(" Surface analysis finished!\n")  
    useful_lines=lines[start+1:end]
    ALIE={}

    ALIE["alie_min_val"],ALIE["alie_max_val"]=float(useful_lines[3].split()[2])*23.0605631,float(useful_lines[3].split()[6])*23.0605631
    ALIE["alie_overall_avg_value"]=float(useful_lines[7].split()[2])*23.0605631
    ALIE["alie_overall_variance"]=float(useful_lines[8].split()[1])*531.78957


    start=lines.index(' Note: The atoms having zero surface area (i.e. buried) are not shown below\n')
    end=lines.index(' If outputting the surface facets to locsurf.pdb in current folder? By which you can visualize local surface via third-part visualization program such as VMD (y/n)\n')    
    useful_lines=lines[start+1:end]
    with open(xyz_file,"r") as f: n_atoms=int( f.readlines()[0]   )  

    qams_atom_overall_surf_area=[0.0 for _ in range(0,n_atoms)]
    alie_atom_min_val=[0.0 for _ in range(0,n_atoms)]
    alie_atom_max_val=[0.0 for _ in range(0,n_atoms)]
    alie_atom_overall_avg=[0.0 for _ in range(0,n_atoms)]
    alie_atom_overall_variance=[0.0 for _ in range(0,n_atoms)]
    i=3
    while useful_lines[i].strip()!="":
        w=useful_lines[i].split()
        atom=int(w[0])-1

        alie_atom_min_val[atom]=float(w[2])
        alie_atom_max_val[atom]=float(w[3])
        alie_atom_overall_avg[atom]=float(w[4])
        alie_atom_overall_variance[atom]=float(w[5])
        i+=1

    ALIE["alie_atom_overall_avg"]=alie_atom_overall_avg
    ALIE["alie_atom_overall_variance"]=alie_atom_overall_variance
    ALIE["alie_atom_min_val"]=alie_atom_min_val
    ALIE["alie_atom_max_val"]=alie_atom_max_val

    return ALIE

#molecular properties:
def multwfn_LEA(filename,n_proc,multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    xyz_file=filename+".molden.xyz"
    os.environ["LEA"]=multiwfn_home
    command=multiwfn_exe+" "+filename+".molden.input <LEA.multiwfn> LEA.txt  -nt "+str(n_proc)
    os.system(command)
    with open("LEA.txt","r") as f: lines=f.readlines()
    start=lines.index('       ================= Summary of surface analysis =================\n')
    end=lines.index(" Surface analysis finished!\n")  
    useful_lines=lines[start+1:end]
    LEA={}


    LEA["lea_min_val"],LEA["lea_max_val"]=float(useful_lines[3].split()[2]),float(useful_lines[3].split()[6])
    LEA["lea_overall_surf_area"]=float(useful_lines[4].split()[6])
    LEA["lea_pos_surf_area"]=float(useful_lines[5].split()[6])
    LEA["lea_neg_surf_area"]=float(useful_lines[6].split()[6])
    LEA["lea_overall_avg_value"]=float(useful_lines[7].split()[6])
    v=useful_lines[8].split()[6]
    if v=="NaN": LEA["lea_pos_avg_value"]=0.0
    else:  LEA["lea_pos_avg_value"]=float(v)
    v=useful_lines[9].split()[6]
    if v=="NaN": LEA["lea_neg_avg_value"]=0.0
    else: LEA["lea_neg_avg_value"]=float(v)
    LEA["lea_overall_variance"]=float(useful_lines[10].split()[7])
    LEA["lea_pos_variance"]=float(useful_lines[11].split()[7])
    LEA["lea_neg_variance"]=float(useful_lines[12].split()[7])


    start=lines.index(' Note: The atoms having zero surface area (i.e. buried) are not shown below\n')
    end=lines.index(' If outputting the surface facets to locsurf.pdb in current folder? By which you can visualize local surface via third-part visualization program such as VMD (y/n)\n')    
    useful_lines=lines[start+1:end]
    with open(xyz_file,"r") as f: n_atoms=int( f.readlines()[0]   )  

    lea_atom_overall_surf_area=[0.0 for _ in range(0,n_atoms)]
    lea_atom_pos_surf_area=[0.0 for _ in range(0,n_atoms)]
    lea_atom_neg_surf_area=[0.0 for _ in range(0,n_atoms)]
    lea_atom_min_val=[0.0 for _ in range(0,n_atoms)]
    lea_atom_max_val=[0.0 for _ in range(0,n_atoms)]
    i=3
    while useful_lines[i].strip()!="":
        w=useful_lines[i].split()
        atom=int(w[0])-1
        lea_atom_overall_surf_area[atom]=float(w[1])
        lea_atom_pos_surf_area[atom]=float(w[2])
        lea_atom_neg_surf_area[atom]=float(w[3])
        lea_atom_min_val[atom]=float(w[4])
        lea_atom_max_val[atom]=float(w[5])
        i+=1
    LEA["lea_atom_overall_surf_area"]=lea_atom_overall_surf_area
    LEA["lea_atom_pos_surf_area"]=lea_atom_pos_surf_area
    LEA["lea_atom_neg_surf_area"]=lea_atom_neg_surf_area
    LEA["lea_atom_min_val"]=lea_atom_min_val
    LEA["lea_atom_max_val"]=lea_atom_max_val


    lea_atom_overall_avg=[0.0 for _ in range(0,n_atoms)]
    lea_atom_pos_avg=[0.0 for _ in range(0,n_atoms)]
    lea_atom_neg_avg=[0.0 for _ in range(0,n_atoms)]
    lea_atom_overall_variance=[0.0 for _ in range(0,n_atoms)]
    lea_atom_pos_variance=[0.0 for _ in range(0,n_atoms)]
    lea_atom_neg_variance=[0.0 for _ in range(0,n_atoms)]
    i+=3
    while useful_lines[i].strip()!="":
        w=useful_lines[i].split()
        atom=int(w[0])-1  
        lea_atom_overall_avg[atom]=float(w[1])
        if w[2]=="NaN": lea_atom_pos_avg[atom]=0.0
        else:           lea_atom_pos_avg[atom]=float(w[2])
        if w[3]=="NaN": lea_atom_neg_avg[atom]=0.0
        else:           lea_atom_neg_avg[atom]=float(w[3])
        if w[4]=="NaN": lea_atom_overall_variance[atom]=0.0
        else:           lea_atom_overall_variance[atom]=float(w[4])
        if w[5]=="NaN": lea_atom_pos_variance[atom]=0.0
        else:           lea_atom_pos_variance[atom]=float(w[5])
        if w[6]=="NaN": lea_atom_neg_variance[atom]=0.0
        else:           lea_atom_neg_variance[atom]=float(w[6])
        i+=1 
    LEA["lea_atom_overall_avg"]=lea_atom_overall_avg
    LEA["lea_atom_pos_avg"]=lea_atom_pos_avg
    LEA["lea_atom_neg_avg"]=lea_atom_neg_avg
    LEA["lea_atom_overall_variance"]=lea_atom_overall_variance
    LEA["lea_atom_pos_variance"]=lea_atom_pos_variance
    LEA["lea_atom_neg_variance"]=lea_atom_neg_variance

    return LEA





def extract_features(filename,n_proc=14,req_features=[],multiwfn_home="/home/multiwfn_3.8",multiwfn_exe="/home/multiwfn_3.8/Multiwfn",orca_mkl_exe="/home/orca5/orca_2mkl"):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    beg_time=time.time()
    print("in extract features")
    if not os.path.isfile(filename.split(".gbw")[0]+".molden.input"):
        create_molden_input(filename,orca_mkl_exe=orca_mkl_exe)
        correct_molden_input(filename)
    n_atoms=read_number_of_atoms (filename.split(".gbw")[0]+".molden.input")
    generate_multiwfn_job_files(n_atoms)
    #if not (os.path.isfile("get_xyz.multiwfn") and os.path.isfile("expensive-IBSI-BO.multiwfn")):
    #    generate_multiwfn_job_files()

    if not os.path.isfile(filename.split(".gbw")[0]+".xyz"):
        get_xyz(filename,multiwfn_exe=multiwfn_exe)

    chg_methods=[]
    bo_methods=[]
    if req_features==[]:
        chg_methods=["hirshfeld","voronoy","mulliken","lowdin","becke","ADCH","CHELPG","MK","CM5","12CM5","RESP","PEOE"]
        bo_methods=["mayer","wiberg","mulliken","fuzzy","laplacian"]

    else:
        if "chg_hirshfeld" in req_features: chg_methods.append("hirshfeld")
        if "chg_voronoy" in req_features: chg_methods.append("voronoy")
        if "chg_mulliken" in req_features: chg_methods.append("mulliken")
        if "chg_lowdin" in req_features: chg_methods.append("lowdin")
        if "chg_becke" in req_features: chg_methods.append("becke")
        if "chg_ADCH" in req_features: chg_methods.append("ADCH")
        if "chg_CHELPG" in req_features: chg_methods.append("CHELPG")
        if "chg_MK" in req_features: chg_methods.append("MK")                                                            
        if "chg_CM5" in req_features: chg_methods.append("CM5") 
        if "chg_12CM5" in req_features: chg_methods.append("12CM5") 
        if "chg_RESP" in req_features: chg_methods.append("RESP") 
        if "chg_PEOE" in req_features: chg_methods.append("PEOE") 
        if "bo_mayer" in req_features: bo_methods.append("mayer")
        if "bo_wiberg" in req_features: bo_methods.append("wiberg")
        if "bo_mulliken" in req_features: bo_methods.append("mulliken")
        if "bo_fuzzy" in req_features: bo_methods.append("fuzzy")
        if "bo_laplacian" in req_features: bo_methods.append("laplacian")


    features={}
    #chg_methods=["hirshfeld","voronoy","mulliken","lowdin","becke","ADCH","CHELPG","MK","CM5","12CM5","RESP","PEOE"]
    for method in chg_methods: 
        features["chg_"+method]=multwfn_chrg(filename,method,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe)

    #bo_methods=["mayer","wiberg","mulliken","fuzzy","laplacian"]
    for method in bo_methods: 
        features["bo_"+method]=multwfn_BO(filename,method,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe)

    if req_features==[] or "bo_IBSI" in req_features:
        features["bo_IBSI"]=multwfn_IBSI_BO(filename,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe)

    if req_features==[] or "qams" in req_features:
        features.update(multwfn_QAMS(filename,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))

    if req_features==[] or "fasa" in req_features:
        features.update(multwfn_FASA(filename,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))

    #currently aim is not calculated by default because multwfn manual says that there could be problems with ECP and I atoms have ECP
    if "aim" in req_features:
        features.update(multwfn_AIM(filename,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))

    if req_features==[] or "v_at_nucl" in req_features:
        features.update(multwfn_v_at_nucleus (filename,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))
        
    if req_features==[] or "grad_v" in req_features:
        features.update(calc_gradient_from_molden(filename,spacing1=0.25,spacing2=0.35,spacing3=0.4,n_proc=n_proc,filter="auto"))

    if req_features==[] or "ALIE" in req_features:
        features.update(multwfn_ALIE(filename,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))

    if req_features==[] or "LEA" in req_features:
        features.update(multwfn_LEA(filename,n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))

    if req_features==[] or "rdg" in req_features:
        features.update(rdg_analysis_weak_inter(filename,promolecular=False,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))

    if req_features==[] or "rdg_promolecular" in req_features:
        features.update(rdg_analysis_weak_inter(filename,promolecular=True,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe))

    json_text=json.dumps(features,indent=4)
    with open(filename+".multwfn.json","w") as f: f.write(json_text)
    print("to json:"+str(time.time()-beg_time))

        

if __name__ == "__main__":
    
    print("in do_multwfn.py")
    gbw_file=sys.argv[1]
    n_proc=int(sys.argv[2])
    if len(sys.argv)>3 and sys.argv[3]=="6": orca_version=6
    else: orca_version=5
    req_features=sys.argv[4:]
    if req_features==["all"]: req_features=[]
    directory="/".join(gbw_file.split("/")[:-1])
    file=gbw_file.split("/")[-1]
    os.chdir(directory)
    os.environ["Multiwfnpath"]=multiwfn_home
    if orca_version==6: orca_mkl_exe=orca6_mkl_exe
    else: orca_mkl_exe=orca5_mkl_exe
    extract_features(file,req_features=req_features,n_proc=n_proc,multiwfn_home=multiwfn_home,multiwfn_exe=multiwfn_exe,orca_mkl_exe=orca_mkl_exe)

























