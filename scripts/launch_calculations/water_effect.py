#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import string
import os
import os.path
import sys
import copy
import numpy as np
import time 



#print xyz file (needed for some methods)
def get_xyz(filename):
    if filename[-4:]==".gbw": filename=filename.split(".gbw")[0]
    elif filename[-13:]==".molden.input":filename=filename.split(".molden.input")[0]
    os.environ["Multiwfnpath"]="/home/multiwfn_3.8"
    with open("get_xyz.multiwfn","w") as f:        f.write("100\n2\n2\n\n\nq\n") 
    command="/home/multiwfn_3.8/Multiwfn "+filename+".molden.input < get_xyz.multiwfn > null  "
    os.system(command)
    command="mv "+filename+".molden.xyz temp.xyz"
    os.system(command)

def read_number_of_atoms(molden_input_file):
    with open (molden_input_file,"r") as f: lines=f.readlines()
    for i in range(0,len(lines)):
        if lines[i].startswith("[Atoms] AU"):
            n_atoms=0
            while "[GTO]" not in lines[i+n_atoms]: n_atoms+=1
            break
    return n_atoms

def get_charge(file_name):
    if "-2an" in file_name: return "-2"
    elif "-3an" in file_name: return "-3"
    elif "-4an" in file_name: return "-4"
    elif "-5an" in file_name: return "-5"
    elif "-6an" in file_name: return "-6"
    elif "-an" in file_name: return "-1"
    elif "-neut" in file_name: return "0"
    elif "-2cation" in file_name: return "2"
    elif "-cation" in file_name: return "1"
    elif "-3cation" in file_name: return "3"
    elif "-4cation" in file_name: return "4"
    elif "-5cation" in file_name: return "5"
    elif "-6cation" in file_name: return "6"


def energy_far_water(xyz_file="temp.xyz",options="--gfn2 --gbsa h2o",n_waters=1,nprocs=28,chrg=0):
    with open(xyz_file,"r") as f: xyz_lines=f.readlines()
    xyz_lines=[l for l,i in zip(xyz_lines,range(0,len(xyz_lines))) if l.strip()!="" or i<3]

    old_n_atoms=int(xyz_lines[0].strip())

    new_xyz_file_text=str(old_n_atoms+3*n_waters)+"\n\n"
    for l in xyz_lines[2:]: new_xyz_file_text+=l

    #up to 46 molecules of water can be added (add more different elemeents to the list if required)
    water_positions=[[40,40,40],[40,0,0],[0,40,0],[0,0,40],[40,40,0],[40,0,40],[0,40,40]]
    water_positions+=[[-40,-40,-40],[-40,0,0],[0,-40,0],[0,0,-40],[-40,-40,0],[-40,0,-40],[0,-40,-40]]
    water_positions+=[[40,-40,-40],[-40,40,40],[40,-40,0],[40,0,-40],[-40,40,0],[-40,0,40]]
    water_positions+=[[60,60,60],[60,0,0],[0,60,0],[0,0,60],[60,60,0],[60,0,60],[0,60,60]]
    water_positions+=[[-60,-60,-60],[-60,0,0],[0,-60,0],[0,0,-60],[-60,-60,0],[-60,0,-60],[0,-60,-60]]
    water_positions+=[[60,40,0],[0,60,40],[40,0,60],[60,0,40],[40,60,0],[0,40,60],[60,60,40],[60,40,60],[40,60,60]]
    water_positions+=[[60,40,40],[40,60,40],[40,40,60]]
    for w in range(0,n_waters):
        O= np.array([ -0.1918040235  ,     1.3862489483    ,    0.0047370042])+water_positions[w]
        H1=np.array([  0.7660977787  ,     1.3911615443    ,   -0.0141642652])+water_positions[w]
        H2=np.array([ -0.4927337474  ,     1.6150799341    ,   -0.8756928250])+water_positions[w]
        new_xyz_file_text+="O  "+"{:15.5}".format(O[0])+"{:15.5}".format(O[1])+"{:15.5}".format(O[2])+"\n"
        new_xyz_file_text+="H  "+"{:15.5}".format(H1[0])+"{:15.5}".format(H1[1])+"{:15.5}".format(H1[2])+"\n"
        new_xyz_file_text+="H  "+"{:15.5}".format(H2[0])+"{:15.5}".format(H2[1])+"{:15.5}".format(H2[2])+"\n"

    with open ("water_fery_far.xyz","w") as f:f.write(new_xyz_file_text)
    command="xtb water_fery_far.xyz "+options+ " --T "+str(nprocs) +" --chrg  "+str(chrg) +" > energy_far_water.out"
    print (command)
    os.system(command)
    with open("energy_far_water.out","r") as f: out_lines=f.readlines()
    for l in out_lines:
        if l.startswith("          | TOTAL ENERGY      "): energy= float(l.split()[3])   
    with open("water_fery_far.xyz","r") as f: lines=f.readlines() 
    lines[1]=" "+str(energy)+"\n"
    with open("water_fery_far.xyz","w") as f: f.write("".join(lines)) 


def run_crest_qgc(xyz_file="temp.xyz", options="--gfn2 --gbsa h2o",n_waters=1,nprocs=28,chrg=0):

    with open(xyz_file,"r") as f: xyz_lines=f.readlines()
    old_n_atoms=int(xyz_lines[0].strip())
    with open(".xcontrol","w") as f: f.write("$constrain\n   force constant=3.0\n   atoms: 1-"+str(old_n_atoms)+"\n$end")

    with open("water.xyz","w") as f:
        f.write(" 3\n\nO         -0.1918040235        1.3862489483        0.0047370042\nH          0.7660977787        1.3911615443       -0.0141642652\nH         -0.4927337474        1.6150799341       -0.8756928250\n")
    
    os.system("rm -rf grow")
    
    command="/home/xtb/bin/crest "+str(xyz_file)+" --qcg water.xyz --nsolv "+str(n_waters)+" --T "+str(nprocs)+" --chrg "+str(chrg)+" --ensemble "+options+ " > crest_qgc.out"
    print (command)
    os.system(command)
    
    with open("crest_qgc.out","r") as f: lines=f.readlines()
    for l in lines: 
        if l.find("G /Eh     :")>-1: solv_energy=float( l.split("G /Eh     :")[1]   )
    
    return solv_energy



def put_water(file_name,suffix="_1wat.conformers.gfn2gbsa.xyz",options="--gfn2 --gbsa h2o",nprocs=28,n_waters=1,launch_directory="/home/lsimon/jobs/"):

    chrg=get_charge(file_name)
    if filename[-4:]!=".xyz":
        if file_name[-4:]==".gbw":file_name=file_name.split(".gbw")[0]
        elif file_name[-13:]==".molden.input":file_name=file_name.split(".molden.input")[0]
        get_xyz(file_name)
        energy_far_water(chrg=chrg,options=options,nprocs=nprocs,n_waters=n_waters)
        avg_energy=run_crest_qgc(chrg=chrg,options=options,nprocs=nprocs,n_waters=n_waters)

    else:
        energy_far_water(filename,chrg=chrg,options=options,nprocs=nprocs,n_waters=n_waters)
        avg_energy=run_crest_qgc(filename,chrg=chrg,options=options,nprocs=nprocs,n_waters=n_waters)

    with open("water_fery_far.xyz","r") as f: very_far_lines=f.readlines()
    with open("./ensemble/final_ensemble.xyz","r") as f: ensemble_lines=f.readlines()
    ensemble_lines[1]= ensemble_lines[1].strip()+" (averaged energy: "+str(avg_energy)+" )\n"
    text_for_xyz_file=ensemble_lines+very_far_lines
    if filename[-4:]==".xyz":
        with open(launch_directory+file_name[:-4]+suffix,"w") as f: f.write("".join(text_for_xyz_file))
    else:
        with open(file_name+suffix,"w") as f: f.write("".join(text_for_xyz_file))


if __name__ == "__main__":
    curr_time=time.time()
    n_proc=int(sys.argv[1])
    suffix="wat.conformers.gfn2_gfnff_gbsa.xyz"
    options="--gfn2//gfnff --gbsa h2o"
    options="--gfn2//gfnff --gbsa h2o --fixsolute --nopreopt --xtbiff"
    #options="--gfn2 --gbsa h2o --fixsolute --nopreopt --notpo"
    #options="--gbsa h2o "
    scratch_dir="/scratch/lsimon/"

    n_waters=1
    
    if len(sys.argv)==2:
        os.chdir("/scratch/lsimon/m06gbw-def")
        os.system("export Multiwfnpath=/home/multiwfn_3.8")
        molden_input_files=[f[:-14] for f in os.listdir() if f[-13:]==".molden.input" and not os.path.isfile(os.getcwd()+"/"+f[:-13]+suffix)]
        print (molden_input_files)
        for mif in molden_input_files:
            put_water(mif+".molden.input",nprocs=n_proc,suffix="_"+str(n_waters)+suffix,options=options,n_waters=n_waters)
    else:
        filename=sys.argv[2]
        if len(sys.argv)>3: n_waters=int(sys.argv[3])
        if filename[-4:]==".xyz":
            curr_dir=os.getcwd()
            if not os.path.isdir(scratch_dir+filename[:-4]):  os.mkdir(scratch_dir+filename[:-4])
            os.system("cp "+filename+" "+scratch_dir+filename[:-4]+"/")
            os.chdir(scratch_dir+filename[:-4])
            print (os.getcwd())#borrame
            n_proc=7
            put_water(filename,nprocs=n_proc,suffix="_"+str(n_waters)+suffix,options=options,n_waters=n_waters)
            os.chdir(curr_dir)
        else:
            put_water(filename+".molden.input",nprocs=n_proc,suffix=suffix,options=options,n_waters=n_waters)
    print ("total time: "+str(time.time()-curr_time))
sys.exit()





