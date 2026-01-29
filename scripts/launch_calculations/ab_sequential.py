#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
# script to launch sequentially all calculations: optimiazation, single point calculations, mltwfn properties determination, 
# crest qgc water solvation, etc. To be called from a .qsub file

import string
import os
import os.path
import sys
sys.path.append('../import')
import subprocess
import Molecular_structure
import numpy as np


import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
from routes import orca5_exe,orca6_exe,scratch_home,do_multwfn_exe,multiwfn_exe,multiwfn_home,ofakeG_exe,nbo_orca5_exe,nbo_orca6_exe



files_to_compress=[]

#figure out charge from file name
def get_charge(file_name):
    if "-2an" in file_name: return "-2",file_name.split("-2an")[0]+"-2an"
    elif "-3an" in file_name: return "-3",file_name.split("-3an")[0]+"-3an"
    elif "-4an" in file_name: return "-4",file_name.split("-4an")[0]+"-4an"
    elif "-5an" in file_name: return "-5",file_name.split("-5an")[0]+"-5an"
    elif "-6an" in file_name: return "-6",file_name.split("-6an")[0]+"-6an"
    elif "-an" in file_name: return "-1",file_name.split("-an")[0]+"-an"
    elif "-neut" in file_name: return "0",file_name.split("-neut")[0]+"-neut"
    elif "-2cation" in file_name: return "2",file_name.split("-2cation")[0]+"-2cation"
    elif "-cation" in file_name: return "1",file_name.split("-cation")[0]+"-cation"
    elif "-3cation" in file_name: return "3",file_name.split("-3cation")[0]+"-3cation"
    elif "-4cation" in file_name: return "4",file_name.split("-4cation")[0]+"-4cation"
    elif "-5cation" in file_name: return "5",file_name.split("-5cation")[0]+"-5cation"
    elif "-6cation" in file_name: return "6",file_name.split("-6cation")[0]+"-6cation"


def correct_xyz_file(file_name):
    with open(file_name,"r") as f: lines=f.readlines()
    lines=[line for line in lines if line.strip()!=""]
    carts=[line.split() for line in lines]
    if len(carts[0])==4: #include number of atoms in the first line and name in the second
        new_lines=[str(len(lines))+"\n",file_name.split(".xyz")[0]+"\n"]
        lines=new_lines+lines
    with open(file_name,"w") as f: f.writelines(lines)


#prepare file for optimization
def prepare_optimization(file_name,n_procs=14,opt_lot="PBEh-3c",orca_version=5,skip_optimization=False,reduce_step=False):
    charge,_=get_charge(file_name)
    if file_name[-4:]!=".xyz": file_name+=".xyz" 

    if opt_lot=="PBEh-3c" and orca_version==5: inp_file_name=file_name[:-4]+".inp"
    else: inp_file_name=file_name[:-4]+"-"+opt_lot+".inp"
    if skip_optimization: return inp_file_name

    if opt_lot=="PBEh3c" or opt_lot=="pbeh3c": opt_orca_txt="! PBEh-3c"
    elif opt_lot=="b973c": opt_orca_txt="! B97-3c"
    else: opt_orca_txt="! "+opt_lot
    opt_orca_txt+=' CPCM defgrid1 CHELPG verytightopt Hirshfeld \n  %cpcm smd true\n  SMDsolvent "water"  end\n  \n%geom'
    if reduce_step: opt_orca_txt+='\n   MaxStep 0.03\n'
    opt_orca_txt+='\n   MaxIter 500\n   end\n'
    if orca_version==5:
        opt_orca_txt+='%pal nprocs '+str(n_procs)+'\nend\n%MaxCore 3906\n %freq\n Numfreq True\n Quasirrho True\n Cutofffreq 100.0\n end\n '
    elif orca_version==6:
        opt_orca_txt+='%pal nprocs '+str(n_procs)+'\nend\n%MaxCore 3906\n %freq\n Anfreq True\n Quasirrho True\n Cutofffreq 100.0\n end\n '
    opt_orca_txt+='* xyz  '+str(charge)+' 1 \n'
    if os.path.isfile(os.path.join(os.getcwd(),file_name)):
        with open(file_name,"r") as f: carts=f.readlines()[2:]
        for cart in carts: 
            if cart.strip()!="": opt_orca_txt+=cart
        opt_orca_txt+="end"
        with open(inp_file_name,"w") as f: f.write(opt_orca_txt)
    else:
        print ("could not find xyz file, type coordinates in vi "+inp_file_name)
        opt_orca_txt+="end"
        with open(inp_file_name,"w") as f: f.write(opt_orca_txt) 
        os.system("vi "+inp_file_name)  
    return    inp_file_name  

def run_optimization(file_name, n_procs=14, orca_exe="/home/orca5/orca",scratch_home="/scratch/lsimon/",ofakeG_exe="/home/g16/g16/OfakeG",opt_lot="PBEh-3c",img_freq_limit=-50):

    optimization_counter=1
    #must load: module load openmpi-4.1.1-gcc-4.9.4-6bb7ahv before running it
    curr_dir=os.getcwd()+"/"
    scratch=os.path.join(scratch_home,file_name.split(".inp")[0])
    if not os.path.isdir(scratch):os.mkdir(scratch)
    os.system("cp ./"+file_name+" "+scratch)
    os.chdir(scratch)
    os.system(orca_exe+" "+file_name+" > "+curr_dir+file_name.split(".inp")[0]+".out --allow-run-as-root")
    os.system("cp "+file_name.split(".inp")[0]+".hess "+curr_dir)
    os.chdir(curr_dir)
    if ofakeG_exe!="":
        print(ofakeG_exe+" "+file_name.split(".inp")[0]+".out")#borrame
        os.system(ofakeG_exe+" "+file_name.split(".inp")[0]+".out")
    files_to_compress.append(file_name.split(".inp")[0]+".hess ")
    files_to_compress.append(file_name.split(".inp")[0]+".out ")
    if ofakeG_exe!="":
        files_to_compress.append(file_name.split(".inp")[0]+"_fake.out ")

    return file_name.split(".inp")[0]+".out"


def repeat_optimization(molecule,file_name, attempt=1, n_procs=14, orca_exe="/home/orca5/orca",scratch_home="/scratch/lsimon/",ofakeG_exe="/home/g16/g16/OfakeG",opt_lot="PBEh-3c"):

    output_text="················WARNING:·····················"
    output_text+="\nthere is at least one imaginary frequency in the optimized structure!!!"
    output_text+="\ncalculation must be repeated with different initial coordinates or discarded. Calculation of other properties will not proceed"
    output_text+="\nbut first let's analyze the relevance of the calculations that are skipped:"
    charge,compound_name=get_charge(file_name)
    similar_structure_files= [f for f in os.listdir(os.getcwd()) if f.startswith(compound_name) and f.endswith(opt_lot+".out") and "imaginary" not in f]
    if file_name in similar_structure_files: similar_structure_files.remove(file_name)
    if "imaginary" in file_name and file_name.split("imaginary")[0]+".out" in similar_structure_files: similar_structure_files.remove(file_name.split("imaginary")[0]+".out")
    print (similar_structure_files)
    not_finished_files=[]
    for f in similar_structure_files:
        with open(f,"r") as out:
            last_line=  out.readlines()[-2]
            print (f)#borrame
            print(last_line)#borrame
            succeed=("ORCA TERMINATED NORMALLY" in last_line)
        if not succeed: not_finished_files.append(f)
    similar_structure_files=[f for f in similar_structure_files if f not in not_finished_files]
    print (similar_structure_files)#borrame
    need_to_repeat=True

    if len(similar_structure_files)>0:
        similar_mols=[]
        for fm in similar_structure_files:
            print (fm)
            similar_mols.append(Molecular_structure.Molecular_structure(fm,"last"))

        #similar_mols=[Molecular_structure.Molecular_structure(s,"last") for s in similar_structure_files ]
        similar_energies=np.array([s.electronic_energy for s in similar_mols])  

        #check if it is not neccesary to repeat the calculation because the energy is too high or the structure is too similar to others
        #not very effective, so it is commented
        """
        for sm,sn,se in zip(similar_mols,similar_structure_files,similar_energies):
                energy_difference=np.abs(627.5095*(molecule.electronic_energy-se))
                if energy_difference>6.0 and molecule.electronic_energy>se:
                    output_text+="calculation will not continue because optimized structure has imaginary frequency and has a very large energy: "+str(energy_difference)
                    with open(file_name,"a") as f: f.write(output_text)
                    need_to_repeat=False
                elif energy_difference<0.5:
                    rmsd,max_dist,_,_,_,_=molecule.min_rmsd(sm,return_max_dist=True)
                    if rmsd<0.15 and max_dist<0.2:
                        output_text+="calculation will not continue because optimized structure has imaginary frequency and is very similar to other structure:"
                        output_text+="with file:"+sn+" rmsd: "+str(rmsd)+" max_dist: "+str(max_dist)+ " energy diff: "+str(energy_difference)
                        with open(file_name,"a") as f: f.write(output_text)
                        need_to_repeat=False
        """
    if need_to_repeat: #let´s repeat the optimization
        if attempt%2==0: sign=-1
        else:sign=1
        factor= sign*attempt*0.2/ ( np.max(np.sum( np.array(molecule.normal_modes[6])**2, axis=0)))
        molecule.aply_normal_mode(normal_mode_number=6,factor=factor)
        if attempt>2:
            factor= sign*(attempt-1)*0.2/ ( np.max(np.sum( np.array(molecule.normal_modes[7])**2, axis=0)))
            molecule.aply_normal_mode(normal_mode_number=7,factor=factor)
        if attempt>4:
            factor= sign*(attempt-2)*0.2/ ( np.max(np.sum( np.array(molecule.normal_modes[8])**2, axis=0)))
            molecule.aply_normal_mode(normal_mode_number=8,factor=factor)
        new_xyz_file_name=file_name.split(".out")[0]+"imaginary"+str(attempt+1)+".xyz"
        molecule.print_xyz(new_xyz_file_name)
        opt_inp_file=prepare_optimization(new_xyz_file_name,opt_lot=opt_lot,orca_version=orca_version,n_procs=n_procs,skip_optimization=False,reduce_step=True)
        opt_out_file=run_optimization(opt_inp_file,n_procs=n_procs,orca_exe=orca_exe,scratch_home=scratch_home,ofakeG_exe=ofakeG_exe,opt_lot=opt_lot)

    return opt_out_file            



#prepare sp input files
def prepare_sp_files(file_name,n_procs=14,levels_of_theory=["pbeh3c","m06","sm06","wb97xd","swb97xd"]):

    if levels_of_theory=="all": levels_of_theory=["pbeh3c","m06","sm06","wb97xd","swb97xd"]
    elif type(levels_of_theory)==str: levels_of_theory=[levels_of_theory] 

    head_commun='RIJCOSX  DEFGRID3 TightSCF NMR CPCM CHELPG Hirshfeld NBO \n  %cpcm smd true\n  SMDsolvent "water"\n  end\n  %pal nprocs '+str(n_procs)+'\n  end\n %MaxCore 3906  \n'
    head_commun+=' %elprop\n   Dipole  true \n   quadrupole true \n    polar   1 \n   end \n'

    m=Molecular_structure.Molecular_structure(file_name,"last")

    inp_sp_files=[]

    for l_o_t in levels_of_theory:
        head=""
        if   l_o_t=="pbeh3c":  head='!  PBEh3c '+ head_commun
        elif l_o_t=="m06":     head='!  m062x def2-TZVPPD '+ head_commun
        elif l_o_t=="sm06":    head='!  m062x def2-SVPD '+ head_commun
        elif l_o_t=="wb97xd":  head='!  wb97x-d3 def2-TZVPPD '+ head_commun
        elif l_o_t=="swb97xd": head='!  wb97x-d3 def2-SVPD '+ head_commun 
        elif l_o_t=="r2scan3c": head='!  r2scan-3c '+ head_commun 
        elif l_o_t=="wb97x3c":  head='!  wb97x-3c '+ head_commun
        sp_file_name=file_name.split(".out")[0]+"_"+l_o_t+"_chrg.inp"
        charge,_=get_charge(file_name)
        head+='\n%nbo\nNBOKEYLIST = "$NBO PRINT=3 NBO NPA AONBO=C CYCLES=100 FILE=\''+sp_file_name.split(".inp")[0]+' ARCHIVE  $END"\nend\n * xyz  '+str(charge)+" 1 \n"
        m.print_orca(filename=sp_file_name,header=head)
        inp_sp_files.append(sp_file_name)
    return inp_sp_files

def run_sp_files(file_name,orca_exe="/home/orca5/orca",scratch_home="/scratch/lsimon/",do_multwfn_exe="/home/scripts/do_multwfn.py",multiwfn_home="/home/multiwfn_3.8",nbo_exe="/home/nbo7/bin/nbo7.i4.exe",orca_version=5):
    #must load: module load openmpi-4.1.1-gcc-4.9.4-6bb7ahv before running it

    os.environ["NBOEXE"]=nbo_exe
    os.environ["Multiwfnpath"]=multiwfn_home
    os.environ["PATH"]=os.environ["PATH"]+multiwfn_home
    curr_dir=os.getcwd()+"/"
    print (curr_dir)
    base_name=file_name.split("_")[0]  #the name of the file must not contain "_"
    scratch=os.path.join(scratch_home,base_name.split(".inp")[0])+"/"
    os.system("echo $Multiwfnpath")
    os.system("echo $PATH")
    print("cp ./"+file_name+" "+scratch) 
    os.system("cp ./"+file_name+" "+scratch)
    os.chdir(scratch)
    os.system(orca_exe+" "+file_name+" > "+curr_dir+file_name.split(".inp")[0]+".out --allow-run-as-root")
    os.system("cp "+file_name.split(".inp")[0]+".cpcm "+curr_dir)
    if orca_version==5:
        os.system(do_multwfn_exe+" "+scratch+file_name.split(".inp")[0]+".gbw "+str(n_procs)+" 5")
    elif orca_version==6:
        os.system(do_multwfn_exe+" "+scratch+file_name.split(".inp")[0]+".gbw "+str(n_procs)+" 6")
    os.system("cp "+file_name.split(".inp")[0]+".out "+curr_dir)
    os.system("cp "+file_name.split(".inp")[0]+".multwfn.json "+curr_dir)
    #os.system("cp "+file_name.split(".inp")[0]+".gbw "+curr_dir)   #uncomment to keep these (large) files
    #os.system("cp "+file_name.split(".inp")[0]+".47 "+curr_dir)
    #os.system("cp "+file_name.split(".inp")[0]+".molden.input "+curr_dir)
    os.system("cp "+file_name.split(".inp")[0]+".cpcm "+curr_dir)
    os.chdir(curr_dir)
    files_to_compress.append(file_name.split(".inp")[0]+".out ")
    files_to_compress.append(file_name.split(".inp")[0]+".multwfn.json ")
    #files_to_compress.append(file_name.split(".inp")[0]+".47 ")
    #files_to_compress.append(file_name.split(".inp")[0]+".gbw ")
    #files_to_compress.append(file_name.split(".inp")[0]+".molden.input ")
    files_to_compress.append(file_name.split(".inp")[0]+".cpcm ")
    return file_name.split(".inp")[0]+".out"

#decompose the output file in a nbo.out and an out file
def split_nbo_section(file_name):
    with open(file_name,"r") as f: lines=f.readlines()
    start_nbo=lines.index("Now starting NBO....\n")
    if "                                *** returned from  NBO  program ***\n" in lines: 
        end_nbo=lines.index("                                *** returned from  NBO  program ***\n")
    else:
        end_nbo=0
        for l in lines:
            end_nbo+=1
            if  "NBO analysis completed" in l: end_nbo+=1;break
    nbo_text="".join(lines[start_nbo+1:end_nbo])
    not_nbo_text="".join(lines[0:start_nbo]+lines[end_nbo:])
    with open(file_name.split(".out")[0]+".nbo.out","w") as f: f.write(nbo_text)
    #os.system("cp "+file_name+" "+file_name+".bak")
    files_to_compress.append(file_name.split(".out")[0]+".nbo.out")
    with open(file_name,"w") as f: f.write(not_nbo_text)

def energy_far_water(xyz_file="temp.xyz",options="--gfn2 --gbsa h2o",n_waters=1,n_procs=14,chrg=0):
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
    command="xtb water_fery_far.xyz "+options+ " --T "+str(n_procs) +" --chrg  "+str(chrg) +" > energy_far_water.out"
    print (command)
    os.system(command)
    with open("energy_far_water.out","r") as f: out_lines=f.readlines()
    for l in out_lines:
        if l.startswith("          | TOTAL ENERGY      "): energy= float(l.split()[3])   
    with open("water_fery_far.xyz","r") as f: lines=f.readlines() 
    lines[1]=" "+str(energy)+"\n"
    with open("water_fery_far.xyz","w") as f: f.write("".join(lines)) 


def run_crest_qgc(xyz_file="temp.xyz", options="--gfn2 --gbsa h2o",n_waters=1,n_procs=14,chrg=0):
    with open(xyz_file,"r") as f: xyz_lines=f.readlines()
    old_n_atoms=int(xyz_lines[0].strip())
    with open(".xcontrol","w") as f: f.write("$constrain\n   force constant=2.5\n   atoms: 1-"+str(old_n_atoms)+"\n$end")

    with open("water.xyz","w") as f:
        f.write(" 3\n\nO         -0.1918040235        1.3862489483        0.0047370042\nH          0.7660977787        1.3911615443       -0.0141642652\nH         -0.4927337474        1.6150799341       -0.8756928250\n")
    
    os.system("rm -rf grow")
    
    command="crest "+str(xyz_file)+" --qcg water.xyz --nsolv "+str(n_waters)+" --T "+str(n_procs)+" --chrg "+str(chrg)+" --ensemble "+options+ " > crest_qgc.out"
    print (command)
    os.system(command)
    
    with open("crest_qgc.out","r") as f: lines=f.readlines()
    for l in lines: 
        if l.find("G /Eh     :")>-1: solv_energy=float( l.split("G /Eh     :")[1]   )
    
    return solv_energy

def put_water(file_name,suffix="_1wat.conformers.gfn2gbsa.xyz",options="--gfn2 --gbsa h2o",n_procs=14,n_waters=1,launch_directory="/home/lsimon/jobs/"):
    chrg,_=get_charge(file_name)

    energy_far_water(file_name,chrg=chrg,options=options,n_procs=n_procs,n_waters=n_waters)
    avg_energy=run_crest_qgc(file_name,chrg=chrg,options=options,n_procs=n_procs,n_waters=n_waters)

    with open("water_fery_far.xyz","r") as f: very_far_lines=f.readlines()
    with open("./ensemble/final_ensemble.xyz","r") as f: ensemble_lines=f.readlines()
    ensemble_lines[1]= ensemble_lines[1].strip()+" (averaged energy: "+str(avg_energy)+" )\n"
    text_for_xyz_file=ensemble_lines+very_far_lines
    if file_name[-4:]==".xyz":
        with open(launch_directory+file_name[:-4]+suffix,"w") as f: f.write("".join(text_for_xyz_file))
    else:
        with open(file_name+suffix,"w") as f: f.write("".join(text_for_xyz_file))

def water_effect(file_name,n_waters=1,n_procs=14,scratch_dir="/scratch/lsimon/",opt_lot=""):
    if opt_lot=="PBEh3c" or opt_lot=="pbeh3c": suffix="_"+str(n_waters)+"wat.conformers.gfn2_gfnff_gbsa.xyz"
    elif opt_lot=="b973c" or opt_lot=="B973c" : suffix="-b973c_"+str(n_waters)+"wat.conformers.gfn2_gfnff_gbsa.xyz"
    else: suffix="-"+opt_lot+"_"+str(n_waters)+"wat.conformers.gfn2_gfnff_gbsa.xyz"
    options="--gfn2//gfnff --gbsa h2o"
    options_with_error="--gfn2//gfnff --tstep 0.1 --shake 0 --wscal 1.0 --nofix  --gbsa h2o"
    #scratch=scratch_dir+file_name.split(".xyz")[0]+"/"
    curr_dir=os.getcwd()+"/"
    os.system("cp "+file_name+" "+scratch_dir)
    os.chdir(scratch_dir)
    put_water(file_name,n_procs=n_procs,suffix=suffix,options=options,n_waters=n_waters,launch_directory=curr_dir)
    with open("crest_qgc.out","r") as f: 
        if "Trial MTD 6 did not converge!" in f.read():
            put_water(file_name,n_procs=n_procs,suffix=suffix,options=options_with_error,n_waters=n_waters,launch_directory=curr_dir)
            #repeat this with more options to improve convergence...            
    os.chdir(curr_dir)





if __name__ == "__main__":

    #default values
    keep_files_compressed=False
    sp_lot=["pbeh3c","m06","sm06","wb97xd","swb97xd"]
    orca_exe,nbo_exe,orca_version=orca5_exe,nbo_orca5_exe,5
    skip_optimization=False
    skip_water=False
    img_freq_limit=-35
    max_attempts=2
    n_procs=28

    if len(sys.argv) > 2:
            i=1
            while i< len(sys.argv):
                if sys.argv[i] in ["-np"]:
                    i+=1
                    if i< len(sys.argv): n_procs= sys.argv[i]
                elif sys.argv[i] in ["-sp_lot"]:
                    i+=1
                    sp_lot= sys.argv[i].split(".")
                elif sys.argv[i] in ["-opt_lot"]:
                    i+=1
                    opt_lot= sys.argv[i]
                elif sys.argv[i] in ["-orca_version","-orca_v"]:
                    i+=1
                    if sys.argv[i]== "orca6" or sys.argv[i]=="6":
                        orca_exe,nbo_exe,orca_version=orca6_exe,nbo_orca6_exe,6
                    else: 
                        orca_exe,nbo_exe,orca_version=orca5_exe,nbo_orca5_exe,5
                elif sys.argv[i] in ["orca6"]:
                    orca_exe,nbo_exe,orca_version=orca6_exe,nbo_orca5_exe,5
                elif sys.argv[i] in ["-auto","-no_edit","-no_vi"]: edit_input=False
                elif sys.argv[i] in ["-only_sp","-skip_optimization"]: skip_optimization=True
                elif sys.argv[i] in ["-skip_water"]: skip_water=True
                elif sys.argv[i] in ["-img_limit"]:
                    i+=1
                    img_freq_limit= float(sys.argv[i])
                i+=1


    xyz_file=sys.argv[1]
    print ("starting...")
    os.system("cp "+xyz_file+" "+xyz_file+".original")
    correct_xyz_file(xyz_file)
    print("rm -f "+xyz_file.split(".xyz")[0]+"*imaginary*.*")
    os.system("rm -f "+xyz_file.split(".xyz")[0]+"*imaginary*.*")
    opt_inp_file=prepare_optimization(xyz_file,opt_lot=opt_lot,orca_version=orca_version,n_procs=n_procs,skip_optimization=skip_optimization)
    if skip_optimization==False:
        opt_out_file=run_optimization(opt_inp_file,n_procs=n_procs,orca_exe=orca_exe,scratch_home=scratch_home,ofakeG_exe=ofakeG_exe,opt_lot=opt_lot)
        n_attempts=0
        no_img_freq_str_found=False
        while n_attempts<max_attempts:
            n_attempts+=1
            m=Molecular_structure.Molecular_structure(opt_out_file,"last")
            with open(opt_out_file,"r") as f: last_line=f.readlines()[-1]
            if not any( [freq<img_freq_limit for freq in m.QM_output.frequencies[0]] ) and "aborting the run" not in last_line: 
                if "imaginary" in opt_out_file: 
                    final_name=opt_out_file.split("imaginary")[0]+".out"
                    os.system("mv "+opt_out_file+" "+final_name)
                    os.system("mv "+opt_out_file.split(".out")[0]+".hess "+final_name.split(".out")[0]+".hess ")
                    os.system("mv "+opt_out_file.split(".out")[0]+"_fake.out "+final_name.split(".out")[0]+"_fake.out ")
                    opt_out_file=final_name
                no_img_freq_str_found=True
                break
            else:
                if n_attempts==1: os.system("mv "+opt_out_file+" "+opt_out_file.split('.out')[0]+"imaginary"+str(n_attempts)+".out")
                opt_out_file=repeat_optimization(m,opt_out_file, attempt=n_attempts, n_procs=n_procs, orca_exe=orca_exe,scratch_home=scratch_home,ofakeG_exe=ofakeG_exe,opt_lot=opt_lot)

        if not no_img_freq_str_found: #if there are several calculations with imaginary frequencies
            molecules_names_list=[f for f in os.listdir(os.getcwd()) if f.startswith(xyz_file.split('.xyz')[0]) and "imaginary" in f and f.endswith("out") and not f.endswith("fake.out") and not f.endswith("nbo.out") ]
            print(molecules_names_list)
            molecules=[Molecular_structure.Molecular_structure(m_file,"last") for m_file in molecules_names_list]
            freqs=[m.QM_output.frequencies[0][6] for m in molecules]
            print (freqs)
            best_molecule_name=molecules_names_list[freqs.index(max(freqs))]
            print ("mv "+best_molecule_name+" "+opt_inp_file.split('.inp')[0]+".out")
            os.system("mv "+best_molecule_name+" "+opt_inp_file.split('.inp')[0]+".out")
            opt_out_file=opt_inp_file.split('.inp')[0]+".out"
    else:
        curr_dir=os.getcwd()+"/"
        scratch=os.path.join(scratch_home,opt_inp_file.split(".inp")[0])
        if not os.path.isdir(scratch):os.mkdir(scratch)
        os.system("cp -f ./"+opt_inp_file+" "+scratch)
        os.system("cp -f ./"+xyz_file+" "+scratch)
        opt_out_file=opt_inp_file.split(".inp")[0]+".out"

    m=Molecular_structure.Molecular_structure(opt_out_file,"last")
    m.print_xyz(xyz_file)
    inp_sp_files=prepare_sp_files(opt_out_file,n_procs=n_procs,levels_of_theory=sp_lot)
    for inp_sp_file in inp_sp_files:
        out_sp_file=run_sp_files(inp_sp_file,orca_exe=orca_exe,scratch_home=scratch_home,do_multwfn_exe=do_multwfn_exe,multiwfn_home=multiwfn_home,nbo_exe=nbo_exe,orca_version=orca_version)
        split_nbo_section(out_sp_file)
    if not skip_water:
        water_effect(xyz_file,n_waters=1,n_procs=n_procs,scratch_dir=scratch_home+opt_inp_file.split(".inp")[0],opt_lot=opt_lot)
        #water_effect(xyz_file,n_waters=2,n_procs=n_procs,scratch_dir=scratch_home)
        #water_effect(xyz_file,n_waters=3,n_procs=n_procs,scratch_dir=scratch_home)
        #water_effect(xyz_file,n_waters=4,n_procs=n_procs,scratch_dir=scratch_home)
        files_to_compress.append(opt_inp_file.split(".inp")[0]+"_1wat.conformers.gfn2_gfnff_gbsa.xyz")
        #files_to_compress.append(xyz_file.split(".xyz")[0]+"_2wat.conformers.gfn2_gfnff_gbsa.xyz")
        #files_to_compress.append(xyz_file.split(".xyz")[0]+"_3wat.conformers.gfn2_gfnff_gbsa.xyz")
        #files_to_compress.append(xyz_file.split(".xyz")[0]+"_4wat.conformers.gfn2_gfnff_gbsa.xyz")

    if keep_files_compressed:
        print (files_to_compress)
        import tarfile 
        with tarfile.open(xyz_file.split(".xyz")[0]+".tar","w") as f:
            for file in files_to_compress: f.add(file.strip()) 
    
    




