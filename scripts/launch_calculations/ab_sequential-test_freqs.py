#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
#script para calcular con todos los niveles de teoría los archivos para un nuevo ácido
#es llamado por genxyz.py
#da un error al calcular con CREST varias moléculas de h2o (sólo funciona bien para la primera) por gfn2/gfnff 

import string
import os
import os.path
import sys
sys.path.append('../import')
import subprocess
import Molecular_structure
import numpy as np

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



#prepare file for optimization
def prepare_optimization(file_name,n_procs=14):
    print("using modified version")#borrame
    charge,compound_name=get_charge(file_name)
    if file_name[-4:]!=".xyz": file_name+=".xyz" 
    inp_file_name=file_name[:-4]+".inp"

    opt_orca_txt='! PBEh-3c CPCM defgrid1 CHELPG opt Hirshfeld \n  %cpcm smd true\n  SMDsolvent "water"  end\n  \n%geom\n   MaxIter 500\n   \n   MaxStep 0.3  \n end\n'
    opt_orca_txt+='%pal nprocs '+str(n_procs)+'\nend\n%MaxCore 3906\n %freq\n Numfreq True\n Quasirrho True\n Cutofffreq 100.0\n end\n '
    opt_orca_txt+='* xyz  '+str(charge)+' 1 \n'
    if os.path.isfile(os.path.join(os.getcwd(),file_name)):
        with open(file_name,"r") as f: carts=f.readlines()[2:]
        for cart in carts: 
            if cart.strip()!="": opt_orca_txt+=cart
        opt_orca_txt+="end"
        with open(inp_file_name,"w") as f: f.write(opt_orca_txt)
    else:
        opt_orca_txt+="end"
        with open(inp_file_name,"w") as f: f.write(opt_orca_txt) 
        os.system("vi "+inp_file_name)  
    return    inp_file_name  


def run_optimization(file_name, n_procs=14, orca_exe="/home/orca5/orca",scratch_home="/scratch/lsimon/",old_files_folder="/home/lsimon/jobs/pka/output_files/optimization/"):
    charge,compound_name=get_charge(file_name)
    #must load: module load openmpi-4.1.1-gcc-4.9.4-6bb7ahv before running it
    curr_dir=os.getcwd()+"/"
    scratch=os.path.join(scratch_home,file_name.split(".inp")[0])
    if not os.path.isdir(scratch):os.mkdir(scratch)
    os.system("cp ./"+file_name+" "+scratch)
    os.chdir(scratch)
    os.system(orca_exe+" "+file_name+" > "+curr_dir+file_name.split(".inp")[0]+".out --allow-run-as-root")
    os.system("cp "+file_name.split(".inp")[0]+".hess "+curr_dir)
    os.chdir(curr_dir)
    os.system("/home/g16/g16/OfakeG "+file_name.split(".inp")[0]+".out")
    files_to_compress.append(file_name.split(".inp")[0]+".hess ")
    files_to_compress.append(file_name.split(".inp")[0]+".out ")
    files_to_compress.append(file_name.split(".inp")[0]+"_fake.out ")

    m=Molecular_structure.Molecular_structure(file_name.split(".inp")[0]+".out","last")
    imaginary_freqs=False

    #check if the structure is very similar to any other already calculated structures
    similar_structure_files= [f for f in os.listdir(old_files_folder) if f.startswith(compound_name) and f.endswith(".out")]
    if file_name.split(".inp")[0]+".out" in similar_structure_files: similar_structure_files.remove(file_name.split(".inp")[0]+".out")
    if len(similar_structure_files)>0:
        similar_mols=[Molecular_structure.Molecular_structure(old_files_folder+s,"last") for s in similar_structure_files ]
        similar_energies=np.array([s.electronic_energy for s in similar_mols])
        for sm,sn,se in zip(similar_mols,similar_structure_files,similar_energies):
            energy_difference=np.abs(627.5095*(m.electronic_energy-se))
            rmsd,max_dist,_,_,_,_=m.min_rmsd(sm,return_max_dist=True)
            if rmsd<0.15 and max_dist<0.2 and energy_difference<0.3:
                    output_text+="with file:"+similar_structure_files+" rmsd: "+str(rmsd)+" max_dist: "+str(max_dist)+ " energy diff: "+str(energy_difference)
                    output_text+="the structures are so similar that I will save time not calculating anything else\n"
                    output_text+="--CONSIDER TO DELETE FILE: "+str(file_name.split(".inp"))+" with command:\n"
                    output_text+="/home/lsimon/jobs/pka/scripts/launch_calculations/borrar_ab.py "+str(file_name.split(".inp"))
                    with open(file_name.split(".inp")[0]+".img_freqs","w") as f: f.write(output_text)        
                    sys.exit()                    

    #check if there are imaginary frequencies
    if any( [freq<-40 for freq in m.QM_output.frequencies[0]] ):
        imaginary_freqs=True
        output_text="················WARNING:·····················"
        output_text+="\nthere is at least one imaginary frequency in the optimized structure!!!"
        output_text+="\ncalculation must be repeated with different initial coordinates or discarded. Calculation of other properties will not proceed"
        output_text+="\nbut first let's analyze the relevance of the calculations that are skipped:"
        if len(similar_structure_files)==0:
            output_text+="\nthe structure seems relevant since there are no other analogue structures already calculated!"
        else:    
            output_text+="\nthere are already other analogue structures:  "+str(similar_structure_files)
            similar_mols=[Molecular_structure.Molecular_structure(old_files_folder+s,"last") for s in similar_structure_files ]
            similar_energies=np.array([s.electronic_energy for s in similar_mols])
            output_text+="\nsimilar energyes "+str(similar_energies)+"min:"+str(np.min(similar_energies))#borrame
            delta_energy=627.5095*(m.electronic_energy-np.min(similar_energies))
            output_text+="\ndelta energy"+str(delta_energy)
            if delta_energy>0: output_text+="\nthe energy of the structure is: "+str(delta_energy)+" kcal/mol higher than the most stable analogue structure"
            else: output_text+="\nthe structure was the most stable by "+ str(delta_energy) +" kcal/mol"
            for sm,sn in zip(similar_mols,similar_structure_files):
                if rmsd<0.5 and max_dist<0.5:
                    output_text+="\nthe structure is very similar to:" +sn
                    output_text+="rmsd: "+str(rmsd)+" max_dist: "+str(max_dist)
                if rmsd<0.15 and max_dist<0.2:
                    output_text+="the structures are so similar that I will save time not calculating anything else"
        
    else: output_text="it seems that there is no imaginary frequency, so calculations will continue"

    if imaginary_freqs:
        with open(file_name.split(".inp")[0]+".img_freqs","w") as f: f.write(output_text)        
        sys.exit()

    return file_name.split(".inp")[0]+".out"


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
        sp_file_name=file_name.split(".out")[0]+"_"+l_o_t+"_chrg.inp"
        charge,compound_name=get_charge(file_name)
        head+='\n%nbo\nNBOKEYLIST = "$NBO PRINT=3 NBO NPA AONBO=C  FILE=\''+sp_file_name.split(".inp")[0]+' ARCHIVE  $END"\nend\n * xyz  '+str(charge)+" 1 \n"
        m.print_orca(filename=sp_file_name,header=head)
        inp_sp_files.append(sp_file_name)
    return inp_sp_files

def run_sp_files(file_name,orca_exe="/home/orca5/orca",scratch_home="/scratch/lsimon/"):
    #must load: module load openmpi-4.1.1-gcc-4.9.4-6bb7ahv before running it

    os.environ["NBOEXE"]="/home/nbo7/bin/nbo7.i4.exe"
    os.environ["Multiwfnpath"]="/home/multiwfn_3.8"
    os.environ["PATH"]=os.environ["PATH"]+"/home/multiwfn_3.8"
    curr_dir=os.getcwd()+"/"
    print (curr_dir)
    base_name=file_name.split("_")[0]  #the name of the file must not contain "_"
    scratch=os.path.join(scratch_home,base_name.split("inp")[0])+"/"
    os.system("echo $Multiwfnpath")
    os.system("echo $PATH")
    print("cp ./"+file_name+" "+scratch) 
    os.system("cp ./"+file_name+" "+scratch)
    os.chdir(scratch)
    os.system(orca_exe+" "+file_name+" > "+curr_dir+file_name.split(".inp")[0]+".out --allow-run-as-root")
    os.system("cp "+file_name.split(".inp")[0]+".cpcm "+curr_dir)
    os.system("do_multwfn.py "+scratch+file_name.split(".inp")[0]+".gbw 14")
    os.system("cp "+file_name.split(".inp")[0]+".out "+curr_dir)
    os.system("cp "+file_name.split(".inp")[0]+".multwfn.json "+curr_dir)
    os.system("cp "+file_name.split(".inp")[0]+".gbw "+curr_dir)
    os.system("cp "+file_name.split(".inp")[0]+".47 "+curr_dir)
    os.system("cp "+file_name.split(".inp")[0]+".molden.input "+curr_dir)
    os.system("cp "+file_name.split(".inp")[0]+".cpcm "+curr_dir)
    os.chdir(curr_dir)
    files_to_compress.append(file_name.split(".inp")[0]+".out ")
    files_to_compress.append(file_name.split(".inp")[0]+".multwfn.json ")
    files_to_compress.append(file_name.split(".inp")[0]+".47 ")
    files_to_compress.append(file_name.split(".inp")[0]+".gbw ")
    files_to_compress.append(file_name.split(".inp")[0]+".molden.input ")
    files_to_compress.append(file_name.split(".inp")[0]+".cpcm ")
    return file_name.split(".inp")[0]+".out"




#decompose the output file in a nbo.out and an out file
def split_nbo_section(file_name):
    with open(file_name,"r") as f: lines=f.readlines()
    start_nbo=lines.index("Now starting NBO....\n")
    end_nbo=lines.index("                                *** returned from  NBO  program ***\n")
    nbo_text="".join(lines[start_nbo+1:end_nbo])
    not_nbo_text="".join(lines[0:start_nbo]+lines[end_nbo:])
    with open(file_name.split(".out")[0]+".nbo.out","w") as f: f.write(nbo_text)
    #os.system("cp "+file_name+" "+file_name+".bak")
    files_to_compress.append(file_name.split(".out")[0]+".nbo.out")
    with open(file_name,"w") as f: f.write(not_nbo_text)


def energy_far_water(xyz_file="temp.xyz",options="--gfn2 --gbsa h2o",n_waters=1,nprocs=14,chrg=0):
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


def run_crest_qgc(xyz_file="temp.xyz", options="--gfn2 --gbsa h2o",n_waters=1,nprocs=14,chrg=0):
    with open(xyz_file,"r") as f: xyz_lines=f.readlines()
    old_n_atoms=int(xyz_lines[0].strip())
    with open(".xcontrol","w") as f: f.write("$constrain\n   force constant=2.5\n   atoms: 1-"+str(old_n_atoms)+"\n$end")

    with open("water.xyz","w") as f:
        f.write(" 3\n\nO         -0.1918040235        1.3862489483        0.0047370042\nH          0.7660977787        1.3911615443       -0.0141642652\nH         -0.4927337474        1.6150799341       -0.8756928250\n")
    
    os.system("rm -rf grow")
    
    command="crest "+str(xyz_file)+" --qcg water.xyz --nsolv "+str(n_waters)+" --T "+str(nprocs)+" --chrg "+str(chrg)+" --ensemble "+options+ " > crest_qgc.out"
    print (command)
    os.system(command)
    
    with open("crest_qgc.out","r") as f: lines=f.readlines()
    for l in lines: 
        if l.find("G /Eh     :")>-1: solv_energy=float( l.split("G /Eh     :")[1]   )
    
    return solv_energy

def put_water(file_name,suffix="_1wat.conformers.gfn2gbsa.xyz",options="--gfn2 --gbsa h2o",nprocs=14,n_waters=1,launch_directory="/home/lsimon/jobs/"):
    chrg,compound_name=get_charge(file_name)

    energy_far_water(file_name,chrg=chrg,options=options,nprocs=nprocs,n_waters=n_waters)
    avg_energy=run_crest_qgc(file_name,chrg=chrg,options=options,nprocs=nprocs,n_waters=n_waters)

    with open("water_fery_far.xyz","r") as f: very_far_lines=f.readlines()
    with open("./ensemble/final_ensemble.xyz","r") as f: ensemble_lines=f.readlines()
    ensemble_lines[1]= ensemble_lines[1].strip()+" (averaged energy: "+str(avg_energy)+" )\n"
    text_for_xyz_file=ensemble_lines+very_far_lines
    if file_name[-4:]==".xyz":
        with open(launch_directory+file_name[:-4]+suffix,"w") as f: f.write("".join(text_for_xyz_file))
    else:
        with open(file_name+suffix,"w") as f: f.write("".join(text_for_xyz_file))

def water_effect(file_name,n_waters=1,n_procs=14,scratch_dir="/scratch/lsimon/"):
    suffix="wat.conformers.gfn2_gfnff_gbsa.xyz"
    options="--gfn2//gfnff --gbsa h2o"
    options_with_error="--gfn2//gfnff --tstep 0.1 --shake 0 --wscal 1.0 --nofix  --gbsa h2o"
    scratch=scratch_dir+file_name.split(".xyz")[0]+"/"
    curr_dir=os.getcwd()+"/"
    os.system("cp "+file_name+" "+scratch)
    os.chdir(scratch)
    put_water(file_name,nprocs=n_procs,suffix="_"+str(n_waters)+suffix,options=options,n_waters=n_waters,launch_directory=curr_dir)
    with open("crest_qgc.out","r") as f: 
        if "Trial MTD 6 did not converge!" in f.read():
            put_water(file_name,nprocs=n_procs,suffix="_"+str(n_waters)+suffix,options=options_with_error,n_waters=n_waters,launch_directory=curr_dir)
            #repeat this with more options to improve convergence...            
    os.chdir(curr_dir)


                                                                            
                                                                             



if __name__ == "__main__":
    orca_exe="/home/orca5/orca"
    scratch_home="/scratch/lsimon/"
    if len (sys.argv)>2: n_procs=int(sys.argv[2])
    else: n_procs=28
    xyz_file=sys.argv[1]
    opt_inp_file=prepare_optimization(xyz_file)
    opt_out_file=run_optimization(opt_inp_file,n_procs=n_procs,orca_exe=orca_exe,scratch_home=scratch_home)
    m=Molecular_structure.Molecular_structure(opt_out_file,"last")
    os.system("cp "+xyz_file+" "+xyz_file+".original")
    m.print_xyz(xyz_file)
    
    inp_sp_files=prepare_sp_files(opt_out_file,n_procs=n_procs,levels_of_theory=["pbeh3c","m06","sm06","wb97xd","swb97xd"])
    for inp_sp_file in inp_sp_files:
        out_sp_file=run_sp_files(inp_sp_file,orca_exe=orca_exe,scratch_home=scratch_home)
        split_nbo_section(out_sp_file)
    water_effect(xyz_file,n_waters=1,n_procs=n_procs,scratch_dir=scratch_home)
    water_effect(xyz_file,n_waters=2,n_procs=n_procs,scratch_dir=scratch_home)
    water_effect(xyz_file,n_waters=3,n_procs=n_procs,scratch_dir=scratch_home)
    water_effect(xyz_file,n_waters=4,n_procs=n_procs,scratch_dir=scratch_home)
    files_to_compress.append(xyz_file.split(".xyz")[0]+"_1wat.conformers.gfn2_gfnff_gbsa.xyz")
    files_to_compress.append(xyz_file.split(".xyz")[0]+"_2wat.conformers.gfn2_gfnff_gbsa.xyz")
    files_to_compress.append(xyz_file.split(".xyz")[0]+"_3wat.conformers.gfn2_gfnff_gbsa.xyz")
    files_to_compress.append(xyz_file.split(".xyz")[0]+"_4wat.conformers.gfn2_gfnff_gbsa.xyz")

    print (files_to_compress)
    import tarfile 
    with tarfile.open(xyz_file.split(".xyz")[0]+".tar","w") as f:
        for file in files_to_compress: f.add(file.strip()) 
    






