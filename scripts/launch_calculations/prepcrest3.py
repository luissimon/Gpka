#! /usr/bin/env python
# -*- coding: utf-8 -*-
# script to run crest on a xyz file to generate conformations. After runing crest, genxyz is called on the resulting conformations.

import string
import os
import os.path
import sys
import subprocess



def prep_xyz_from_xyz(mxyz_file,head,filename,first=0):
    with open(mxyz_file,"r") as f: xyz_lines=f.readlines()
    xyzs=[]
    i=0
    text=""
    files=[]
    while i < len(xyz_lines):
        if xyz_lines[i].strip().isdigit() or i+1==len(xyz_lines): 
            xyzs.append(text)
            text=""
            i+=1
        i+=1
        if i < len(xyz_lines) and not xyz_lines[i].strip().isdigit(): text+=xyz_lines[i]
    i=first
    for xyz in xyzs[1:]:
        i+=1
        text=str(xyzs[0])+"\n"+xyz
        files.append(filename+str(i)+".xyz")
        with open(files[-1],"w") as f: f.write(text)
    return files

def get_charge(xyz_file_name):
    if "-6an" in xyz_file_name: return "-6"
    elif "-5an" in xyz_file_name: return "-5"
    elif "-4an" in xyz_file_name: return "-4"
    elif "-3an" in xyz_file_name: return "-3"
    elif "-2an" in xyz_file_name: return "-2"
    elif "-an" in xyz_file_name: return "-1"
    elif "-neut" in xyz_file_name: return "0"
    elif "-2cation" in xyz_file_name: return "2"
    elif "-cation" in xyz_file_name: return "1"
    elif "-3cation" in xyz_file_name: return "3"
def complete_xyz(xyz_file_name):
    with open(xyz_file_name,"r") as f: xyz_lines=f.readlines()
    not_empty_lines=[l for l in xyz_lines if l.strip()!=""]
    text=str(len(not_empty_lines))+"\n\n"
    for l in not_empty_lines: text+=l
    with open(xyz_file_name,"w") as f: f.write(text)

def get_initial_point(xyz_file_name):
    if "-neut2" in xyz_file_name: return 1000
    elif "-cation2" in xyz_file_name: return 1000
    elif "-an2" in xyz_file_name: return 1000
    if "-neut3" in xyz_file_name: return 2000
    elif "-cation3" in xyz_file_name: return 2000
    elif "-an3" in xyz_file_name: return 2000
    if "-neut4" in xyz_file_name: return 3000
    elif "-cation4" in xyz_file_name: return 3000
    elif "-an4" in xyz_file_name: return 3000
    if "-neut5" in xyz_file_name: return 4000
    elif "-cation5" in xyz_file_name: return 4000
    elif "-an5" in xyz_file_name: return 4000    
    if "-neut6" in xyz_file_name: return 5000
    elif "-cation6" in xyz_file_name: return 5000
    elif "-an6" in xyz_file_name: return 5000
    if "-neut7" in xyz_file_name: return 6000
    elif "-cation7" in xyz_file_name: return 6000
    elif "-anion7" in xyz_file_name: return 6000


    else: return 0

xyz_file_name= sys.argv[1]

os.system("rm -f crest_conformers.xyz")
if xyz_file_name.endswith("xyz"): xyz_file_name=xyz_file_name.split(".xyz")[0]

with open(xyz_file_name+".xyz","r") as f: l=f.readlines()
if len(l[0].split())>3: complete_xyz(xyz_file_name+".xyz")

charge= get_charge(xyz_file_name)
command="crest "+xyz_file_name+".xyz --zs --hflip --maxflip 10000  --cbonds 0.1 --fc 1  --chrg "+str(charge)+" --gfn2//gfnff -g h2o -T 28 --scratch "+xyz_file_name+"_crest "
#command="crest "+xyz_file_name+".xyz --zs --hflip --maxflip 10000  --cheavy 0.1 --fc 1  --chrg "+str(charge)+" --gfn2 -g h2o -T 28 --scratch "+xyz_file_name+"_crest "
command="crest "+xyz_file_name+".xyz --zs --hflip --maxflip 10000  --cbonds 0.1 --fc 1  --chrg "+str(charge)+" --gfn2 -g h2o -T 28 --scratch "+xyz_file_name+"_crest "
#command="crest "+xyz_file_name+".xyz --zs --hflip --maxflip 10000  --cheavy 0.1 --fc 1 --nocross --chrg "+str(charge)+" -gfn2 -g h2o -T 28 --scratch "+xyz_file_name+"_crest "
if "-tautomerize" in sys.argv: command+=" --tautomerize "
print (command) 
os.system(command)
#sys.exit()
head='! PBEh-3c CPCM defgrid1 CHELPG opt Hirshfeld \n\n  %cpcm smd true\n  SMDsolvent "water"\n  end\n  %pal nprocs 14\n  end\n %MaxCore 3906  \n'
head+='%freq \n Numfreq True \n Quasirrho True \n Cutofffreq 100.0 \n end \n \n %geom \n   MaxIter 500 \n   end \n * xyz   '+charge+' 1 \n'
if len(sys.argv)>2 and sys.argv[2].isdigit(): first=int(sys.argv[2])
else: first=get_initial_point(xyz_file_name)
if "-tautomerize" not in sys.argv: files= prep_xyz_from_xyz("crest_conformers.xyz",head,xyz_file_name,first) 
elif "-tautomerize" in sys.argv: files= prep_xyz_from_xyz("./"+xyz_file_name+"_crest/tautomers.xyz",head,xyz_file_name,first)
for f in files:
    command2="./genxyz.py "+f  +" -auto " #+" -qp 1  " 
    print (command2)
    os.system(command2)





