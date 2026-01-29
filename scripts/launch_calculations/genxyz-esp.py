#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import string
import getpass
import subprocess

def get_queue_priorities():
    # CHANGE 'lsimon' TO YOUR USER NAME IN COMET
    user_name = getpass.getuser()

    queue_info=subprocess.check_output(['squeue', '-u', user_name, '-o'," '%.8i %.52j %.2t %.10M %.6C %20R %Q' "])
    job_lists=queue_info.split("\n")
    priorities=[]
    j=1
    while j+1<len(job_lists):
        priorities.append(int(job_lists[j].split()[7].strip("'")))    
        j=j+1    
    return priorities


filename=sys.argv[1]
nprocs=28
node=""
queue_position=0
if len(sys.argv) > 2:
        i=1
        while i< len(sys.argv):
            if sys.argv[i] in ["-np"]:
                i+=1
                if i< len(sys.argv): nprocs= sys.argv[i]
            elif sys.argv[i] in ["-node","-n"]:
                i+=1
                if i< len(sys.argv): node= sys.argv[i]
            elif sys.argv[i] in ["-qp"]:
                i+=1
                if i< len(sys.argv): queue_position= int(sys.argv[i])
            i+=1
        

if filename[-4:]==".xyz": filename=filename.split(".xyz")[0]
os.system("vi "+filename+".xyz")
with open(filename+".xyz","r") as f: lines=f.readlines()
n_atoms=len(lines)
if not (lines[0].strip().isdigit()):
    text=str(n_atoms)+"\n\n"+"".join(lines)
    with open(filename+".xyz","w") as f: f.write(text)

sbatch_text="#!/bin/bash\n#SBATCH -t 99:00:00\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node="+str(nprocs)
sbatch_text+="\n#SBATCH --job-name="+filename+"\n#SBATCH --output="+filename+".qlog\n#SBATCH --error="+filename+".err\n"

if queue_position!=0:
    priorities=sorted(get_queue_priorities(),reverse=True)
    if queue_position<len(priorities): priority=str(priorities[queue_position-1]+1)
    else: priority=str(priorities[-1]-1)
    sbatch_text+="\n#SBATCH --priority="+priority

if node!="":
    sbatch_text+='\n#SBATCH --constraint="'+node+'" \n'
sbatch_text+= "\nmodule load openmpi-4.1.1-gcc-4.9.4-6bb7ahv\n"
sbatch_text+="\n./ab_sequential_wat.py "+filename+".xyz"

with open(filename+".qsub","w") as f: f.write(sbatch_text)

os.system("sbatch "+filename+".qsub")


