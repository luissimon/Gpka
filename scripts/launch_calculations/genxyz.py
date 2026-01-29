#! /usr/bin/env python
# -*- coding: utf-8 -*-
# script to launch calculations: ./genxyz.py name_of_compound-neut (or -cation, -2cation, -an, -2an...) 
# it calls system vi to allow pasting the geometry and generates orca inputs. Charge is set according to -neut (0), -an (-1), -cation (+1), etc.
# the calculations are launched on the queeue, calling ab_sequential.py script.


import sys
import os
import string
import getpass
import subprocess

import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
from routes import mpi_module

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
edit_input=True
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
            elif sys.argv[i] in ["-auto","-no_edit","-no_vi"]: edit_input=False

            i+=1


if filename[-4:]==".xyz": filename=filename.split(".xyz")[0]
if edit_input: os.system("vi "+filename+".xyz")
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
sbatch_text+= "\nmodule load "+mpi_module+" \n"
sbatch_text+="\n./ab_sequential.py "+filename+".xyz"

with open(filename+".qsub","w") as f: f.write(sbatch_text)

os.system("sbatch "+filename+".qsub")


