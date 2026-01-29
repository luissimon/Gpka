#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# script to search for structures with very high energy or structures that matches other structure
# requires molecular_structure methods to work. Generates "reports" with the structures that it suggest to delete
# and files with the commands to delete them  

import string
import os
import os.path
import sys
sys.path.append('../import')
import pandas as pd
import numpy as np
import Molecular_structure


from routes import output_files_route
from routes import sampl_output_files_route
from routes import extracted_data_route
from routes import sampl_extracted_data_route
from routes import labels_csv_file_name
from routes import sampl_labels_csv_file_name

route=output_files_route+"PBEh3c_optimized/optimization" 
route=output_files_route+"b973c_optimized/optimization" 
route=sampl_output_files_route+"PBEh3c_optimized/optimization" 
route=sampl_output_files_route+"b973c_optimized/optimization"



labels_route= sampl_extracted_data_route
labels_file="sampl_all_pkas.csv"

#the csv file containing the experimental pka values
labels=pd.read_csv(labels_route+labels_file,encoding='unicode_escape')
charges=["neut","cation","2cation","3cation","4cation","5cation","an","2an","3an","4an","5an"]

all_files=[f.split(".hess")[0] for f in os.listdir(route) if f.endswith(".hess")]
print (len(all_files))
already_done=[]
counter=0
large_energy_difference_txt=""
borrar_large_energy_difference_txt= ""

repeated_structures=[]
doubtful_repeated_structures=[]
repeated_structures_text=""
doubtful_repeated_structures_text=""


for entry in labels["compn"][0:]:  #change this to prevent analyzing all entries

    counter+=1
    compn=entry.split("_")[0]
    print(compn)
    for  charge in charges:
        file_names=[f for f in all_files if    f.startswith( compn+"-"+charge )   ]
        if len(file_names)>1:
            molecules_optz=[Molecular_structure.Molecular_structure(route+f+".out"  ,"last") for f in file_names]
            energies=[m_s.gibbs_free_energy for m_s in molecules_optz]
            #print (energies)
            #print (file_names)
            file_names=[x for _,x in sorted(zip(energies,file_names))]
            #print (energies)
            #print (file_names)
            #print (len(molecules_optz))

            molecules_optz=[x for _,_,x in sorted(zip(energies,range(len(energies)),molecules_optz))]
            
            energies=sorted(energies)
            #check energies
            for f,e in zip(file_names,energies):
                if 627.5095*(float(e)-float(energies[0]))>6.0: 
                    print("energy diff of "+f+" is large:"+str ( 627.5095*(float(e)-float(energies[0]))  )   )
                    large_energy_difference_txt+="energy diff of "+f+" is large:"+str ( 627.5095*(float(e)-float(energies[0]))  )+"; most stable structure is: "+file_names[0]+"\n"
                    borrar_large_energy_difference_txt+="./borrar_ab.py "+f+" \n"
            
            #check rmsds
            for i in range(0,len(molecules_optz)):
                m=molecules_optz[i]
                for j in range (i+1,len(molecules_optz)   ): 
                    m2=molecules_optz[j]
                    if file_names[j] not in repeated_structures+doubtful_repeated_structures:
                        rmsd,max_dist,_,_,_,_=m.min_rmsd(m2,return_max_dist=True)
                        reflx_rmsd,reflx_max_dist,_,_,_,_=m.min_rmsd(m2,return_max_dist=True,reflex=True)
                        if rmsd<reflx_rmsd: use_reflexed=""
                        else: rmsd=reflx_rmsd; max_dist<reflx_max_dist; use_reflexed=" (reflexing)"

                        if rmsd<0.15 and max_dist<0.2: 
                            repeated_structures.append(file_names[j])
                            s="\nalmost certain structures repeated: "+file_names[i] +"("+str(m.QM_output.frequencies[0][6]) +"cm-1) and "+file_names[j]+"("+str(m2.QM_output.frequencies[0][6]) +"cm-1) rmsd:"+str(rmsd)+"max_dist"+str(max_dist) +use_reflexed 
                            repeated_structures_text+=s; print (s)
                        elif rmsd<0.25 and max_dist<0.45:
                            doubtful_repeated_structures.append(file_names[j])
                            energy_diff_i=627.5095*(float(energies[i])-float(energies[0]))
                            energy_diff_j=627.5095*(float(energies[j])-float(energies[0]))
                            s="\npossible structures repeated: "+ file_names[i] +" ("+str(energy_diff_i)+","+str(m.QM_output.frequencies[0][6]) +"cm-1)" +" and "+file_names[j]+" ("+str(energy_diff_j)+","+str(m2.QM_output.frequencies[0][6]) +"cm-1)"  +  " rmsd:"+str(rmsd)+" max_dist:"+str(max_dist) +use_reflexed 
                            doubtful_repeated_structures_text+=s; print (s)


            


    with open("large_energy.txt","w") as f: f.write(borrar_large_energy_difference_txt )      
    with open("large_energy_justification.txt","w") as f: f.write(large_energy_difference_txt )
    borrar_repeated_structures_text=""
    for rs in repeated_structures:  
        borrar_repeated_structures_text+="./borrar_ab.py "+rs+" \n"
    with open("repeated_structures.txt","w") as f: f.write(borrar_repeated_structures_text)      
    with open("repeated_structures_justification.txt","w")   as f: f.write(repeated_structures_text)   
    borrar_doubtful_repeated_structures_text=""      
    for rs in doubtful_repeated_structures:  
        borrar_doubtful_repeated_structures_text+="./borrar_ab.py "+rs+" \n"
    with open("possible_repeated_structures.txt","w") as f: f.write(borrar_doubtful_repeated_structures_text)      
    with open("possible_repeated_structures_justification.txt","w")   as f: f.write(doubtful_repeated_structures_text) 
    


