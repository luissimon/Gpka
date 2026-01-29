#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
#script for moving files from their corresponding directories to a "removed" directory 

import string
import os
import os.path
import sys
imports_path="/home/lsimon/jobs/pka/Gpka/scripts/import"
sys.path.insert(0,imports_path)
from routes import output_files_route
from routes import sampl_output_files_route

route=output_files_route+"PBEh3c_optimized/" 
route=output_files_route+"b973c_optimized/"
#route=sampl_output_files_route+"PBEh3c_optimized/"
#route=sampl_output_files_route+"b973c_optimized/"
print (route)

test=True
if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]!="": 
        if "*" not in sys.argv[1]: file_names=[sys.argv[1]]
            
        else: file_names=[f.split(".hess")[0] for f in os.listdir(route+"optimization") if f.startswith(sys.argv[1].split("*")[0]) and f.endswith(".hess")]

    else: print("gimme names!"); sys.exit()
    if len(sys.argv)>2 and "test" in sys.argv[2]: test=True
    else: test=False
    
    commands=[]
    for file_name in file_names:
        commands.append("mv "+route+"fake_out/"+file_name+"_fake.out "+route+"deleted/")
        commands.append("mv "+route+"optimization/"+file_name+".out "+route+"deleted/")
        commands.append("mv "+route+"optimization/"+file_name+".hess "+route+"deleted/")
        commands.append("mv "+route+"1water/"+file_name+"_1wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"deleted/")
        #commands.append("mv "+file_name+"_2wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"output_files/2water/")
        #commands.append("mv "+file_name+"_3wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"output_files/3water/")
        #commands.append("mv "+file_name+"_4wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"output_files/4water/")
        commands.append("mv "+route+"SP-M06/"+file_name+"_m06_chrg.cpcm "+route+"deleted/")
        commands.append("mv "+route+"SP-M06/"+file_name+"_m06_chrg.out "+route+"deleted/")
        commands.append("mv "+route+"SP-M06/"+file_name+"_m06_chrg.multwfn.json "+route+"deleted/")
        commands.append("mv "+route+"SP-sM06/"+file_name+"_sm06_chrg.cpcm "+route+"deleted/")
        commands.append("mv "+route+"SP-sM06/"+file_name+"_sm06_chrg.out "+route+"deleted/")
        commands.append("mv "+route+"SP-sM06/"+file_name+"_sm06_chrg.multwfn.json "+route+"deleted/")
        commands.append("mv "+route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.cpcm "+route+"deleted/")
        commands.append("mv "+route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.out "+route+"deleted/")
        commands.append("mv "+route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.multwfn.json "+route+"deleted/")
        commands.append("mv "+route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.cpcm "+route+"deleted/")
        commands.append("mv "+route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.out "+route+"deleted/")
        commands.append("mv "+route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.multwfn.json "+route+"deleted/")
        commands.append("mv "+route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.cpcm "+route+"deleted/")
        commands.append("mv "+route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.out "+route+"deleted/")
        commands.append("mv "+route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.multwfn.json "+route+"deleted/")
        commands.append("mv "+route+"nbo/M06/"+file_name+"_m06_chrg.nbo.out "+route+"deleted/")
        commands.append("mv "+route+"nbo/sM06/"+file_name+"_sm06_chrg.nbo.out "+route+"deleted/")
        commands.append("mv "+route+"nbo/wb97xd/"+file_name+"_wb97xd_chrg.nbo.out "+route+"deleted/")
        commands.append("mv "+route+"nbo/swb97xd/"+file_name+"_swb97xd_chrg.nbo.out "+route+"deleted/")
        commands.append("mv "+route+"nbo/pbeh3c/"+file_name+"_pbeh3c_chrg.nbo.out "+route+"deleted/")


    for c in commands:
        print (c)  
        if test==False:
            #print ("moving") 
            os.system(c)

