#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
#script para renombrar los archivos cada vez que hago un nuevo cálculo en sus carpetas correspondientes

import string
import os
import os.path
import sys
sys.path.append('../import')
from routes import output_files_route
from routes import sampl_output_files_route
if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]!="":
        file_name=sys.argv[1]
        if len(sys.argv)>2 and sys.argv[2]!="":
            new_file_name=sys.argv[2]
        else: print("gimme names!"); sys.exit()

    else: print("gimme names!"); sys.exit()
    if "-test" in sys.argv: test=True
    else: test=False

    route=output_files_route+"PBEh3c_optimized/" 
    #route=output_files_route+"b973c_optimized/"
    #route=sampl_output_files_route+"PBEh3c_optimized/" 
    #route=sampl_output_files_route+"b973c_optimized/"

    commands=[]
    commands.append("mv "+route+"fake_out/"+file_name+"_fake.out "+route+"fake_out/"+new_file_name+"_fake.out ")
    commands.append("mv "+route+"optimization/"+file_name+".out "+route+"optimization/"+new_file_name+".out ")
    commands.append("mv "+route+"optimization/"+file_name+".hess "+route+"optimization/"+new_file_name+".hess ")
    commands.append("mv "+route+"1water/"+file_name+"_1wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"1water/"+new_file_name+"_1wat.conformers.gfn2_gfnff_gbsa.xyz ")
    commands.append("mv "+route+"SP-M06/"+file_name+"_m06_chrg.cpcm "+route+"SP-M06/"+new_file_name+"_m06_chrg.cpcm ")
    commands.append("mv "+route+"SP-M06/"+file_name+"_m06_chrg.out "+route+"SP-M06/"+new_file_name+"_m06_chrg.out ")
    commands.append("mv "+route+"SP-M06/"+file_name+"_m06_chrg.multwfn.json "+route+"SP-M06/"+new_file_name+"_m06_chrg.multwfn.json ")
    commands.append("mv "+route+"SP-sM06/"+file_name+"_sm06_chrg.cpcm "+route+"SP-sM06/"+new_file_name+"_sm06_chrg.cpcm ")
    commands.append("mv "+route+"SP-sM06/"+file_name+"_sm06_chrg.out "+route+"SP-sM06/"+new_file_name+"_sm06_chrg.out ")
    commands.append("mv "+route+"SP-sM06/"+file_name+"_sm06_chrg.multwfn.json "+route+"SP-sM06/"+new_file_name+"_sm06_chrg.multwfn.json ")
    commands.append("mv "+route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.cpcm "+route+"SP-wb97xd/"+new_file_name+"_wb97xd_chrg.cpcm ")
    commands.append("mv "+route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.out "+route+"SP-wb97xd/"+new_file_name+"_wb97xd_chrg.out ")
    commands.append("mv "+route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.multwfn.json "+route+"SP-wb97xd/"+new_file_name+"_wb97xd_chrg.multwfn.json ")
    commands.append("mv "+route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.cpcm "+route+"SP-swb97xd/"+new_file_name+"_swb97xd_chrg.cpcm ")
    commands.append("mv "+route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.out "+route+"SP-swb97xd/"+new_file_name+"_swb97xd_chrg.out ")
    commands.append("mv "+route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.multwfn.json "+route+"SP-swb97xd/"+new_file_name+"_swb97xd_chrg.multwfn.json ")
    commands.append("mv "+route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.cpcm "+route+"SP-pbeh3c/"+new_file_name+"_pbeh3c_chrg.cpcm ")
    commands.append("mv "+route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.out "+route+"SP-pbeh3c/"+new_file_name+"_pbeh3c_chrg.out ")
    commands.append("mv "+route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.multwfn.json "+route+"SP-pbeh3c/"+new_file_name+"_pbeh3c_chrg.multwfn.json ")
    commands.append("mv "+route+"nbo/M06/"+file_name+"_m06_chrg.nbo.out "+route+"nbo/M06/"+new_file_name+"_m06_chrg.nbo.out ")
    commands.append("mv "+route+"nbo/sM06/"+file_name+"_sm06_chrg.nbo.out "+route+"nbo/sM06/"+new_file_name+"_sm06_chrg.nbo.out ")
    commands.append("mv "+route+"nbo/wb97xd/"+file_name+"_wb97xd_chrg.nbo.out "+route+"nbo/wb97xd/"+new_file_name+"_wb97xd_chrg.nbo.out ")
    commands.append("mv "+route+"nbo/swb97xd/"+file_name+"_swb97xd_chrg.nbo.out "+route+"nbo/swb97xd/"+new_file_name+"_swb97xd_chrg.nbo.out ")
    commands.append("mv "+route+"nbo/pbeh3c/"+file_name+"_pbeh3c_chrg.nbo.out "+route+"nbo/pbeh3c/"+new_file_name+"_pbeh3c_chrg.nbo.out ")


    for c in commands:
        print (c)
        if not test:
            os.system(c)


