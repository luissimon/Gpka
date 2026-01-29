#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
#script para colocar los archivos cada vez que hago un nuevo cálculo en sus carpetas correspondientes

import string
import os
import os.path
import sys
sys.path.append('../import')
from routes import output_files_route
from routes import sampl_output_files_route

route=output_files_route+"PBEh3c_optimized/"
#route=output_files_route+"b973c_optimized/"
#route=sampl_output_files_route+"PBEh3c_optimized/"
#route=sampl_output_files_route+"b973c_optimized/"

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]!="": file_names=[sys.argv[1]]
    else: file_names=[f.split(".hess")[0] for f in os.listdir() if f.endswith(".hess") ]

     

    for file_name in file_names:
    
        commands=[]
        commands.append("mv "+file_name+"_fake.out "+route+"fake_out/")
        commands.append("mv "+file_name+".out "+route+"optimization/")
        commands.append("mv "+file_name+".hess "+route+"optimization/")
        commands.append("mv "+file_name+"_1wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"1water/")
        #commands.append("mv "+file_name+"_2wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"2water/")
        #commands.append("mv "+file_name+"_3wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"3water/")
        #commands.append("mv "+file_name+"_4wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"4water/")
        commands.append("mv "+file_name+"_m06_chrg.cpcm "+route+"SP-M06/")
        commands.append("mv "+file_name+"_m06_chrg.out "+route+"SP-M06/")
        commands.append("mv "+file_name+"_m06_chrg.multwfn.json "+route+"SP-M06/")
        commands.append("mv "+file_name+"_sm06_chrg.cpcm "+route+"SP-sM06/")
        commands.append("mv "+file_name+"_sm06_chrg.out "+route+"SP-sM06/")
        commands.append("mv "+file_name+"_sm06_chrg.multwfn.json "+route+"SP-sM06/")
        commands.append("mv "+file_name+"_wb97xd_chrg.cpcm "+route+"SP-wb97xd/")
        commands.append("mv "+file_name+"_wb97xd_chrg.out "+route+"SP-wb97xd/")
        commands.append("mv "+file_name+"_wb97xd_chrg.multwfn.json "+route+"SP-wb97xd/")
        commands.append("mv "+file_name+"_swb97xd_chrg.cpcm "+route+"SP-swb97xd/")
        commands.append("mv "+file_name+"_swb97xd_chrg.out "+route+"SP-swb97xd/")
        commands.append("mv "+file_name+"_swb97xd_chrg.multwfn.json "+route+"SP-swb97xd/")
        commands.append("mv "+file_name+"_pbeh3c_chrg.cpcm "+route+"SP-pbeh3c/")
        commands.append("mv "+file_name+"_pbeh3c_chrg.out "+route+"SP-pbeh3c/")
        commands.append("mv "+file_name+"_pbeh3c_chrg.multwfn.json "+route+"SP-pbeh3c/")
        commands.append("mv "+file_name+"_m06_chrg.nbo.out "+route+"nbo/M06/")
        commands.append("mv "+file_name+"_sm06_chrg.nbo.out "+route+"nbo/sM06/")
        commands.append("mv "+file_name+"_wb97xd_chrg.nbo.out "+route+"nbo/wb97xd/")
        commands.append("mv "+file_name+"_swb97xd_chrg.nbo.out "+route+"nbo/swb97xd/")
        commands.append("mv "+file_name+"_pbeh3c_chrg.nbo.out "+route+"nbo/pbeh3c/")

        for c in commands:
            print (c)
            #os.system(c)

