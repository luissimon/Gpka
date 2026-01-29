#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# script to check if any file is missed.

import string
import os
import os.path
import sys
sys.path.append('../import')

route="/home/lsimon/jobs/pka/Gpka/PBEh3c_optimized/output_files/"
optz_route=route+"optimization"
fake_route=route+"fake_out"
pbe_route=route+"SP-pbeh3c"
nbo_pbe_route=route+"nbo/pbeh3c"
m06_route=route+"SP-m06"
nbo_m06_route=route+"nbo/m06"
sm06_route=route+"SP-sm06"
nbo_sm06_route=route+"nbo/sm06"
wb97_route=route+"SP-wb97xd"
nbo_wb97_route=route+"nbo/wb97xd"
swb97_route=route+"SP-swb97xd"
nbo_swb97_route=route+"nbo/swb97xd"
wat_route=route+"1water"

hess_files=[f.split(".hess")[0] for f in os.listdir(optz_route) if f.endswith(".hess")]
out_files=[f.split(".out")[0] for f in os.listdir(optz_route) if f.endswith(".out")]
wat_files=[f.split("_1wat.conformers.gfn2_gfnff_gbsa.xyz")[0] for f in os.listdir(wat_route) if f.endswith("_1wat.conformers.gfn2_gfnff_gbsa.xyz")]
fake_files=[f.split("_fake.out")[0] for f in os.listdir(fake_route) if f.endswith("_fake.out")]


pbe_out_files=[f.split("_pbeh3c_chrg.out")[0] for f in os.listdir(pbe_route) if f.endswith("_pbeh3c_chrg.out") ]
pbe_cpcm_files=[f.split("_pbeh3c_chrg.cpcm")[0] for f in os.listdir(pbe_route) if f.endswith("_pbeh3c_chrg.cpcm") ]
pbe_mltwn_files=[f.split("_pbeh3c_chrg.multwfn.json")[0] for f in os.listdir(pbe_route) if f.endswith("_pbeh3c_chrg.multwfn.json") ]
pbe_nbo_files=[f.split("_pbeh3c_chrg.nbo.out")[0] for f in os.listdir(nbo_pbe_route) if f.endswith("_pbeh3c_chrg.nbo.out")]

m06_out_files=[f.split("_m06_chrg.out")[0] for f in os.listdir(m06_route) if f.endswith("_m06_chrg.out") ]
m06_cpcm_files=[f.split("_m06_chrg.cpcm")[0] for f in os.listdir(m06_route) if f.endswith("_m06_chrg.cpcm") ]
m06_mltwn_files=[f.split("_m06_chrg.multwfn.json")[0] for f in os.listdir(m06_route) if f.endswith("_m06_chrg.multwfn.json") ]
m06_nbo_files=[f.split("_m06_chrg.nbo.out")[0] for f in os.listdir(nbo_m06_route) if f.endswith("_m06_chrg.nbo.out")]

sm06_out_files=[f.split("_sm06_chrg.out")[0] for f in os.listdir(sm06_route) if f.endswith("_sm06_chrg.out") ]
sm06_cpcm_files=[f.split("_sm06_chrg.cpcm")[0] for f in os.listdir(sm06_route) if f.endswith("_sm06_chrg.cpcm") ]
sm06_mltwn_files=[f.split("_sm06_chrg.multwfn.json")[0] for f in os.listdir(sm06_route) if f.endswith("_sm06_chrg.multwfn.json") ]
sm06_nbo_files=[f.split("_sm06_chrg.nbo.out")[0] for f in os.listdir(nbo_sm06_route) if f.endswith("_sm06_chrg.nbo.out")]

wb97_out_files=[f.split("_wb97xd_chrg.out")[0] for f in os.listdir(wb97_route) if f.endswith("_wb97xd_chrg.out") ]
wb97_cpcm_files=[f.split("_wb97xd_chrg.cpcm")[0] for f in os.listdir(wb97_route) if f.endswith("_wb97xd_chrg.cpcm") ]
wb97_mltwn_files=[f.split("_wb97xd_chrg.multwfn.json")[0] for f in os.listdir(wb97_route) if f.endswith("_wb97xd_chrg.multwfn.json") ]
wb97_nbo_files=[f.split("_wb97xd_chrg.nbo.out")[0] for f in os.listdir(nbo_wb97_route) if f.endswith("_wb97xd_chrg.nbo.out")]

swb97_out_files=[f.split("_swb97xd_chrg.out")[0] for f in os.listdir(swb97_route) if f.endswith("_swb97xd_chrg.out") ]
swb97_cpcm_files=[f.split("_swb97xd_chrg.cpcm")[0] for f in os.listdir(swb97_route) if f.endswith("_swb97xd_chrg.cpcm") ]
swb97_mltwn_files=[f.split("_swb97xd_chrg.multwfn.json")[0] for f in os.listdir(swb97_route) if f.endswith("_swb97xd_chrg.multwfn.json") ]
swb97_nbo_files=[f.split("_swb97xd_chrg.nbo.out")[0] for f in os.listdir(nbo_swb97_route) if f.endswith("_swb97xd_chrg.nbo.out")]

print (len(out_files))
print (len(hess_files))
print (len(wat_files))
print (len(pbe_out_files))
print (len(pbe_cpcm_files))
print (len(pbe_mltwn_files))
print (len(pbe_nbo_files))
print (len(m06_out_files))
print (len(m06_cpcm_files))
print (len(m06_mltwn_files))
print (len(m06_nbo_files))
print (len(sm06_out_files))
print (len(sm06_cpcm_files))
print (len(sm06_mltwn_files))
print (len(sm06_nbo_files))
print (len(wb97_out_files))
print (len(wb97_cpcm_files))
print (len(wb97_mltwn_files))
print (len(wb97_nbo_files))
print (len(swb97_out_files))
print (len(swb97_cpcm_files))
print (len(swb97_mltwn_files))
print (len(swb97_nbo_files))


for f in hess_files:
    if "m06" in f or "pbeh3c" in f or "wb97" in f:
        print ("probably file:" + f + " should not be in optimization directory"); continue
    if f not in out_files: print ("missing: "+f+".out")
    
    if f not in hess_files: print ("missing: "+f+".hess")
    if f not in wat_files: print ("missing: "+f+"_1wat.conformers.gfn2_gfnff_gbsa.xyz")
    #if f not in fake_files: print ("missing: "+f+"_fake.out")
    if f not in pbe_out_files: print ("missing: "+f+"_pbeh3c_chrg.out")
    if f not in pbe_cpcm_files: print ("missing: "+f+"_pbeh3c_chrg.cpcm")
    if f not in pbe_mltwn_files: print ("missing: "+f+"_pbeh3c_chrg.multwfn.json") 
    if f not in pbe_nbo_files: print ("missing: "+f+"_pbeh3c_chrg.nbo.out")  
    if f not in m06_out_files: print ("missing: "+f+"_m06_chrg.out")
    if f not in m06_cpcm_files: print ("missing: "+f+"_m06_chrg.cpcm")
    if f not in m06_mltwn_files: print ("missing: "+f+"_m06_chrg.multwfn.json") 
    if f not in m06_nbo_files: print ("missing: "+f+"_m06_chrg.nbo.out")   
    if f not in sm06_out_files: print ("missing: "+f+"_sm06_chrg.out")
    if f not in sm06_cpcm_files: print ("missing: "+f+"_sm06_chrg.cpcm")
    if f not in sm06_mltwn_files: print ("missing: "+f+"_sm06_chrg.multwfn.json")    
    if f not in sm06_nbo_files: print ("missing: "+f+"_sm06_chrg.nbo.out")    
    if f not in wb97_out_files: print ("missing: "+f+"_wb97xd_chrg.out")
    if f not in wb97_cpcm_files: print ("missing: "+f+"_wb97xd_chrg.cpcm")
    if f not in wb97_mltwn_files: print ("missing: "+f+"_wb97xd_chrg.multwfn.json")
    if f not in wb97_nbo_files: print ("missing: "+f+"_wb97xd_chrg.nbo.out")
    if f not in swb97_out_files: print ("missing: "+f+"_swb97xd_chrg.out")
    if f not in swb97_cpcm_files: print ("missing: "+f+"_swb97xd_chrg.cpcm")
    if f not in swb97_mltwn_files: print ("missing: "+f+"_swb97xd_chrg.multwfn.json")
    if f not in swb97_nbo_files: print ("missing: "+f+"_swb97xd_chrg.nbo.out")
    
