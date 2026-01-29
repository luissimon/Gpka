#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import string
import os
import os.path
import numpy as np
import pandas as pd
sys.path.append('../import')
from routes import extracted_data_route
from routes import labels_file_name
from routes import output_files_route


#the place where all the files live
labels_route= extracted_data_route
files_route= output_files_route+"PBEh3c_optimized/"
#files_route= output_files_route+"b973c_optimized/"

#the csv file containing the experimental pka values
labels=pd.read_csv(labels_route+labels_file_name,encoding='unicode_escape')
labels.set_index("compn",inplace=True)
#labels.drop(['alternative pka','alternative ref.','is Lange?'],axis=1,inplace=True)
labels.dropna(how='all', axis=1, inplace=True)

names=labels.index



expd_names=set([])

def append_once(my_list,item):
    if item not in my_list: my_list.append(item)

for name in names:
    if name.split("_")[1].startswith("neut"): expd_names.add(name.split("_")[0]+"-neut");  expd_names.add(name.split("_")[0]+"-an") 
    if name.split("_")[1].startswith("an"):   expd_names.add(name.split("_")[0]+"-an");    expd_names.add(name.split("_")[0]+"-2an")
    if name.split("_")[1].startswith("2an"):  expd_names.add(name.split("_")[0]+"-2an");    expd_names.add(name.split("_")[0]+"-3an")
    if name.split("_")[1].startswith("3an"):  expd_names.add(name.split("_")[0]+"-3an");    expd_names.add(name.split("_")[0]+"-4an")
    if name.split("_")[1].startswith("4an"):  expd_names.add(name.split("_")[0]+"-4an");    expd_names.add(name.split("_")[0]+"-5an")
    if name.split("_")[1].startswith("5an"):  expd_names.add(name.split("_")[0]+"-5an");    expd_names.add(name.split("_")[0]+"-6an")
    if name.split("_")[1].startswith("6an"):  expd_names.add(name.split("_")[0]+"-6an");    expd_names.add(name.split("_")[0]+"-7an")
    if name.split("_")[1].startswith("cation"): expd_names.add(name.split("_")[0]+"-cation");  expd_names.add(name.split("_")[0]+"-neut")
    if name.split("_")[1].startswith("2cation"): expd_names.add(name.split("_")[0]+"-2cation");  expd_names.add(name.split("_")[0]+"-3cation")
    if name.split("_")[1].startswith("3cation"): expd_names.add(name.split("_")[0]+"-3cation");  expd_names.add(name.split("_")[0]+"-4cation")
    if name.split("_")[1].startswith("4cation"): expd_names.add(name.split("_")[0]+"-4cation");  expd_names.add(name.split("_")[0]+"-5cation")
    if name.split("_")[1].startswith("5cation"): expd_names.add(name.split("_")[0]+"-5cation");  expd_names.add(name.split("_")[0]+"-6cation")

all_files=[f[:-5] for f in os.listdir(files_route+"optimization") if f[-5:]==".hess"]

print ("found "+str(len(expd_names))+ "different compounds in csv file")
print ("found "+str(len(all_files))+ "files")
orphans=[]
log=""
not_delete=["methane-neut","methane-an","ammonia-cation","ammonia-neut","h2o-neut","h2o-cation","h2o-an","ph3-neut","sulfhidric-neut","sulfhidric-an","ph3-cation"]
for f in all_files:
    if       "-neut" in f: ff=f.split("-neut")[0]+"-neut"
    elif     "-7an" in f: ff=f.split("-7an")[0]+"-7an" 
    elif     "-6an" in f: ff=f.split("-6an")[0]+"-6an"
    elif     "-5an" in f: ff=f.split("-5an")[0]+"-5an"
    elif     "-4an" in f: ff=f.split("-4an")[0]+"-4an"
    elif     "-3an" in f: ff=f.split("-3an")[0]+"-3an"
    elif     "-2an" in f: ff=f.split("-2an")[0]+"-2an"
    elif     "-an" in f:  ff=f.split("-an")[0]+"-an"
    elif     "-6cation" in f: ff=f.split("-6cation")[0]+"-6cation"
    elif     "-5cation" in f: ff=f.split("-5cation")[0]+"-5cation"
    elif     "-4cation" in f: ff=f.split("-4cation")[0]+"-4cation"
    elif     "-3cation" in f: ff=f.split("-3cation")[0]+"-3cation"
    elif     "-2cation" in f: ff=f.split("-2cation")[0]+"-2cation"
    elif     "-cation" in f: ff=f.split("-cation")[0]+"-cation"

    if ff not in expd_names and ff not in orphans and ff not in not_delete: 
       log+="\n./borrar_ab.py "+ ff+"*"
       orphans.append(ff)
with open("orphans.txt","w") as f: f.write(log)
number_of_files_to_delete=0
for of in orphans:
    #print(of);print ("======")
    files_to_move= [f[:-5] for f in os.listdir(files_route+"optimization") if f.startswith(of) and f[-5:]==".hess"]
    #print (files_to_move)
    number_of_files_to_delete+=len(files_to_move)
    commands=[]
    for file_name in files_to_move:
        commands.append("mv "+files_route+"fake_out/"+file_name+"_fake.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"optimization/"+file_name+".out "+files_route+"orphans/")
        commands.append("mv "+files_route+"optimization/"+file_name+".hess "+files_route+"orphans/")
        commands.append("mv "+files_route+"1water/"+file_name+"_1wat.conformers.gfn2_gfnff_gbsa.xyz "+files_route+"orphans/")
        #commands.append("mv "+file_name+"_2wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"output_files/2water/")
        #commands.append("mv "+file_name+"_3wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"output_files/3water/")
        #commands.append("mv "+file_name+"_4wat.conformers.gfn2_gfnff_gbsa.xyz "+route+"output_files/4water/")
        commands.append("mv "+files_route+"SP-m06/"+file_name+"_m06_chrg.cpcm "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-m06/"+file_name+"_m06_chrg.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-m06/"+file_name+"_m06_chrg.multwfn.json "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-sm06/"+file_name+"_sm06_chrg.cpcm "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-sm06/"+file_name+"_sm06_chrg.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-sm06/"+file_name+"_sm06_chrg.multwfn.json "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.cpcm "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-wb97xd/"+file_name+"_wb97xd_chrg.multwfn.json "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.cpcm "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-swb97xd/"+file_name+"_swb97xd_chrg.multwfn.json "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.cpcm "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"SP-pbeh3c/"+file_name+"_pbeh3c_chrg.multwfn.json "+files_route+"orphans/")
        commands.append("mv "+files_route+"nbo/m06/"+file_name+"_m06_chrg.nbo.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"nbo/sm06/"+file_name+"_sm06_chrg.nbo.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"nbo/wb97xd/"+file_name+"_wb97xd_chrg.nbo.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"nbo/swb97xd/"+file_name+"_swb97xd_chrg.nbo.out "+files_route+"orphans/")
        commands.append("mv "+files_route+"nbo/pbeh3c/"+file_name+"_pbeh3c_chrg.nbo.out "+files_route+"orphans/")


    with open("to_run.txt","a") as f: f.write("\n".join(commands))
print("number of files to delete: "+str(number_of_files_to_delete))  
