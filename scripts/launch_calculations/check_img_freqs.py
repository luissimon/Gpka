#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-
#script para colocar los archivos cada vez que hago un nuevo cálculo en sus carpetas correspondientes

import string
import os
import os.path
import sys
sys.path.append('../import')
import Molecular_structure


if __name__ == "__main__":
    report_text=""
    route="/home/lsimon/jobs/pka/output_files/PBEh3c_optimized/optimization/"
    #route="/home/lsimon/jobs/pka/output_files/b973c_optimized/optimization/"
    all_files=[f for f in os.listdir(route) if f.endswith(".out")]
    i,counter=1,0
    for file in all_files:
        print ("checking file: "+str(i)+" out of "+str(len(all_files)),end="\r")
        m=Molecular_structure.Molecular_structure(route+file, "last")
        if any( [freq<-40.0 for freq in m.QM_output.frequencies[0]] ):
            counter+=1
            #report_text+="file: "+file+" has imaginary frequencies: "+ str(m.QM_output.frequencies[0][0:8])+"\n"
            report_text+="./genreopt.py "+file+"\n"
            print ("file: "+file+" has imaginary frequencies: "+ str(m.QM_output.frequencies[0][0:8])+" "+str(counter)+" so far\n")
        i+=1
    with open ("report-img-freqs.txt","w") as f: f.write(report_text)

