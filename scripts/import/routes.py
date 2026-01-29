#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

root_route="/home/lsimon/jobs/pka/Gpka/"
extracted_data_route=root_route+"extracted_data/"
output_files_route=root_route+"output_files/"
labels_csv_file_name="exp_pka_values.csv"
sampl_extracted_data_route=root_route+"sampl/extracted_data/"
sampl_output_files_route=root_route+"sampl/output_files/"
sampl_labels_csv_file_name="sampl_all_pkas.csv"



scratch_home="/scratch/lsimon/"

orca5_exe="/home/orca5/orca"
orca6_exe="/home/orca6/orca"
orca5_mkl_exe="/home/orca5/orca_2mkl"
orca6_mkl_exe="/home/orca6/orca_2mkl"
do_multwfn_exe=root_route+"scripts/launch_calculations/do_multwfn.py"
multiwfn_exe="/home/multiwfn_3.8/Multiwfn"
multiwfn_home="/home/multiwfn_3.8"
ofakeG_exe="/home/g16/g16/OfakeG"
nbo_orca5_exe="/home/nbo7/bin/nbo7.i4.exe"
nbo_orca6_exe="/home/nbo7/bin/nbo7.i8.exe"
mpi_module="openmpi-4.1.1-gcc-4.9.4-6bb7ahv" #the module that must be loaded to run orca in parallel