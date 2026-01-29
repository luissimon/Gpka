#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-




#list of data to drop because it does not fit well (most of the times there is a good reason...)
#equilibrium has tfhe format: name_(charge of the protonated form)->(charge of the deprotonated form)
#(charge of the protonated form): 2cation: +2; cation: +1; neut: 0; an: -1; 2an: -2, 3an: -3, etc.
drop_compounds=[
                "5me2mercaptobenzimidazole_cation->neut","2mercaptobenzimidazole_cation->neut","4me2hidrazinoquinoline_2cation->cation",
                "2hidrazinoquinoline_2cation->cation","1hydrazinophthalazine_2cation->cation","3br45dih55tetramepyrazole_cation->neut","3br45dih55pentamepyrazole_cation->neut",
                "1me2oxo3456tetracl12dihpyridine_cation->neut","2oh3456tetraclpyridine_cation->neut",
                "nitropropene_neut->an", #kinetic or thermodynamic pka?
                "4pyridinecarboxamide_neut->an", #in Chemisches Zentralblatt (1964), 135(33), 46-46  pKa>15
                "14dihydro1245tetrazine_cation->neut", #1906 reference
                "11dichloroacetone_neut->an", # in:doi/pdf/10.1139/v79-193 value is 11+-1.4; possible hydrate (doi/pdf/10.1139/v86-208)
                "25diamino46dihydroxypyrimidine_cation->neut", #in reference: The asterisk indicates that a precipitate was formed in acidic solutions and their values are approx
                "methylnndimethylanthranilate_cation->neut", #very small pka, dimethylaniline is +2 pka units, and this should stabilize cation by H-bonding. Reference from 1906
                "4aminomethylimidazole_neut->an", #not accessible reference
                "2formyl3hydroxypyridine_cation->neut","2formyl3hydroxypyridine_neut->an","2formyl3metoxypyridine_cation->neut",
                "4formyl3hydroxypyridine_cation->neut","4formyl3hydroxypyridine_neut->an","4formyl3metoxypyridine_cation->neut", #hydrate formation; it is not a direct meassure:assumptions made to calculate keto pka
                #"quinoline3nitro_cation->neut","quinoline5nitro_cation->neut","quinoline6nitro_cation->neut","quinoline7nitro_cation->neut","quinoline8nitro_cation->neut", #unpublished data
                "pyrazine2aminomethyleneamino3form_cation->neut","pyrazine2aminomethyleneamino3acet_cation->neut",#hypothetical structures, the result of the hydrolysis of pteridines, but not confirmed.
                #"hydrastinine_cation->neut",#?
                "aadime1piperidineacetonitrile_cation->neut", # retro-strecker reaction with acid pH?: doi.org/10.1016/S0040-4020(01)97226-6 
                "27dime45dioh18phenanthroline_cation->neut", #only one source (unreachable)
                "68dime4571h6h8htriopyrmdn45cpyrdzne_neut->an", #only one source, hydrated?
		"8me4am2im28dihpteridine_cation->neut", #hydrated?
                #"6methyl2amino4hydroxypteridine_cation->neut", 
                "6methoxypteridine_cation->neut", #hydrated?
                "6dimeampteridine_cation->neut",  #hydrated?
                "pteridine267trime_cation->neut", #hydrated?
                #"6hydroxy24diaminopteridine_cation->neut",
                "5ac4me5678tetrahpteridie_cation->neut", #hydrated?
                "pteridine6amino_cation->neut",          #hydrated?
                "5ac4me5678tetrahpteridie_cation->neut", #hydrated?
                "pteridine2acetamide_cation->neut",      #hydrated?
                "2methiopteridine_cation->neut",         #hydrated?
                "4methiopteridine_cation->neut",         #hydrated?
                "2thioxo2358tetrahydropteridine4671Htrione_an->2an",      #hydrated?
                "12cyclohexanedioneoxime_neut->an","12cyclohexanedioneoxime_an->2an",   #hydrated?
                "dithiolanetetraoxide_neut->an",#   paper says “approximate"
                "25dioxydinitrobenzoquinone_neut->an",# value -3, out of range
                "35dicyanopyridine_cation->neut", #value of 3cyanopyridine is 1.35, almost the same!!!  only one source
                "6thiocian24diam135triazine_cation->neut","6thiocian24diam135triazine_neut->an", # “Solubility was too low to permit accurate measurements of absorptivity values, but ionization constants were obtained from absorbance us. pH plots”
                "14thiazine_cation->neut", #only one source (paper from 1948). “The only 14thiazine whose pKa has been determined”
                "nmethyl5nitrocitosine_neut->an", #only one source (another is claimed, but it does not include this compound). hydrate? cation->neut does not fit well either... delete it?
                "adenineNoxyde_an->2an", #ca. 13 in other sources
                "2methylnitropropene_neut->an","3methylnitropropene_neut->a", # kinetic or thermodynamic?
                "1oxa4azacyclohex4carbodithiolic_neut->an", #only one source
                #"4bromonn26tetramethylaniline_cation->neut ", #only one source, no Cl or I
                "2mercapto5methyl134thiadiazole_neut->an", # only one source, unreachable paper.
                "hydrastinine_neut->an","hydrastinine_cation->neut", #only one source, from 1925 
                "16dih8oh1me6oxopurine_an->2an", #in 10.1039/P19740002229  this is >13 instead of 11.8

                "139trime6oco69dih3hpurin1ium8olate_cation->neut",
                "13dime8oxo78dih1hpurin3ium6olate_cation->neut",
                "13dime8oxo78dih1hpurin3ium6olate_neut->an",
                "14benzoquinoneimine_cation->neut", #only one source from 1930, tautomerization, indirect measurement?
                "nme14benzoquinoneimine_cation->neut", #only one source from 1930, tautomerization, indirect measurement?
                "6me1234tetrahydroquinozaline_2cation->cation","6meo1234tetrahydroquinozaline_2cation->cation", # value of 1,2-phenylenediamine is 0.8, much lower. Only one source; In ohter paper value is 1.17 for 234tetrahydroquinozaline_2cation->cation, not 2. High dependence with concentration, mention some hydrolysys. 
                "1nitro23indanone_neut->an", #only one source
                "33dimethylnitropropene_neut->an", # tautomerization?
                "2aminobenzoccinnolin_cation->neut", #only one source (but a nature!)
                "2methoxypteridine_cation->neut", # hydrate?
                #"4bromonn26tetramethylaniline_cation->neut", #only one source, difference with nn26tetramethylaniline is much larger than difference between bromonndimeaniline and dimeaniline 
                "5am6carbxm1h123triazolo45bpyridine_neut->an", #only one source,  more acidic than 6-CN!!
                "8azapurine_cation->neut", "2am123treiazolo5p4p54pyrimidine_cation->neut","2am6me123triazolo5p4p45pyrimidine_cation->neut", #cation hydrates (mentioned in the same paper from which the pka is taken!!)
                "2oh123triazolo5p4p54pyrimidine_neut->an", #neutral hydrates (mentioned in the same paper from which the pka is taken!!)
                "emimycin_neut->an", #could not find value in the reference
                "4formyl123triazole_cation->neut", #it forms a sort of bis-hemyaminal Russ J Org Chem 40, 1804–1809 (2004). https://doi.org/10.1007/s11178-005-0103-4
                "8oh1oxquinoline_neut->an",  #in https://doi.org/10.1016/0022-1902(69)80331-3 the value 11.75 is given (but it is not a direct measurement...)
                "2pme123triazolo5p4p45pyrimidine_cation->neut", #in the reference: "The pK, is an equilibrium value (anhydrous and hydrated species"
                "1methyl4pyridoneimine_cation->neut",  # only one source; in: https://cdnsciencepub.com/doi/pdf/10.1139/v76-130 value >17 is given from extrapolation from mixtures of DMSO-water
                "7methiopteridine_cation->neut",       #        hydrated?
                "3oh6mercaptopyridazine_cation->neut", #  value -1.7 also published
                "125trimethylpyrrole_cation->neut","25dimethylpyrrole_cation->neut", #  it attempts to discriminate between pka for alpha or beta protonation based on populations derived from NMR in 10.1021/ja00884a005 (but different conditions). The equilibrium pka is not given.
                "furantetracarboxylic_neut->an","furantetracarboxylic_an->2an", "furantetracarboxylic_2an->3an", "furantetracarboxylic_3an->4an",    #only one source, hydrated at low pH?
                "2am345678hxh4quinazolinecarbx_neut->an",    # only one source, not sure about structure
                "chlorcyclizine_2cation->cation",  #it is another compound!!!
                "24diamino7ph5oxypterin_cation->neut" ,"7ph5oxypterin_cation->neut","7ph5oxypterin_neut->an",  # not sure about the structure: other compounds are oxydized under the same conditions in a different N atom
                "4aminophenyl2pyridine1oxide_2cation->cation" , "4aminophenyl2pyridine1oxide_cation->neut", "2maminophenylpyridine1oxide_2cation->cation","2maminophenylpyridine1oxide_cation->neut","4aminophenyl4pyridine1oxide_cation->neut",  #unsure of the structure: obtained by reduction from the nitro compounds (whose pKa is predicted well) so maybe N-oxide is reduced and they obtain the hydroxylamine
                "14butanediamine_2cation->cation", "14butanediamine_cation->neut",     #too many conformations, values too close
                "dibromosuccinic_an->2an",    # in reference pubs.rsc.org/en/content/articlepdf/1959/jr/jr9590002492: DL-dibromosuccinic acid is hydrolized and pKa value is not given
                "7methyl159triazabicyclo550dodecane_cation->neut", "7methyl159triazabicyclo550dodecane_2cation->cation", "7methyl159triazabicyclo550dodecane_3cation->2cation",   # all conformations?
                "147triazacyclononane_3cation->2cation",   "147triazacyclononane_2cation->cation", "147triazacyclononane_cation->neut", #all conformations ?
                "1278tetradimethylaminonaphtalene_cation->neut",   # value extrapolated from DMSO due to poor solubility
                "34dimethylpyrrole_cation->neut",  "2methylpyrrole_cation->neut",   
                #"tristrifluoromethylmethane_neut->an", #could not find the paper!!! there are not any solubility problems?
                "ptolylpyrroline_cation->neut", # 1930 paper, unsure of the structure (isomerization after protonation in C atom?)  
                "chlorodinitromethane_neut->an", "bromodinitromethane_neut->an", #determined indirectly in 10.1021/jo01035a035, value copied in the cited reference
                "fluorodinitromethane_neut->an", # only two sources: an unaccessible russian journal and a "personal comunication"; also, for clhlorodinitromethane it was determined indirectly
                "27dimethoxy18dimorpholinonaphtalene_cation->neut","18dimorpholinonaphtalene_cation->neut", #protonation and deprotonation is slow, so it is not clear how it affects to the equilibrium pka value. The values are much more smalr than for corresponding dimethylaminonaphthalene, 5 pka units for 18dimorpholio... and 4  pka units for 27dimethoxy18dimorpholino...
                "trichloromethylphosphonic_neut->an", "trifluoromethylphosphonic_neut->an", #paper warns of some problems of linearity with this value
                "4me5carboxythiazole_cation->neut", # possible hydrate? in https://doi.org/10.1016/S0040-4039(01)90160-1 they mention that this possition can be deuterated by D2O addition on this possition on a richer protonated thiazole
                "7me4cl7hpyrrolo23dpyrimidine_cation->neut", #paper mentions solubility problems: "Due to its insolubility in aqueous perchloric acid, the pKa value for I V was obtained spectrophotometrically"
                #"238trimethylquinoline_cation->neut", #only one paper from 1933, is it really soluble?
                "15diph37diazabicyclo3331nona9one_cation->neut","37dime15diph37diazabicyclo3331nona9one_cation->neut" #only in one paper  
                "mesulfonic1oh12pyridyl_2an->an","mesulfonic1oh13pyridyl_2an->an","mesulfonic1oh14pyridyl_2an->an"   #b973c is unable to optimize 2an structure
 ]


#data that will be forced to be in the test set
force_in_test_set_compounds=["quinoline3nitro_cation->neut","quinoline5nitro_cation->neut","quinoline6nitro_cation->neut","quinoline7nitro_cation->neut","quinoline8nitro_cation->neut",
                             "6methyl2amino4hydroxypteridine_cation->neut","6hydroxy24diaminopteridine_cation->neut","4bromonn26tetramethylaniline_cation->neut","tristrifluoromethylmethane_neut->an",
                              "238trimethylquinoline_cation->neut"]

#data that will be forced to be in the train set
force_in_train_set_compounds=[]


import pandas as pd


def do_drop_compounds(data):
    for d in drop_compounds:    data =data[data["compn"].str.startswith(d)==False]
    return data


