#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import sys
import pandas as pd
#import dask
import json
from datetime import date, datetime, timedelta
from os.path import getmtime

"""
requires modified*  submitit for parallel distribution of jobs using slurm, 
            +submitit library has to be transformed to make it work:
                modify file: ...python3.8/site-packages/submitit/slurm/slurm.py to set use_srun=False (instead of True)
                modify file: ...python3.8/site-packages/submitit/core/utils.py   to set self.task_id  always 0 in line 55: self.task_id = 0 #task_id or 0 
            also imposed by submitit is that any module needed by the score_function that is not in the default imports_path MUST be imported inside the function
            after instruction: sys.path.insert(0,"path to module")

TO DO: implement dask for parallelization in distributed memory systems (but I have not been able to make it run!!!)
"""

credits=[
"  ..   .      ..                .      .   ..   ..   . ..   ..    .   .       .   .                           ............ ..   ..   .      ..   . ",
"  ..   .      ..                .      .   ..   ..   . ..   ..    .   .       .   .                    ..-#@@@@%@@@@@@@@@@@@@@+...   .      ..   . ",
"  ..   .      ..                .      .   ..   ..   . ..   ..    .   .       .   .                 :+#@-+%=#@*#@@%@+@@@%@@@@@@@@@%%#*-   . ..   . ",
"  ..   .      ..                .      .   ..   ..   . ..   ..    .   .       .   .             .#%@%@%=.%@-%%%%@@%@%@@@@*@@@%@@@@@-:%@@@%: ..   . ",
"  ..   .      ..     ......@@%%@@@@@@*:.   ..   ..   . ..   ..    .   .       .   .          .=%@=%@@%%@@@@@%@@%%%%%@%%@%%@%@@@@@@@@@*@@@@@@#.   . ",
"  ..   .      ..  ..:*@@@*=@@@@%...*@=-@@=...   ..   . ..   ..    .   .       .   .         .**@%.-@+%+@@*@@@%@%@@@@@@@@@%@@%%@@@@@@@@@@@@@@@@#. . ",
"  ..   .      ..:#@@@:....:..-%#.#+.+%-.#@@*.   ..   . ..   ..    .   .       .   .        .%+:.*::+%-:+-*.:-@#%*@@@%*#%%@@%@@@%%@%@@@%%@@@@@@@@@- ",
"  ..   .    .=@@@***+#%:-*%#*#=---*#@%#*-@+@@@#-..   . ..   ..    .   .       .   .      .+#.*:-..-:-@@%@@@--%=+****%*%%%#%%%@#+@@@%@@%@@@@@@@@@@@ ",
"  ..   .  .%@@-%%:::.#.:.. =-::=@-+-.%@@@-=@%#@*@@+-.. ..   ..    .   .       .   .    .#:..:.*.. =-@%%+.@@@::+*%#*%@%#%%@%%%@@%%%%%@%%%@%@@@@@@@% ",
"  ..   ..-%@=*::..+..:%%%@%+#@#:=+.:@@++%%##.-@=@::%@@%=.   ..    .   .       .   .  .+-..-%.+.:@-..@@@@@@@%:+==+@@%%@%#%%%@@%%%@%#%@@%@%@@@@@%@@# ",
"  ..  :#@#*=.+#. +@#..... ####::-.:.%@@@%@=.=:@.%@:.-..=@@=...    .   .       .   .:#+:.... :.:@#+%+.+@@%+.:=::-=**#=:.%@%#%#%@%%%%@@%%@@%+@@@@@@+ ",
"  ...@%%-:+#*-.:%#. .-%-.:+:    .::-.#@@@*-*+%@-:#@#*:....+@%.    .   .       .  .*: -:.:=-:=#+-..   ..:. ..+-..=..=:-:*+-%#*@@%##@%@%@@%@%%**:*#* ",
"  .=@==..:#-::#=.:-=: .=-+.     . :..  . :::. @=...  . ..   :%#:  .   .       ..+:.=-.:-.*-...==-=-...::.:.. :.. ..-.--*.--=+*=**%%*###@#+%#%##*@: ",
":%@:++..:+. ==.-*:.+#=..        .  .=-+*....-:%...   . ..   ...##..   .       =+. :.:.=. ..=-..::=+..:.::...  ..   ...-..:==*#:@:###-+##%@*@****@. ",
".--##.*@+...+%.:#.+%.     :  .. .     -#:-#++@#@@+===----=*#%*+.+@-   .     .*: .+. .       ..:#....:=: -.    ..   .....  -%...-:##=--.:+-++#%%-=. ",
"  ..*%:.:.%@%=.:+.#.  ....+....-.  .=**%**:=%@=-.... ....+..=+#@*=:   .    .#..   .             .*:...:#. .   ...::+=...   ...::-:*:--.%*%%#*##@%. ",
"  ..  ..   ##.#.  -:   ..:..  ...:-+#=*=*%==:*@%.-:#:@@@@%-...    .   .   .*  .   .                  ++..:=:  ::.:.-.+-..-.:. .*.*+.::=:=--:.++#.. ",
"  ..   .   ::-%:        ...  :.:.#-*:.#+..%::%#@@%#=-. ..   ..    .   .  .+.  .   .                  ..-=.   +-...::.... :-.:-- ...  .-.:::.:.::.. ",
"  ..   .    ..=#+..    ......%@%*-..=+*.:- +@@=...   . ..   ..    .   . .*.   .   .   .... : ..... ...:=+%: .=.-+=:-.--.:-=..   ..   . .... .... . ",
"  ..   .     ..+#.*-  .#+#=-@.%@@:%*=..#-=:@%.  ..   . ..   ..    .   . =.    .  .. ..---.:.-*%#%#%%%%@*.*-. .=... ..... . ..   ..   .      ..   . ",
"  ..   .      .=*+.*%.#-==:#-#.-**::=-%+#+*@.   ..   . ..   ..    .   .-=     ....:-*#*+--::= -:.:....:=**+ .-:. :*--*..   ..   ..   .      ..   . ",
"  ..   .      ...:.+#+++:#::::#*..*: :@:+*...   ..   . ..   ..    .   :*     .::*#:            .::.-.:-:*@#.....:. .....   ..   ..   .      ..   . ",
"  ..   .      ..     . ..:=::::-=. +=##%@  ..   ..   . ..   ..    .   *...-=++-.  .                  :*@*:..-...  .+:...   ..   ..   .      ..   . ",
"  ..   .      ..       ..   ..  . ++..@@.  ..   ..   . ..   ..    .   *%%%#..=.=-....  .  ..      =##@#:...:*.-.  ......   ..   ..   .      ..   . ",
"  ..   .      ..             -:++-=.#@..   ..   ..   . ..   ..    .   .   .....-==*#%@%%%@%#%#%@@@@@@==@@-.=.-..   .....   ..   ..   .      ..   . ",
"  ..   .      ..   .. .. .:*-#:*.:+@@- .   ..   ..   . ..   ..    .   .       .   .                   ..+@-+@.--*.=-*:..   ..   ..   .      ..   . ",
"  ..   .      ..   .....:.+:.*=::.%::  .   ..   ..   . ..   ..    .   .       .   .                       +*..:+........   ..   ..   .    . ..   . ",
"  ..   .      ..              :++@-    .   ..   ..   . ..   ..    .   .       .   .                       ..@.....=.+... . ..   ..   .      ..   . ",
"  ..   .      ..           .  +:#=     .   ..   ..   . ..   ..    .   .       .   .                         :@-+...-.*.=.---.   ..   .      ..   . ",
"  ..   .      ..           ..%%+@+     .   ..   ..   . ..   ..    .   .       .   .                          :%.: ..:.:-.=.:.   ..   .      ..   . ",
"  ..   .      ..                .      .   ..   ..   . ..   ..    .   .       .   .                           =#=*:**...   ..   ..   .      ..   . ",
"  ..   .      ..                .      .   ..   ..   . ..   ..    .   .       .   .                           .*+.=.+*..   ..   ..   .      ..   . ",
"  'In the Galapagos Archipelago, many even of the birds,                                                      ..@++..=.=   ..   ..   .      ..   . ",
"  though so well adapted for flying from island to island, are distinct on each.'                              .+@=+#-..   ..   ..   .      ..   . ",
"  Charles Darwin, in: 'On the Origin of Species'                                                               .   .....   ..   ..   .      ..   . ",
"                                                                                                                                                   ",
]

for l in credits: print (l)

#required to jsonize numpy types
class NpEncoder(json.JSONEncoder):
    import pandas as pd
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.string_) or isinstance(obj,np.str_):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, (pd.Series,pd.DataFrame )):
            return obj.to_json()

        return super(NpEncoder, self).default(obj)

def outlier_proof_statistics(a,score_values=[],eps_dist_quantile=[0.5,0.8],return_stats=["mean","std"],exclude_limit=[]):
    """
    auxiliary function to remove outliers and return mean and standard deviation
    uses DBSCAN to cluster data and remove outliers 
    or a "exclude_limit" so anything that is not within the limits is excluded
    """
    #obsolete; based on mean and standard deviation, but might not work well if outliers have a non-negigible contribution to mean and std
    #orig_mean,orig_std=np.mean(a),np.std(a)
    #mod_a=[x for x in a if (x-orig_mean)/orig_std<limit and (x-orig_mean)/orig_std>-limit]

    if score_values==[]: score_values=a

    if exclude_limit==[]:#use DBSCAN to cluster data
        from dbscan1d.core import DBSCAN1D
        dbscan=DBSCAN1D(eps=np.abs(np.nanquantile(score_values,eps_dist_quantile[0])-np.nanquantile(score_values,eps_dist_quantile[1])), min_samples=5)
        clusters=10000-dbscan.fit_predict(np.array(score_values)) #bincount cannot use negative numbers, so clusters are shifted 10000
        max_cluster=np.argmax(np.bincount(clusters))

    else:
        max_cluster=1
        clusters=[1 if (score_values[i]>exclude_limit[0] and score_values[i]<exclude_limit[1]) else 0 for i in range(len(score_values))]

    #remove outliers
    mod_a=[a[i] for i in range(len(a)) if clusters[i]==max_cluster]

    #prepare output
    if return_stats==[]: return [np.mean(mod_a),np.std(mod_a)]
    else:
        output={"mean":np.mean(mod_a),
                "std":np.std(mod_a)}
        if "median" in return_stats: output["median"]=np.nanmedian(mod_a)
        if "max" in return_stats: output["max"]=np.nanmax(mod_a)
        if "min" in return_stats: output["min"]=np.nanmin(mod_a)
        if "q50" in return_stats: output["q50"]=np.nanquantile(mod_a,0.5)
        if "q80" in return_stats: output["q80"]=np.nanquantile(mod_a,0.8)
        if "q90" in return_stats: output["q90"]=np.nanquantile(mod_a,0.9)
        if "q95" in return_stats: output["q95"]=np.nanquantile(mod_a,0.95)
        if "q99" in return_stats: output["q99"]=np.nanquantile(mod_a,0.99)
        if "valid_instances" in return_stats: output["valid_instances"]=[i for i in range(len(a)) if clusters[i]==max_cluster]
        if "valid_values" in return_stats: output["valid_values"]=mod_a
        if "standarized_values"  in return_stats: output["standarized_values"]=[ (a[i]-np.mean(mod_a))/np.std(mod_a) if clusters[i]==max_cluster else np.nan for i in range(len(a))]
        return output


class finch(): 
    """
    class that define "finchess" ojbects. Each "finch" is one member of the flock_of_finches unit (the population that will evolve)
    requires:
    genotype:         a dictionary with keys: name of each of the genes, 
                                        values: list of value of each of the genes. If the list contain more than one element, only the first will be used in scoring the finch,
                                                so the colection of the first elements of these lists are the fenotype, but the other elements can also be transmitted, so it can
                                                implement a Mendel-like evolution.
    score_function:   a reference to the function that will be used to score each finch performance, 
                    that accepts score_function_params and is able to "translate" the genotype to the parameters required for evaluating the score.
                    the score_function must return at least a score and optionally it can return a genotype that will be used to modify the finch (for Lamarckian evolution),
                    and a memotype, a dictionary with information that the finch has learned during evaluation but that will not be inherited.
                    if the score function returns:
                            a single value: it will be taken as the score, 
                            a list, the first element will be considered the score and the second the genotype; the third, if present, will be the genotype
                            a dictionary, the ["score"] value wil be considered the score and the ["genotype"] value will be considered the genotype and ["memotype"] the memotype.
                    best performing finches correspond to higher scores.
    a new finch can be created:
    1) passing to the creator of the class the genotype: (in a dicionary format: {"genotype": xxxx  })
    2) passing the possible values of each gene ("alleles"), so gene values will be randomly* selected from these values (*probability of choosing gene values can be conditioned on the frequency of a population)
    3) passing two finches (parents) that will be combined, in a dictionary format: {"parent1":xxxx, "parent2":yyyy})
        weights for each parent can also be specified, and also how the parents cromosomes are combined:
        a) for boolean or string values, or if mix_parents==False, a value of the parents will be randomly chosen
        b) for int or float values, and if mix_parents==average, the average value of the values of their parents
        c) for int or float values, and if mix_parents==norm_distribution, a sample from a normal distribution centered at the average of the two values and with 
        std. deviation equal to (parent gene value - median)
        in case c), the sampling may lead to values outside the limits of the gene values defined in the "alleles". 
        To prevent this, a "bounds" dictionary can also be passed, similar to "alleles" dictionary, 
        that includes the min and max values in a list: {"allele_key":[min_value,max_value]} 
        The formats [None,max_value] or [min_value,None] are also possible -meaining that there is not inf. and sup. limits, repectively.
    
    The property "mix_parents" controls how genes from parents are combined. 
    if mix_parents==False, the genes of only one of the parents will be selected, depending on parametesr weight_parent1 and weight_parent2
    if mix_parents!=False, the behaviour depends on the kind of data of the gene:
                        if the data in the gen is bool or str: it choses between parent1 and parent2 based on weight_parent1, and takes alternatively
                                                                one gen from the chosen parent and one gen from the other until reaching the original size.
                                                                Note that in case that each gen is a list of only one element, only one element is choen.
                                                                In case of Mendelian evolution, the size will be 2, so one gen is taken from each parent, 
                                                                but only the first in the list will be "dominant" -the other will be "recessive".
                                                                but it will also work with any size greater than 2!!!.
                        if the data in gen is float: 
                                            for each value in the list:
                                                        if mix==average: it calculates the average 
                                                        if mix_parents=normal_distribution: it samples a new value based on a normal distribution with mean=average value of the 2 genes
                                                                                            and std.dev. the half of the difference of the values of the 2 genes.
                                                                                            if the value sampled exceed the bounds, it is corrected.
                                                        
                        if the data in gen is int: 
                                            operated like in the case of float values, but rounded later.
    """


    def __init__(self,**kwargs):
        import numpy as np
        self.genotype={}
        self.age=1
        self.score="not_determined"

        #a unique finch identifier based on creation time
        self.finch_id=str(datetime.now().strftime("%Y%m%d%H%M%S%f"))+str(np.random.randint(1111,9999))

        if "mix_parents" in kwargs.keys(): mix_parents=kwargs["mix_parents"]
        else: mix_parents=False
        if "age" in kwargs.keys(): self.age=kwargs["age"]

        if "copies_per_gen" in kwargs.keys(): self.copies_per_gen=kwargs["copies_per_gen"]
        else: self.copies_per_gen=1
        
        # bound limits of genes (optional); 
        if "bounds" in kwargs.keys(): 
            self.bounds= kwargs["bounds"]
            for k in self.bounds.keys(): #the conversion to float is convenient ot prevent that a float gene value will be transformed in an int if bounds contains an int instead of a float
                if self.bounds[k][0]!=None: self.bounds[k][0]=float(self.bounds[k][0]) 
                if self.bounds[k][1]!=None: self.bounds[k][1]=float(self.bounds[k][1])   
        else: self.bounds={}

        # if the genotype is in the parameters, simply copy it
        if "genotype" in kwargs.keys(): 
            self.genotype= kwargs["genotype"]
            
        else:   #otherwise...
            #if two parents are given in the constructor, generate a new finch combining the parents:
            if "parent1" in kwargs.keys() and "parent2" in kwargs.keys():
                parent1,parent2=kwargs["parent1"],kwargs["parent2"]
                if "weight_parent1" in kwargs.keys(): weight_parents=[kwargs["weight_parent1"],1-kwargs["weight_parent1"]]
                if "weight_parent2" in kwargs.keys(): weight_parents=[1-kwargs["weight_parent2"],kwargs["weight_parent2"]]
                if "weight_parents" in kwargs.keys(): weight_parents=kwargs["weight_parents"]
                else:  weight_parents=[0.5,0.5]
                if mix_parents==False:
                    for k in parent1.genotype.keys():
                        self.genotype[k]=np.random.choice([parent1.genotype[k],parent2.genotype[k]],p=weight_parents)
                else: 
                    for k in parent1.genotype.keys():
                        #boolean or string values cannot be mixed:
                        import copy
                        g1,g2=copy.copy(parent1.genotype[k]),copy.copy(parent2.genotype[k])
                        if isinstance(g1[0],(bool,np.bool,np.bool_,str,np.str)):
                            np.random.shuffle(g1)
                            np.random.shuffle(g2)
                            if np.random.random()<weight_parents[0]: 
                                g=np.ravel((g1,g2),order="F")
                            else: 
                                g=np.ravel((g2,g1),order="F")
                            self.genotype[k]=list(g[0:len(parent1.genotype[k])])

                        else:
                            if mix_parents=="average": 
                                mixed_gene=[ np.average([float(g1),float(g2)],weights=weight_parents) for g1,g2 in zip(parent1.genotype[k],parent2.genotype[k])]
                                #mixed_gene=np.average([float(parent1.genotype[k]),float(parent2.genotype[k])],weights=weight_parents)
                                
                            elif mix_parents=="norm_distribution":
                                mixed_gene=[]
                                for g1,g2 in zip (parent1.genotype[k],parent2.genotype[k]):
                                    if g1!=g2:
                                        loc=np.mean([float(g1),float(g2)])
                                        scale=abs(loc-float(g1))
                                        new_value=np.random.normal(loc=loc,scale=scale)
                                        #the sampling from a normal distribution may lead to values out of the bounds. To prevent this:
                                        if k in self.bounds.keys():
                                            if self.bounds[k][1]!=None and new_value>self.bounds[k][1]: new_value=self.bounds[k][1]
                                            elif self.bounds[k][0]!=None and  new_value<self.bounds[k][0]: new_value=self.bounds[k][0]
                                        mixed_gene.append(new_value)
                                    else: mixed_gene.append(g1)

                            # for int values, round mixed gene to the closer integer
                            if type(parent1.genotype[k][0]) is int or isinstance(parent1.genotype[k][0],np.integer): 
                                #for i in range(len(parent1.genotype[k])):
                                #    self.genotype[k][i]=round(mixed_gene[i])
                                self.genotype[k]=[round(mixed_gene[i]) for i in range(len(parent1.genotype[k]))]
                            # for float values
                            elif type(parent1.genotype[k][0]) is float or isinstance(parent1.genotype[k][0],np.floating):  
                                self.genotype[k]=mixed_gene

            #if no parents are given but an alleles dictionary is passed, generate a new finch choosing randomly to each gene a value from the allele
            if "parent1" not in kwargs.keys() and "parent2" not in kwargs.keys() and "alleles" in kwargs.keys():
                alleles=kwargs["alleles"]
                #if flock_of_finches.discrete_gene_frequencies is passed, the new finch will choose each gene value with a probability 
                # proportional (if genes_frequencies="proportional") or inv. proportional to to each gene frequency
                if "gene_frequencies" in kwargs.keys(): gene_frequencies=kwargs["gene_frequencies"]
                else: gene_frequencies={}
                for k in alleles.keys():
                    if len(alleles[k])>1 and k in gene_frequencies.keys(): 
                        if "gene_frequencies_mode" in kwargs.keys() and kwargs["gene_frequencies_mode"]=="proportional":
                            gen_probabilities=[gene_frequencies[k][g] for g in alleles[k]]
                        else:
                            gen_probabilities=[(1-gene_frequencies[k][g]) for g in alleles[k]]
                    else: gen_probabilities=[1/len(alleles[k]) for _ in range(len(alleles[k]))]
                    gen_probabilities=gen_probabilities/np.sum(gen_probabilities) #this shouldn't be needed, but just in case...
                    self.genotype[k]=list(np.random.choice(alleles[k],self.copies_per_gen,p=gen_probabilities)) #note that if len(alleles[k])==1, this allele will always be selected

        #the finch requires a scoring function
        if "score_function" in kwargs.keys(): self.score_function=kwargs["score_function"]
        if "score_function_args" in kwargs.keys(): self.score_function_args=kwargs["score_function_args"]
        else: print("I need an score function!!!") #to do: raise an error in this case? 

        #if the memotype is give, use it; otherwise, create an empty one
        if "memotype" in kwargs.keys(): 
            self.memotype=kwargs["memotype"]
        else: self.memotype={}


    def __lt__(self,other):
        """
        Comparison method for sorting a list of finches by their id (for cases in which two scores are the same)

        Parameters
        ----------
        other : finch
            the finch to be compared with self

        Returns
        -------
        bool
            True if self.finch_id > other.finch_id, False otherwise
        """
        return self.finch_id<other.finch_id 


    #given a dictionary of possible values of each gene ("alleles"), change current finch gene value according to mutation_rate
    def mutate(self,alleles,mutation_rate):
        """
        given a dictionary of possible values of each gene ("alleles"), change current finch gene value according to mutation_rate

        Parameters
        ----------
        alleles : dict
            dictionary of genes (keys) and list of possible values
        mutation_rate : float in interval 0-1
            probability of mutating gene
        """
        for k in self.genotype.keys():
            new_gen=list(np.random.choice(alleles[k],self.copies_per_gen))
            if mutation_rate>=0 and mutation_rate<1:
                self.genotype[k]=np.random.choice([new_gen,self.genotype[k]],p=[mutation_rate,1-mutation_rate])
            else:
                print ("error: mutation rate must be in the [0,1) interval")
            self.age+=1

    def edit_genes(self,changed_genes):
        """
        sort of CRISPR to allow a finch to change its own genes. This might be useful for implementing a Lamarckian genetic algorithm.
        will only change genes already present in self.finch

        Parameters
        ----------
        changed_genes : dict
            dictionary of genes (keys) and new values
        """
        for ck in changed_genes.keys():
            if ck in self.genotype.keys():
                self.genotype[ck]=changed_genes[ck]

    def edit_memes(self,changed_memes):
        """

        modifies a finch memotype

        Parameters
        ----------
        changed_memes : dict
            dictionary of memes (keys) and new values
        """
        for ck in changed_memes.keys():
            if ck in self.memotype.keys():
                self.memotype[ck]=changed_memes[ck]

    def evaluate(self,time_limit=180,fail_score=-100,verbose=True,force_reevaluation=True):
        """
        evaluates the score of each finch
        IMPORTANT: it does not modify self.score and self.genotype; if this is required, it must be done after calling this method.
        Uses multithreading as a trick to limit the time  (time_limit) that the evaluate funciton is allowed to run;
        if the limit is overflown, the fail_score value is assigned (it can also be set to "time", and the score will be -time.time() output) 
        Note that higher scores are "better".

        Parameters
        ----------
        time_limit : int, optional, by default 180
            the time limit (seconds) for the execution of the score function before fail_score is assigned  
        fail_score : int, string, or list, optional, by default -100
            the score that will be assigned if the calculation fails 
            It can be a single value (for example: -1000.0, "failed"), or a list (for example: [-10000.0, "failed","time"] for the cases in which finch's score is multidimensional.
            If "time", the value is calculated by -1*time.time()
             Note that higher scores are "better".
        verbose : bool, default=True.
                whether or not to print how things are going...
        force_reevaluation: bool, optional
            if True, the scores will be calculated even if they are already calculated, by default False
            This is useful when the scoring function changes the genotype of the finch AFTER evaluating, or when the ambient changes (e.g., when injecting noise in the score function) 

        Returns
        -------
        dict
            a dictionary with keys: "score" and "genotype", with the values returned by the score function 
            (if scoring fails, the original genotype will be returned along with fail_score)
        """

        import multiprocessing   
        import time
        ctime=time.time()

        calculate=False
        if force_reevaluation or self.score=="not_determined": calculate=True
        if calculate:

            import timeout_decorator
            @timeout_decorator.timeout(time_limit,use_signals=False)
            def foo(genotype):
                return self.score_function(genotype,**self.score_function_args)
            try:
               results=foo(self.genotype)
            except:
               results="error"

            if results=="error":
                if verbose: print ("terminating process due to time limit in finch.evaluate after "+str(int(time.time()-ctime))+" seconds")
                with open("errors.log","a") as f:
                    f.write(str(self.genotype))
                if fail_score=="time": score=-(time.time())
                elif type(fail_score)==float: score=fail_score
                elif type(fail_score)==list: score=[-1*time.time() if s=="time" else s for s in fail_score]
                output={"score":score,"genotype":self.genotype}
                if verbose: print("with score: "+ str(score))
                if verbose: print("terminating at: "+str(time.time()))
            else:
                if type(results)in [float,np.float]: output={"score":results,"genotype":None}
                elif type(results)==list or type(results)==tuple and len(results)>1:
                    output={"score":results[0],"genotype":results[1]}
                    if len(results)>2: output["memotype"]=results[2]
                elif type(results)==dict and "score" in results.keys() and "genotype" in results.keys(): output=results

            """
            def foo(genotype,q):
                #note that the memotype is not explicitely passed, but the score function may want to do somethig wit it if it is passes in score_function_args
                score=self.score_function(genotype,**self.score_function_args)
                #print ("got score:"+str(score))
                q.put(score)
            q=multiprocessing.Queue()
            p=multiprocessing.Process(target=foo, name="Foo",args=(self.genotype,q)) 
            
            
            p.start()
            p.join(time_limit)

            if p.is_alive():
                if verbose: print ("terminating process due to time limit in finch.evaluate after "+str(int(time.time()-ctime))+" seconds") 
                with open("errors.log","a") as f:  
                    f.write(str(self.genotype))                         
                if fail_score=="time": score=-(time.time())
                elif type(fail_score)==float: score=fail_score
                elif type(fail_score)==list: score=[-1*time.time() if s=="time" else s for s in fail_score]
                output={"score":score,"genotype":self.genotype}
                if verbose: print("with score: "+ str(score))
                if verbose: print("terminating at: "+str(time.time()))
                p.terminate()
                #p.join()
            else: 
                results=q.get()
                if type(results)in [float,np.float]: output={"score":results,"genotype":None}
                elif type(results)==list or type(results)==tuple and len(results)>1: 
                    output={"score":results[0],"genotype":results[1]}
                    if len(results)>2: output["memotype"]=results[2]
                elif type(results)==dict and "score" in results.keys() and "genotype" in results.keys(): output=results 
            """
            #print ("time for evaluation: " +str(time.time()-ctime))
            #update current flinch with the new score and genotype, and optionally with whatever the finch has learned during the evaluation
            self.score=output["score"]
            #note that if score_function changes genotype, it will also be in the output, which can be used for Lamarckian evolution.
            self.genotype=output["genotype"]
            if "memotype" in output.keys(): self.memotype=output["memotype"]
            if verbose: print("exiting evaluate in finch at: "+str(time.time() ))
 
        else:
            output={"score":self.score,"genotype":self.genotype}
            if self.memotype!={}: output["memotype"]=self.memotype
            print(output["score"])
        return output 

    def hamming_distance(self,other,gene_indexes="all"):
        """
        calculates the Hamming distance (number of different genes) between this finch and other.
        note that genes containing float values are not compared, since it is very unlikely that two float values will be exactly the same.
        
        Parameters
        ----------
        other : finch
            the other finch that will be compared
        gene_indexes : list, int or str, optional, default="all"
            the indexes of the genes that will be compared.

        Returns
        -------
        int
            the number of different genes between both finches
        """
        if gene_indexes=="all": gene_indexes=range(len(self.genotype[list(self.genotype.keys())[0]]))
        elif gene_indexes=="first": gene_indexes=[0]
        elif type() ==int: gene_indexes=[gene_indexes]
        hamming_distance=0
        for gene_index in gene_indexes:
            different_genes=[self.genotype[k][gene_index]!=other.genotype[k][gene_index] for k in self.genotype.keys() if not isinstance(self.genotype[k],(float,np.floating))]
            hamming_distance+=np.sum(different_genes)
        return hamming_distance

    
    def toJson(self):
        """
        method to jsonize the finch so it can be written in a file

        Returns
        -------
        string
            jsonized finch (the scoring function is not included)
        """
        import json
        import copy

        mydict=copy.copy(vars(self)) #work with a copy since otherwise "score_function" will also be deleted from current object
        del(mydict["score_function"]) #score function is not jsonable, so delete it
        del(mydict["score_function_args"])
        return json.dumps(mydict,cls=NpEncoder,indent=4)




class flock_of_finches():
    """
    class that define the flock_of_finches (the population of finches that evolves)
     requires: 
     a score function, that will be passed to each finch.
     alleles:  a dictionary with possible values of each gene 
     to use it with slurm, it also requires:
    The generation of an flock_of_finches can be created: 
     1) from the list of json files containing each finch's genome
     2) from the population (number of finchs), assigning random values of each finch gene according to the possible values in "alleles"
    Note that higher scores are better

    """

    def __init__(self,**kwargs):  

        
        """
        constructor for flock_of_finches class

        Parameters
        ----------
        name : name of the flock of finches; if absent, will be created from data and time
        alleles : dict
            dictionary with possible values of each gene 
        bounds : dict
            optional dictionary with lower and upper bounds of each gene
        copies_per_gen : number of copies of each gene, considering that all will be used during reproduction but only the first element will be expressed
            (to implement mendelian evolution)
        score_function : function
            function that evaluates the fitness of a finch
        score_function_args : dict or list
            arguments to be passed to the score function
        valid_score_range : list or tuple
            scores outside this range will be considered invalid for some statistics
        population : int
            number of finches in the flock
        generation_number : int
            number of the generation, increased after each reproduction/hunting cycle.
        falcons : int
            number of falcons per generation (for now, only used for logging)
        generation : list
            optional list of json files containing each finch's genome
        initial_mix : bool
            if True, the first generation will be a mixed generation created by randomly mixing the genes of two randomly chosen finches.

        Returns
        -------
        flock_of_finches
            a new flock_of_finches object

        Notes
        -----
        The generation of an flock_of_finches can be created: 
         1) from the list of json files containing each finch's genome
         2) from the population (number of finchs), assigning random values of each finch gene according to the possible values in "alleles"
        Note that higher scores are better

        """

        if "name"  in kwargs.keys():                self.name=kwargs["name"]
        else:                                       self.name=time.strftime("%Y%m%d-%H%M%S")
        if self.name=="":                           self.name=time.strftime("%Y%m%d-%H%M%S") 
        if "alleles" in kwargs.keys():              self.alleles=kwargs["alleles"]
        if "copies_per_gen" in kwargs.keys():       self.copies_per_gen=kwargs["copies_per_gen"]
        else:                                       self.copies_per_gen=1
        if "score_function" in kwargs.keys():       self.score_function=kwargs["score_function"]
        if "score_function_args" in kwargs.keys():  self.score_function_args=kwargs["score_function_args"]
        if "population" in kwargs.keys():           self.population=kwargs["population"]
        else:                                       self.population=100
        if "generation_number" in kwargs.keys():    self.generation_number=kwargs["generation_number"]
        else:                                       self.generation_number=1         
        if "bounds" in kwargs.keys():               self.bounds= kwargs["bounds"] # bound limits of genes (optional)
        else:                                       self.bounds={}
        if "falcons" in kwargs.keys():              self.falcons= kwargs["falcons"]
        else:                                       self.falcons=1
        if "valid_score_range" in kwargs.keys():    self.valid_score_range=kwargs["valid_score_range"]
        else:                                       self.valid_score_range=[]

        if "generation" in kwargs.keys():#if a generation is given, use it to create flock_of_finches
            self.generation=[finch(alleles=self.alleles,bounds=self.bounds,score_function=self.score_function, score_function_args=self.score_function_args ,genotype=json.loads(g)["genotype"]) for g in kwargs["generation"] ]  
        else:  #otherwise, create finches by randomly selected genotype from alleles; if "initial_mix" is True, it will create a mixed generation
            if "initial_mix" in kwargs.keys() and kwargs["initial_mix"]==True: 
                self.generation=[]
                for i in range (0,self.population):
                    parents=[finch(alleles=self.alleles,bounds=self.bounds,copies_per_gen=self.copies_per_gen,score_function=self.score_function,score_function_args=self.score_function_args) for i in range(0,2)]
                    new_finch=finch( parent1=parents[0],parent2=parents[1],score_function=self.score_function,score_function_args=self.score_function_args,mix_parents="norm_distribution",bounds=self.bounds)
                    self.generation.append(new_finch)   
            else:
                self.generation=[finch(alleles=self.alleles,bounds=self.bounds,copies_per_gen=self.copies_per_gen,score_function=self.score_function,score_function_args=self.score_function_args) for i in range (0,self.population)] 

        #in any case, set the initial diversity entropy and the dictionary iwth frequency of discrete genes
        self.discrete_gene_frequencies=self.calculate_discrete_gene_frequencies(fraction=1.0,gene_indexes="all")
        self.initial_diversity_entropy=self.diversity_entropy(fraction=1.0,gene_indexes="all",freqs=self.discrete_gene_frequencies)
    
    def calculate_combinations(self):
        """

        method to calculate the number of different unique combinations (only valid for hard-mixing: it cannot calculate the number of intermediate possible values of a gene)

        Returns
        -------
        int
            the number of combinations
        """
        return np.prod([len(self.alleles[key]) for key in self.alleles.keys()])  

    def sort_by_scores(self,index=0):

        """
        sort the finches in a generation based on their scores. 

        Parameters
        ----------
        index : int, optional
            in case scores is a list, use "index" element for sorting, by default 0

        Returns
        ----------
        Bool : if any of the scores cannot be sorted, it does nothing and returns False.
        """
        generation_scores=[b.score for b in self.generation]
        #if scores is a list, use the first element to sort
        generation_scores_0=[gs[index] if isinstance(gs,(list,np.ndarray)) else gs for gs in generation_scores  ]
        #only sort if there is no item in generation_scores_0 that does not allow sorting, 
        sortable=all([isinstance(cc,(float,np.floating,int,np.int64)) for cc in generation_scores_0])
        if sortable: 
            self.generation=[b for _,b in sorted(zip(generation_scores_0,self.generation), reverse=True)]

        return sortable


    def save_generation(self,file_name="",include_generation_number=False):
        """

        method to save the generation to a json file

        Parameters
        ----------
        file_name : str, optional, by default ""
            the name of the file 
        include_generation_number : bool, optional, by default False
            include in the name of the file the generation number

        Returns
        -------
        str
            the name of the file generated
        """
        import json
        if file_name=="": file_name=self.name
        to_write={"name":self.name,"alleles":self.alleles,"population":self.population,"generation_number":self.generation_number,
                  "generation":[finch.toJson() for finch in self.generation]}
        json_text=json.dumps(to_write,indent=4)
        if include_generation_number: name_of_file=file_name+"-"+str(self.generation_number)
        name_of_file=file_name
        with open(name_of_file,"w") as f: f.write(json_text)
        return name_of_file

    def Lokta_Volterra_new_generation(self,
                        reproduction_rate=0.75,
                        inmigration_rate=0.25,
                        predation_rate=0.22,
                        falcon_death_rate=1.4,
                        falcon_feeding_yield=0.2,
                        min_finches=100,
                        min_falcons=1,
                        inmigration_mode="inv_freq",
                        mix_parents=False,
                        probability_factor=1,
                        score_index=0,
                        hunt_criteria=["age","entropy"],
                        hunt_criteria_combination="product",
                        depredation_policy="mean",
                        age_halving_life_expectation=10,    
                        ):

        """
        method to create a new generation of finches based on a "Lokta Volterra" ecosystem with prey(finches) and predators (falcons):

        The original Lotka-Volterra equations are (https://en.wikipedia.org/wiki/Lotka–Volterra_equations):

        dx/dt=reproduction_rate * finches - predation_rate * finches * prey  (+ inmigration_rate * finches)
        dy/dt=(falcon_feeding_yield * predation_rate) * finches * falcons - falcon_death_rate * falcons

        The model is modified because the probabilities of reproducing and being hunted are different for finches:
        - The probability of reproducing is proportional to the score of the finches. Depends on the probability_factor term
            (the ratio between the probability of reproducing of the finch with best score and a finch with a media score).
        - The probability of being hunted is inversely proportional to the contribution to the overall entropy of the finches 
            (the ratio between the probability of the less contributing score and the median contributin score is probability_factor) 
            and/or the age of the finches (the probability of being hunted of a finch with age=age_halving_life_expectation doubles
            the probabilities of the younger finches)
        

        Parameters
        ----------
        reproduction_rate : float, optional, by default 2
            reproduction rate of the finches
        predation_rate : float, optional, by default 0.1
            predation rate of the finches
        falcon_feeding_yield : float, optional, by default 0.1 (should be <1.0)
            due to 2nd law of thermodynamics, not all energy in a prey is transformed in a hunter. 
        falcon_death_rate : float, optional, by default 1.0
            the rate of death of the falcons.
        inmigration_rate : float, optional, by default 0.1
            the rate of new fresh finches introduced in every generation. The sum with reproduction rate condition the 
            increase of the number of finches in new generation, but the origin of new finches is diffeerent (aleatory creation, not breeding) 
        mix_parents : bool, optional, by default False            
            if True, it will mix the parents of the finches
        probability_factor : float, optional, by default 0.3                       
            the probability of a finch being selected for breeding or hunting is determined as: 1/(exp(-x^2)) if x>0, 1/(exp(x^2)) otherwise (equivalent to: exp(sign(x)*x^2)) ), 
            where x = probability_factor *  normalized score or entropy contribution of the finch.
            probability_factor = 1.0 would "compensate" the smaller frequency of best performing finches with respect to "normal" finches, (and penalyze worst performing finches)
        score_index : int, optional, by default 0
            in case scores is a list, use "index" element for sorting, by default 0
        hunt_criteria : list, optional, by default ["age","entropy"]
            the criteria for the finches to be hunted. If "score-x" is found, will use the score in index x to calculate, assuming that 
            larger x mean smaller probability of being hunted; if "-score-x" is found, will use -score.
        hunt_criteria_combination: string, "sum" or "product, by default: "product"
            if there are different criteria to be hunted, how they are combined. "product" correspond to "and", so probability of being hunted is small if one criteria is high but
            the other is low. "sum" correspond to "or", so probability is large if any criteria is high, and small if both are low.
        depredation_policy : str, "child first","elder first","median" or "mean", by default: "mean",
            when any score is used in hunt_criteria, the new flinches (that has not yet been scored) are assigned a depredation probability
            equal as max. of old finches ("child first"), min. of old finches ("elder first"), median or mean of old finches
        age_halving_life_expectation : int, optional, by default 10            
            the age halving life expectation of the finches
        min_finches :  int, optional, by default: 20
            the minimum number of finches (to prevent extiction).
        min_falcons :  int, optional, by default: 1
            the minimum number of falcons (to prevent extiction).
        inmigration_mode: str, "inv_freq", "freq", "", by default "inv_freq"
            the criteria that will be used for choosing inmigrants genes values; if "inv_freq", probability of choosing less frequent gene values in current population is higher.
            If "freq", probability of choosing more frequent gene values in current population is higher; if "", it will not use this criteria 

        Returns
        -------
        list
            the new generation
        """

        #calculate the number of finches that will be hunted, reproduce or added
        n_reproducing=round(reproduction_rate*len(self.generation)) 
        n_hunted=   round(predation_rate * self.falcons * len(self.generation) )
        n_inmigrants= round(inmigration_rate*len(self.generation))
        n_failures=[(f.score[score_index]<self.valid_score_range[1] and f.score[score_index]>self.valid_score_range[0]) for f in self.generation].count(False)

        #ensure that the limits are respected
        if self.population-n_hunted+n_reproducing+n_inmigrants<min_finches: n_hunted=self.population+n_reproducing+n_inmigrants-min_finches
        print ("n_reproducing:"+str(n_reproducing)) #borrame
        print ("n_hunted:"+str(n_hunted))
        print ("n_inmigrants:"+str(n_inmigrants))
        print ("n_failures:"+str(n_failures))
        print ("self.population"+str(self.population))

        #since failures will be killed, increase n_reproducing and n_hunting to account for it
        n_reproducing+=round( n_failures*reproduction_rate/(reproduction_rate+inmigration_rate ))
        if n_reproducing%2!=0: n_reproducing+=1 #ensure this is even 
        n_inmigrants+=n_failures - round( n_failures*reproduction_rate/(reproduction_rate+inmigration_rate ))
        n_hunted+=n_failures
        print("after including failures:")
        print ("n_reproducing:"+str(n_reproducing)) #borrame
        print ("n_hunted:"+str(n_hunted))
        print ("n_inmigrants:"+str(n_inmigrants))
        print ("n_failures:"+str(n_failures))
        print ("self.population"+str(self.population))

        #update the number of falcons
        self.falcons+=   (np.round( falcon_feeding_yield*n_hunted - self.falcons*falcon_death_rate))
        if self.falcons<min_falcons: self.falcons=min_falcons # prevent extinction!

        #increase the age of current finches:
        for f in self.generation: f.age+=1
        
        #selecting finches that will reproduce 
        #determine scores, their normalized values, and the reproduciton probability according to the scores
        if isinstance(self.generation[0].score,list) or isinstance(self.generation[0].score,np.ndarray): 
            #in case scores is a list, only the first element will be used for ranking
            generation_scores=[b.score[score_index] for b in self.generation] 
        else: 
            generation_scores=[b.score for b in self.generation]

        normalized_generation_scores=outlier_proof_statistics(generation_scores,return_stats="standarized_values",exclude_limit=self.valid_score_range)["standarized_values"]
        #those finches that failed to score will not reproduce
        generation_scores_reproduction_probability=[ (np.exp(np.sign(normalized_score) * (normalized_score*probability_factor)**2)) if np.isnan(normalized_score)!=True else 0 for normalized_score in normalized_generation_scores ]
        generation_scores_reproduction_probability=generation_scores_reproduction_probability/np.sum(generation_scores_reproduction_probability) #scale to sum 1
        reproducing=np.random.choice([i for i in range(len(self.generation))],n_reproducing,p=generation_scores_reproduction_probability,replace=True) #a finch can be chosen twice
        
        #add the new finches; it is important to do it before hunting because n_hunt may exceed the population

        #generate pairs without repetition
        shuffle_reproducing=np.random.permutation(reproducing) 
        parent_list=[[self.generation[shuffle_reproducing[i]],self.generation[shuffle_reproducing[i+1]]] for i in range(0,len(shuffle_reproducing),2)]
        aggregated_probabilities=[p[0].score[score_index] + p[1].score[score_index] for  p in parent_list ]
        #the number of childs of each couple depends on the sum of the scores of the parents: 
        # Those with sum of scores larger than mean + 1 std. deviation will have 3
        # Those with sum of scores smaller than mean -1 std. deviation will have 1
        # Those with sum of scores between mean -1 std. deviation and mean + 1 std. deviation will have 2
        # to prevent problems with very small scoring function values for finches that failed to score, outlier-proof mean and std dev. are used:
        median=outlier_proof_statistics(aggregated_probabilities,return_stats=["median"])["median"]
        childs=[]
        for i,p in enumerate(parent_list):
            if aggregated_probabilities[i]>median: childs_per_couple=2
            else: childs_per_couple=2 
            
            for j in range(childs_per_couple):
                childs.append(finch( parent1=p[0],parent2=p[1],
                                    score_function=self.score_function,
                                    score_function_args=self.score_function_args,
                                    mix_parents=mix_parents,
                                    bounds=self.bounds) ) 

        #add the  inmigrants:
        if inmigration_mode in ["inv_freq","freq"]:
            if self.discrete_gene_frequencies==None: self.calculate_discrete_gene_frequencies(fraction=1.0,gene_indexes="all")
            gene_frequencies=self.discrete_gene_frequencies
            if inmigration_mode=="inv_freq": gene_frequencies_mode=""
            else:  gene_frequencies_mode="proportional"
        else:  gene_frequencies={}; gene_frequencies_mode=""
        inmigrants=[finch(alleles=self.alleles,
                          copies_per_gen=self.copies_per_gen,
                          score_function=self.score_function,
                          score_function_args=self.score_function_args, 
                          bounds=self.bounds,
                          gene_frequencies=gene_frequencies,
                          gene_frequencies_mode=gene_frequencies_mode) 
                          for _ in range(0,n_inmigrants)]
 
        #if any score will be used for determining hunting, as neither childs nor inmigrants have been scored, it is neccessary to assign the probability of being hunted to old members
        #before updating self.generation (score is not yet determined for new members). 
        #The probability of being hunted of new members will be updated later.
        if hunt_criteria_combination=="product": depredated_probability=np.ones(len(self.generation))
        elif hunt_criteria_combination=="sum": depredated_probability=np.zeros(len(self.generation))
        if any([a.find("score")>-1 for a in hunt_criteria]):
            for h in hunt_criteria:
                if h.find ("score")>-1:
                    if h.startswith("-score"): sign=-1
                    else: sign=1
                    score_index=int(h.split("score-")[1])
                    generation_hunting_scores=[f.score[score_index] for f in self.generation]
                    standardized_generation_hunting_scores=outlier_proof_statistics(generation_hunting_scores,score_values=generation_scores,return_stats=["standarized_values"],exclude_limit=self.valid_score_range)["standarized_values"]
                    #the probability of being hunted of finches that failed to score is 1000 times larger than for the rest,to ensure they are killed first
                    generation_scores_hunting_probability=[np.exp(-1*sign*np.sign(normalized_score) * (normalized_score*probability_factor)**2) if np.isnan(normalized_score)!=True else 1000 for normalized_score in standardized_generation_hunting_scores]
                    generation_scores_hunting_probability=generation_scores_hunting_probability/np.sum(generation_scores_hunting_probability) #scale to sum 1
                    if hunt_criteria_combination=="product": depredated_probability=depredated_probability*generation_scores_hunting_probability
                    elif hunt_criteria_combination=="sum": depredated_probability=depredated_probability+generation_scores_hunting_probability
            depredated_probability=depredated_probability/np.sum(depredated_probability)  #scale to sum 1 


        #update generation with the new members
        self.generation=self.generation+childs+inmigrants

        #selecting the finches that will die 
        #assing depredated probability of new members; if any score is used for this, new members have not yet calculate it, so its value will be assinged from
        # current deprecated_probability depending on deprecation_policy
        if len(depredated_probability)< len(self.generation):
            extra_members=len(self.generation)-len(depredated_probability)
            if depredation_policy=="child first":
                depredated_probability_for_new_members= np.max(depredated_probability)
            elif depredation_policy=="elder first":
                depredated_probability_for_new_members=np.min(depredated_probability)
            elif depredation_policy=="median":
                depredated_probability_for_new_members=np.median(depredated_probability)
            elif depredation_policy=="mean":
                depredated_probability_for_new_members=np.mean(depredated_probability)
            depredated_probability=np.hstack((depredated_probability, np.array([depredated_probability_for_new_members]*extra_members)))
            depredated_probability=depredated_probability/np.sum(depredated_probability)

        #calculate the entropy contribution of each finch in the generation, their normalized values, and the probability of being hunt
        if "entropy" in hunt_criteria:
            self.calculate_discrete_gene_frequencies()
            entropy_contributions=[self.change_diversity_entropy(changed_finch=f,fraction=1.0,gene_indexes="all",freqs=self.discrete_gene_frequencies) for f in self.generation]
            normalized_entropy_contributions=outlier_proof_statistics(entropy_contributions,return_stats="standarized_values")["standarized_values"]
            #the probability of being hunted of finches that failed to score is 1000 times larger than for the rest, to kill them first
            #note that multiplication with -1 makes more possible to hunt finches with lower entropy (with negative normalized entropy contributions)
            generation_entropy_hunted_probability=np.array([ np.exp(-1*np.sign(normalized_entropy) * (normalized_entropy*probability_factor)**2)  if np.isnan(normalized_entropy)!=True else 100 for normalized_entropy in normalized_entropy_contributions  ])
            generation_entropy_hunted_probability=generation_entropy_hunted_probability/np.sum(generation_entropy_hunted_probability) #scale to sum 1
            if hunt_criteria_combination=="product": depredated_probability=depredated_probability*generation_entropy_hunted_probability
            elif hunt_criteria_combination=="sum": depredated_probability=depredated_probability+generation_entropy_hunted_probability
       
        #calculate the probabilities of being hunt by age
        if "age" in hunt_criteria:
            generation_ages=[b.age for b in self.generation]
            if len(set(generation_ages))==1: generation_age_hunted_probability=[1/len(generation_ages)]*len(generation_ages) #if all entries have the same age, the probability is the same for all
            else:
                generation_age_hunted_probability=np.array([2**(age-min(generation_ages))/age_halving_life_expectation for age in generation_ages])
                generation_age_hunted_probability=generation_age_hunted_probability/np.sum(generation_age_hunted_probability)
            if hunt_criteria_combination=="product": depredated_probability=depredated_probability*generation_entropy_hunted_probability
            elif hunt_criteria_combination=="sum": depredated_probability=depredated_probability+generation_entropy_hunted_probability

        #normalize depredated probability (make sure they sum 1)
        depredated_probability=depredated_probability/np.sum(depredated_probability)
     
        hunted=np.random.choice([i for i in range(len(self.generation))],n_hunted,p=depredated_probability,replace=False)
        
        #update the generation:
        self.generation=[self.generation[i] for i in range(len(self.generation)) if i not in hunted]
        self.generation_number=self.generation_number+1
        self.population=len(self.generation)
        self.discrete_gene_frequencies=self.calculate_discrete_gene_frequencies(fraction=1.0,gene_indexes="all")


        

    def fixed_population_new_generation(self,elite_rate=0.05,reproduction_rate=0.5,mix_parents=False,probability_factor=0.5,promiscuity=5,score_index=0,diversity_entropy_kept=0,inmigration_mode="inv_freq"):
        """
        method to create a new generation for the current flock_of_finches keeping the number of finches constant, given the fraction of finches that will be preserved as elite,
        the fraction of finches that will be obtained as descendents of current generation's parents, and completing the new generation with inmigrants.
        The probability of choosing two parents for a future finch is done by a normal distribution according to their scores, so best-scoring parents are selected more often.
        Optionally, the initial diversity entropy of the population will be preserved by substituting resulting finches of lower entropy with finches of higher entropy 
        
        Parameters
        ----------
        elite_rate : float, optional
            is the fraction of best-scoring finches that will be preserved in next generation, by default 0.05
        reproduction_rate : float, optional
            is the fraction of finches of next generation that will be obtained as descendents of current generation's parents, by default 0.5.
            therefore, (1 - reproduction_rate - elite_rate) is the fraction of "inmigrants" in next generation.
        mix_parents : bool, optional
            wether to mix genes from parents or simply choose one gene, by default False
        probability_factor : float, optional, by default 0.5                       
            the probability of a finch being selected for breeding or hunting is determined as: 1/(exp(-x^2)) if x>0, 1/(exp(x^2)) otherwise (equivalent to: exp(sign(x)*x^2)) ), 
            where x = probability_factor *  normalized score or entropy contribution of the finch.
            probability_factor = 1.0 would "compensate" the smaller frequency of best performing finches with respect to "normal" finches, (and penalyze worst performing finches)
        promiscuity : int, optional, by default 5
            when choosing two parents, "promiscuity" candidates for being the two parents will be considered. 
            The first parent will be chosen randomly from promiscuity subset. The second parent will be the one 
            among the remaining candidates that is less similar (=smaller number of equal values in their genes) to the first. 
            Therefore, the larger "promiscuity", the more variability will be enforced in next generation.
        score_index : int, optional, by default 0
            if the score is a list, the index that will be used for comparing
        diversity_entropy_kept : float, optional
            the fraction of diversity entropy in the 1st generation that will be kept after the next generation, by default 0.
            If 0, it will not be calculated. If (float), attempts will be made to keep the diversity entropy over 
            diversity_entropy_kept*initial_diversity_entropy; this is done by iteratively replacing a random finch (inmigrant or child) 
            with a new inmigrant finch only if the substitution increases the diversity entropy. The procedure is repeated 
            until the goal is achieved or the limit of iterations (100 times the number of childs and inmigrants) is reached.
        inmigration_mode: str, "inv_freq", "freq", "", by default "inv_freq"
            the criteria that will be used for choosing inmigrants genes values; if "inv_freq", probability of choosing less frequent gene values in current population is higher.
            If "freq", probability of choosing more frequent gene values in current population is higher; if "", it will not use this criteria 

        Returns
        -------
        list
            the new generation
        """

        self.generation_number+=1
        #determine number of elites, inmigrants and childs
        n_elites=round(self.population*elite_rate)   
        n_childs=round(self.population*reproduction_rate)  
        n_inmigrants=round(self.population-n_childs-n_elites) 
        print (n_elites) #borrame
        #sort finches by score
        a=self.sort_by_scores(index=score_index)
        #select finches in elites
        elites=self.generation[:n_elites]      
        for e in elites: e.age+=1

        #create inmigrants
        if inmigration_mode in ["inv_freq","freq"]:
            if self.discrete_gene_frequencies==None: self.calculate_discrete_gene_frequencies(fraction=1.0,gene_indexes="all")
            gene_frequencies=self.discrete_gene_frequencies
            if inmigration_mode=="inv_freq": gene_frequencies_mode=""
            else:  gene_frequencies_mode="proportional"
        else:  gene_frequencies={}; gene_frequencies_mode=""
        inmigrants=[finch(alleles=self.alleles,
                          copies_per_gen=self.copies_per_gen,
                          score_function=self.score_function,
                          score_function_args=self.score_function_args, 
                          bounds=self.bounds,
                          gene_frequencies=gene_frequencies,
                          gene_frequencies_mode=gene_frequencies_mode)
                          for _ in range(0,n_inmigrants)]

        #select finches for reproduction based on scores
        if isinstance(self.generation[0].score,list) or isinstance(self.generation[0].score,np.ndarray): 
            #in case scores is a list, only the first element will be used for ranking
            generation_scores=[b.score[score_index] for b in self.generation] 
        else: 
            generation_scores=[b.score for b in self.generation]
        mean_score,std_dev_score=outlier_proof_statistics(generation_scores,exclude_limit=self.valid_score_range)
        normalized_generation_scores=outlier_proof_statistics(generation_scores,return_stats="standarized_values",exclude_limit=self.valid_score_range)["standarized_values"]
        generation_scores_reproduction_probability=[ np.exp(np.sign(normalized_score) * (normalized_score*probability_factor)**2) if np.isnan(normalized_score)!=True else 0 for normalized_score in normalized_generation_scores  ]
        generation_scores_reproduction_probability=generation_scores_reproduction_probability/np.sum(generation_scores_reproduction_probability) #scale to sum 1

        list_of_parents=[]
        for i in range(0,n_childs):
            parent_candidates=np.random.choice(self.generation,size=promiscuity,replace=True,p=generation_scores_reproduction_probability)
            disimilarities_of_parents=[parent_candidates[0].hamming_distance(p) for p in parent_candidates]
            #the most similar to the first one (the argmin(desimilarities_of_parents)th element) is the one used for calculating disimilarities (the first element)
            #the less similar to the first one is the second parent selected
            list_of_parents.append([parent_candidates[np.argmin(disimilarities_of_parents)],parent_candidates[np.argmax(disimilarities_of_parents)]])
        childs=[ finch(parent1=parents[0],parent2=parents[1],score_function=self.score_function,score_function_args=self.score_function_args,mix_parents=mix_parents,bounds=self.bounds) for parents in list_of_parents   ]

        self.generation=elites+inmigrants+childs
        self.discrete_gene_frequencies=self.calculate_discrete_gene_frequencies(fraction=1.0,gene_indexes="all")

        if diversity_entropy_kept!=0:
            import time
            stime=time.time()
            #calculate the goal diversity entropy
            goal_diversity_entropy=self.initial_diversity_entropy*diversity_entropy_kept
            current_diversity_entropy=self.diversity_entropy(fraction=1.0,gene_indexes="all",freqs=self.discrete_gene_frequencies)

            #calculate the entropy contribution of each finch in the generation
            entropy_contributions=[self.change_diversity_entropy(changed_finch=f,fraction=1.0,gene_indexes="all",freqs=self.discrete_gene_frequencies) for f in self.generation]

            #entropy contributions are the change in the entropy when removing the finch (negative when the finch contributed to entropy), 
            #so finch with the largest (less negative or more positive) entropy contribution value is the 
            #one that, if removed, will affect least to variability 
            less_entropic_finch_orders=np.argsort(entropy_contributions)
            less_entropic_finch_orders=list(reversed(less_entropic_finch_orders))
            for i,f in enumerate(less_entropic_finch_orders):
                if current_diversity_entropy>goal_diversity_entropy:
                    #print("diversity entropy:"+str(current_diversity_entropy)+" requested: "+str(goal_diversity_entropy))
                    #print("goal achieved!")
                    break
                #print("replacing finch #"+str(i)+" with entropy contribution "+str(entropy_contributions[f]))
                replaced_finch=self.generation[f]
                freqs_wo_f=self.change_discrete_gene_frequencies(f,fraction=1.0,gene_indexes="all",remove=True,freqs=self.discrete_gene_frequencies)
                for j in range(0,1000):
                    new_inmigrant=finch(alleles=self.alleles,copies_per_gen=self.copies_per_gen,score_function=self.score_function,score_function_args=self.score_function_args, bounds=self.bounds)
                    self.generation[f]=new_inmigrant
                    new_freqs=self.change_discrete_gene_frequencies(new_inmigrant,fraction=1.0,gene_indexes="all",remove=False,freqs=freqs_wo_f)  
                    new_diversity_entropy=self.diversity_entropy(fraction=1.0, gene_indexes="all", freqs=new_freqs)
                    if new_diversity_entropy<current_diversity_entropy: 
                        #print ("entropy does not raise, old: "+str(current_diversity_entropy)+", new: "+str(new_diversity_entropy)+" let's try again..."+str(j))
                        continue #new attempt
                    else:
                        #print ("success!, old entropy:"+str(current_diversity_entropy)+", new entropy"+str(new_diversity_entropy)+", goal: "+str(goal_diversity_entropy))
                        current_diversity_entropy=new_diversity_entropy
                        self.discrete_gene_frequencies=new_freqs
                    if j==999: 
                        #print("maximum number of attempts reached, finch not replaced")
                        self.generation[f]=replaced_finch
                        break
            print ("time needed for keeping entropy: "+str(time.time()-stime))

    def add_generation(self,other_flock):
        """
        add the generation of finches in "other_flock" to current generation, and sort by their scores

        Parameters
        ----------
        other_flock : flock_of_finches
        """
        self.generation+=other_flock.generation
        _=self.sort_by_scores()

    def evaluate_generation(self,verbose=False,nodes=1,interval=[0,"all"],scores_file="",slurm_temp_folder="slurm_jobs",time_limit=300,fail_score="time",scores_folder="scores",force_reevaluation=True):
        """calculate the scores of each finch in the population (generation) in the specified interval, 
        and modify each finch's score or even each finch's genotype (this only if the scoring function returns also a modified genotype)

        Parameters
        ----------
        verbose : bool, optional
            wether to print information of what is going on, by default False
        nodes : int, optional
            nodes in which it will be calculated, by default 1
            if nodes==slurmxx, will use slurm to run in parallel, requesting xx processor sin each run
            if nodes==daskxx, will try to use dask (currently not working) 
        interval : list, optional
            the list of finches of which the scores will be calculated, by default [0,"all"]
        scores_file : str, optional
            if specified, the name of the file that will be used to store the files, 
            by default "" (scores will not be saved)
        scores_folder : str, optional
            the name of the folder in which scores_file will be created, by default "scores"
        slurm_temp_folder : str, optional
            in case of using slurm, the name of the folder in which the slurm scripts, pickle files and log and error files will be temporarily stored
        time_limit : int, optional
            the time in seconds that each work will be allowed to run, by default 300.
            When using slurm, the slurm script is scheduled to run time_limit+60 seconds.
        fail_score : float, str, or list, optional
            the score that will be attributed to finches in which the calculation of the scores failed, by default "time".
            It can be a single value (for example: -1000.0, "failed"), or a list (for example: [-10000.0, "failed","time"] for the cases in which finch's score is multidimensional.
            If "time", the value is calculated by -1*time.time()
        force_reevaluation: bool, optional
            if True, the scores will be calculated even if they are already calculated, by default False
            This is useful when the scoring function changes the genotype of the finch AFTER evaluating, or when the ambient changes (e.g., when injecting noise in the score function) 
        """
        
        start_time=time.time()
        if interval==[0,"all"] or interval==[] or interval=="all":evaluate_finches=self.generation
        else: evaluate_finches=self.generation[interval[0]:interval[1]]
        
        # no parallel execution at all:
        if nodes==1:
            i=0 
            outputs=[]
            for f in evaluate_finches:
                i+=1
                if verbose: print ("evaluation #"+str(i)+" out of "+str(len(self.generation))+"    ",end="\r") 
                outputs.append(f.evaluate(time_limit=time_limit,fail_score=fail_score,force_reevaluation=force_reevaluation))
                
        #do some distributed memory paralellism using slurm queue system.
        elif nodes[0:5]=="slurm":
            import submitit
            #submitit library has to be transformed to make it work:
            #modify file: ...python3.8/site-packages/submitit/slurm/slurm.py to set use_srun=False (instead of True)
            #modify file: ...python3.8/site-packages/submitit/core/utils.py   to set self.task_id  always 0 in line 55: self.task_id = 0 #task_id or 0 
            #also imposed by submitit is that any module needed by the score_function that is not in the default imports_path MUST be imported inside the function
            #after instruction: sys.path.insert(0,"path to module")
            n_cpu=nodes[6:]
            executor = submitit.AutoExecutor(folder=slurm_temp_folder)
            executor.update_parameters(timeout_min=int(1+time_limit/60),ntasks_per_node=int(n_cpu))  
            jobs=[]
            with executor.batch():
                for f in evaluate_finches:
                    job=executor.submit(f.evaluate,time_limit,fail_score,verbose,force_reevaluation)
                    #job=executor.submit(f.evaluate,time_limit,fail_score)
                    jobs.append(job)
            #jobs = [executor.submit(f.evaluate,time_limit,fail_score)  for f in self.generation[interval[0]:interval[1]]  ]
            if verbose: print("jobs sumitted... waiting?")
            outputs = [job.results() for job in jobs]
            #clean-up temporary files
            #for f in os.listdir(os.getcwd()+"/"+slurm_temp_folder+"/"):
            #    if f[0]!="." and time.time()-getmtime(os.getcwd()+"/"+slurm_temp_folder+"/"+f)>2*time_limit: #only delete files accessed more than 2 times the time limit before
            #        os.remove(os.getcwd()+"/"+slurm_temp_folder+"/"+f) 

        #another attempt to parallelize this using dask, but I have not been able to make it work
        elif  nodes[0:4]=="dask":
            num_cpus=int(nodes[4:])
            from dask.distributed import Client, LocalCluster
            cluster= LocalCluster()
            client = Client(cluster,n_workers=nodes)

            pack_size=int(self.population/nodes)
            import socket
            #@ray.remote(num_cpus=14)
            
            #def evaluate_remote(j):
            #    scores=[]
            #    for b in self.generation[j*pack_size:(j+1)*pack_size]: scores.append(b.evaluate())
            #    return scores

            #futures=[evaluate_remote.remote(k) for k in range(0,nodes)]
            #score_packs=ray.get(futures)
            #all_scores=[]
            #for sp in score_packs:all_scores+=sp            
            
            @dask.delayed
            def evaluate_parallel(j):
                print ("inside dask.delayed in "+str(socket.gethostname()))#borrame
                return self.generation[j].evaluate()
            outputs=[evaluate_parallel(j) for j in range(0,len(self.generation))]
            print ("computing in parallel?")
            dask.compute(outputs)

        #update scores and genotype of current finches
        for o,f in zip(outputs,self.generation):
            f.score=o[0]["score"]
            print (f.score)#borrame
            f.genotype=o[0]["genotype"]
        if verbose: print ("jobs finished")
 

        #save results in a file
        if scores_file!="":
            if not os.path.exists(scores_folder): os.makedirs(scores_folder)
            with open(scores_folder+"/"+scores_file+str(interval[0])+"-"+str(interval[1]),"w") as f: 
                f.write(json.dumps([finch.toJson() for finch in evaluate_finches] ))

        #sort finches in generation by the first value of the scores
        _=self.sort_by_scores()

        self.evaluation_time=start_time-time.time()

    def gene_value_frequency(self,gene,value,interval="all",gene_indexes=0):
        """
        returns the frequency of a gene value (or value[0]<value<value[1])

        Parameters
        ----------
        gene : string
               the name of the gene (key of the genotype dictionary)  that will be investigated. 
               in case the gene value is a list, only the first element (the fenotype) will be compared
        value : float, string or list
                the value that will be used to compare. 
                gene value == value or value[0]< gene value < value[1] will be used in the comparison
                depending if value is a single value or a 2-element list.
        interval : list or str
                optional, by default "all"
                the collection of finches in the generation for which the gene frequency will be evaluated
        gene_index : int, optional
                the index of the genes in the genotype dictionary (for cases in which there are fenotypes and genotypes), by default 0 (use the fenotype)
        """
        
        if isinstance(gene_indexes,(int,np.integer)):gene_indexes=[gene_indexes]
        if interval=="all":interval=range(0,len(self.generation))

        n=len(interval)*len(gene_indexes)
        if isinstance(value,(list,np.ndarray)):
            if len(value)==2 and isinstance(value[0],(int,float)):
                min_value,max_value=np.min(value),np.max(value)
                return np.sum([[ self.generation[i].genotype[gene][gene_index]<=max_value and self.generation[i].genotype[gene][gene_index]>=min_value for i in interval ]for gene_index in gene_indexes])/n 
            else:
                return np.sum([[ (self.generation[i].genotype[gene][gene_index] in value) for i in interval ] for gene_index in gene_indexes])/n 
        else:
            return np.sum([[self.generation[i].genotype[gene][gene_index]==value for i in interval ]for gene_index in gene_indexes])/n

    def _get_interval_from_fraction(self,fraction,gene_indexes):
        """

        Auxiliary function to get the interval of finches in the generation that will be used for calculating the gene frequency.

        Parameters
        ----------
        fraction : str, int, float, list or np.ndarray
        gene_indexes : int, str, list or np.ndarray

        Returns
        -------
        list
            the list of indexes of the finches in the generation that will be used for calculating the gene frequency
        """
        if fraction=="quartile": fraction=0.25
        elif fraction=="half": fraction=0.5
        elif fraction=="decile": fraction=0.1 
        elif fraction=="all" or fraction==1: fraction=1.0
        if type(fraction)==list or type(fraction)==np.ndarray: interval=fraction
        #elif type(fraction)==int or isinstance(fraction,(int,np.integer)):interval=[fraction] #it has not sense, will yield infinite
        elif type(fraction)==float or isinstance(fraction,np.floating):  
            interval=list(range(0,int(len(self.generation)*fraction))) 
            if fraction>1.0: 
               print("fraction should be <1")

        if type(gene_indexes)==int: gene_indexes=[gene_indexes]
        elif type(gene_indexes)==str and gene_indexes=="all": 
            gene_indexes=range(0,self.copies_per_gen)
        
        return interval,gene_indexes

    def gene_enrichment(self,gene,value,fraction="quartile",gene_index=0):
        """
        returns frequency of a certain gene value and how much time is more frequent this value in the fenotype of the best performing fraction of finches than in the overall generation

        Parameters
        ----------
        gene : string 
              the name of the gene (key of the genotype dictionary)  that will be investigated. 
               in case the gene value is a list, only the first element (the fenotype) will be compared
        value : float, string or list
                the value that will be used to compare. 
                gene value == value or value[0]< gene value < value[1] will be used in the comparison
                depending if value is a single value or a 2-element list.
        fraction : string "quartile", "half" or "decile", by default "quartile", or float, optional
                the fraction of best-scoring finches on which it will be evaluated, by default "quartile"
        gene_index : int, optional
                the index of the gene in the genotype dictionary, by default 0
        Returns
        -------
        float
                the index of the genes in the genotype dictionary (for cases in which there are fenotypes and genotypes), by default 0 (use the fenotype)
        """

        best_interval,_=self._get_interval_from_fraction(fraction,gene_index)
  
        worst_interval=list(range(0,len(self.generation)))
        #worst_interval=[int(len(self.generation)*fraction),len(self.generation)]
        best_freq=self.gene_value_frequency(gene,value,best_interval,gene_indexes=gene_index)
        worst_freq= self.gene_value_frequency(gene,value,worst_interval,gene_indexes=gene_index)
        return best_freq,best_freq/worst_freq

    def calculate_discrete_gene_frequencies(self,fraction=1.0,gene_indexes=0):
        """

        returns a dictionary, with keys: each key in self.alleles, and as values other dictionaries. 
        In these dictionaries, keys are each possible gene value in self.alleles, values are the frequency of each gen
        These frequencies are calculated only for genes with discrete values (int, str or bool)

        Parameters
        ----------
        fraction : string "quartile", "half" or "decile", by default "quartile", or float, optional
                the fraction of best-scoring finches on which it will be evaluated, by default "quartile"
        gene_index : int, optional
                the index of the gene in the genotype dictionary, by default 0
        Returns
        -------
        float
            a dictionary with each gene's frequency        
        """


        interval,gene_indexes=self._get_interval_from_fraction(fraction,gene_indexes)
    
        #print ("len interval:"+str(len(interval)))
        #print ("len gene_indexes:"+str(len(gene_indexes)))

        freqs={}
        for g in self.alleles.keys():
            f={} 
            if isinstance(self.alleles[g][0],(str,np.str_,bool,np.bool_)):
                for v in set(self.alleles[g]): 
                    f[v]=self.gene_value_frequency(g,v,interval,gene_indexes=gene_indexes)
                    #f[v]=np.sum ( [self.gene_value_frequency(g,v,interval,gene_indexes=gene_index) for gene_index in gene_indexes])
                freqs[g]=f

        return freqs         

    def change_discrete_gene_frequencies(self,new_finch,fraction=1.0,gene_indexes=0,remove=True,freqs="calculate"):
        """
        updates the frequencies of discrete genes in the generation, removing the contribution of the removed finch
        returns a dictionary, with keys: each key in self.alleles, and as values other dictionaries. 
        In these dictionaries, keys are each possible gene value in self.alleles, values are the frequency of each gen
        These frequencies are calculated only for genes with discrete values (int, str or bool)

        Parameters
        ----------
        new_finch : finch object or int, required.
                the finch that will be removed (or added) or the index of the finch in the population.
                If it is a finch, it will be added or removed, depending on the value of "remove"; 
                if it is an int, the finch with that index will be removed for the calculation
        fraction : string "quartile", "half" or "decile", by default "quartile", or float, optional
                the fraction of best-scoring finches on which it will be evaluated, by default "quartile"
        gene_index : int, optional
                the index of the gene in the genotype dictionary, by default 0
        remove : bool, optional, default True
                whether to remove the finch or add it, by default True
        freqs: "calculate" or dictionary with de format generated by calculate_discrete_gene_frequencies, default: "calculate"
                the frequencies of the genes in the original generation.

        Returns
        -------
        float
            a dictionary with each gene's frequency        
        """

        interval,gene_indexes=self._get_interval_from_fraction(fraction,gene_indexes)

        if isinstance(new_finch,finch) and remove: 
            removed_finch_index=self.generation.index(new_finch)

        elif isinstance(new_finch,(int,np.integer)): 
            removed_finch_index=new_finch
   

        #calculate the change in the entropy contribugion from genes with discontinuous (int,str or bool) values
        #these formulas are correct when n/n-1 ≈ 1

        if freqs=="calculate": freqs=self.calculate_discrete_gene_frequencies(fraction=fraction,gene_indexes=gene_indexes)
        #calculate the new freqs
        n=len(interval)*len(gene_indexes)
        freqs_prime={}
        for g in freqs.keys():
            freqs_prime[g]={}
            if remove:
                v=[self.generation[removed_finch_index].genotype[g][gene_index] for gene_index in gene_indexes]
                sign=-1
            else:
                v=[new_finch.genotype[g][gene_index] for gene_index in gene_indexes]
                sign=1
            for gn in freqs[g].keys():
                freqs_prime[g][gn]=(n*freqs[g][gn]+sign*v.count(gn))/(n+sign*len(gene_indexes))

        return freqs_prime 

    def diversity_entropy(self,fraction=1.0,gene_indexes=0,freqs="calculate",added_finch=None):
        """
        returns the normalized entropy diversity of the distribution of values of a gene in the fenotype of the best performing fraction of finches
        according to: https://www.socs.uoguelph.ca/~wineberg/publications/GECCO2003_diversity.pdf
        note that there, the entropy is divided by the number of genes, but since we expect all finches to have the same number of genes, it will not have any effect here.

        Parameters
        ----------
        fraction : string, "quartile", "half" or "decile", by default "quartile", or float, or list, optional
                the fraction of best-scoring finches (assuming the list is ordered), the range or the indivitual elements
                on which it will be evaluated, by default evaluate on all population.
        gene_index : int or list, optional
                the index of the gene in the genotype dictionary, by default 0
        freqs: "calculate" or dictionary with de format generated by calculate_discrete_gene_frequencies, default: "calculate"
        added_finch : finch object, optional.
                an additional finch that will be added to the population for the calculation of the entropy, by default None

        Returns
        -------
        float
            entropy of the distribution of values of a gene in the fenotype of the best performing fraction of finches
        """

        interval,gene_indexes=self._get_interval_from_fraction(fraction,gene_indexes)

        entropy_continuous,entropy_discrete=0.0,0.0
        #calculate the entropy contribution of genes with continuous (float or int) values
        for g in self.alleles.keys():
            if type(self.alleles[g][0])==float or isinstance(self.alleles[g][0],(np.floating,int,np.int,np.int_)): 
                if added_finch!=None:
                    var= np.var([[ self.generation[i].genotype[g][gene_index] for i in interval  ] + [added_finch.genotype[g][gene_index]] for gene_index in gene_indexes])
                else:
                    var= np.var([[ self.generation[i].genotype[g][gene_index] for i in interval  ] for gene_index in gene_indexes])                
                if var>0: entropy_continuous+=0.5*np.log(2*np.pi*np.e*var)#entropy of normal distribution, from wikipedia; if var=0, entropy=0
        
        #calculate the entropy contribution of genes with discrete values. 
        if freqs=="calculate": 
            freqs=self.calculate_discrete_gene_frequencies(fraction=fraction,gene_indexes=gene_indexes)
            if added_finch!=None: freqs=self.change_discrete_gene_frequencies(new_finch=added_finch,fraction=fraction,gene_indexes=gene_indexes,remove=False)
        for g in freqs.keys():
                a=len(freqs[g].keys())
                if a==1: continue #if there is only one option, there is no contribution to entropy
                norm_factor=1/(np.log(a)) #calculation of normalizing factor in eq. 12
                #norm_factor=a/( (a-1) -(self.population%a)*(a-self.population%a)/(self.population**2)) #alternative calculation of normalizing factor in Theorem 1b
                entropy_discrete+= norm_factor * np.sum( [freqs[g][f]*np.log(1/freqs[g][f]) for f in freqs[g].keys()   if freqs[g][f]>0] )

        entropy=entropy_continuous+entropy_discrete
        return  entropy
    
    def change_diversity_entropy(self,changed_finch,fraction=1.0,gene_indexes="all",freqs="calculate",freqs_prime="calculate",remove=True):
        """
        calculates the change in the entropy diversity of the distribution of values of a gene in the fenotype of the best performing fraction of finches when 
        removed_finch is eliminated, according to: https://www.socs.uoguelph.ca/~wineberg/publications/GECCO2003_diversity.pdf

        Parameters
        ----------
        changed_finch : finch object or int, required.
                the finch that will be removed or the index of the finch in the population
        fraction : string, "quartile", "half" or "decile", by default "quartile", or float, or list, optional
                the fraction of best-scoring finches (assuming the list is ordered), the range or the indivitual elements
                on which it will be evaluated, by default evaluate on all population.
        gene_indexes : int, list or string, optional
                the index of the gene in the genotype dictionary, by default 0
        freqs: "calculate" or dictionary with de format generated by calculate_discrete_gene_frequencies, default: "calculate"
        freqs_prime: "calculate" or dictionary with de format generated by change_discrete_gene_frequencies, default: "calculate"
        remove : bool, optional, default True
                whether to remove the finch or add it, by default True

        Returns
        -------
        float
            difference in the entropy of the distribution of values when the specified finch is removed.
        """
        interval,gene_indexes=self._get_interval_from_fraction(fraction,gene_indexes)

        #calculate the change in the entropy contribugion from genes with discontinuous (int,str or bool) values
        #these formulas are correct when n/n-1 ≈ 1
        if freqs=="calculate": freqs=self.calculate_discrete_gene_frequencies(fraction=fraction,gene_indexes=gene_indexes)
        if freqs_prime=="calculate": freqs_prime=self.change_discrete_gene_frequencies(new_finch=changed_finch,fraction=fraction,gene_indexes=gene_indexes,remove=remove)
            
        orig_entropy=self.diversity_entropy(fraction=interval,gene_indexes=gene_indexes,freqs=freqs)
        if remove:
            if isinstance(changed_finch,finch): removed_finch_index=self.generation.index(changed_finch)
            else: removed_finch_index=changed_finch
            new_entropy=self.diversity_entropy(fraction=[i for i in interval if i!=removed_finch_index],gene_indexes=gene_indexes,freqs=freqs_prime)
        else:
            new_entropy=self.diversity_entropy(fraction=interval,gene_indexes=gene_indexes,freqs=freqs_prime,added_finch=changed_finch)
        return (new_entropy-orig_entropy)

    def finch_diversity_entropy_contribution(self,evaluated_finch,fraction=1.0,gene_indexes="all"):
        """
        returns the contribution of a finch to the diversity entropy of the distribution of values of a gene in the fenotype of the best performing fraction of finches
        inspired in: https://www.socs.uoguelph.ca/~wineberg/publications/GECCO2003_diversity.pdf
        Quite slow... but exact: it calculates sntropy with and without the finch. change_diversity_entropy should be faster and almos as accurate, unless 
        the population is small and n/n-1 is very different to 1. 
        Parameters
        ----------
        evaluated_finch : finch
            the finch that will be evaluated           
        fraction : string, "quartile", "half" or "decile", by default "quartile", or float, optional
                the fraction of best-scoring finches on which it will be evaluated, by default "quartile", or interval of finches
        gene_index : int, list or str, optional
                the index of the gene in the genotype dictionary, by default "all"
        Returns
        -------
        float
            entropy of the distribution of values of a gene in the fenotype of the best performing fraction of finches
        """

        interval,gene_indexes=self._get_interval_from_fraction(fraction,gene_indexes)
        if isinstance(evaluated_finch,finch): finch_index=self.generation.index(evaluated_finch)
        elif isinstance(evaluated_finch,(int,np.integer)): finch_index=evaluated_finch
        else: finch_index=int(evaluated_finch)
        interval_wo_instance=[i for i in interval if i!=finch_index]

        entropy=self.diversity_entropy(fraction=interval,gene_indexes=gene_indexes)
        entropy_wo_instance=self.diversity_entropy(fraction=interval_wo_instance,gene_indexes=gene_indexes) 
        return entropy_wo_instance - entropy


    def scores_statistics(self,test_fail_score_index=0,report_scores_indexes=[0],score_index_for_best=0):
        """

        reports statistics about generation scores, incluiding failing rate, best, worst, mean, std dev. values, etc.

        Parameters
        ----------
        test_fail_score_index : int, optional, by default 0
            in case finch's score is a list, the index of the score that will be used to check if scoring was not successful.
        report_scores_indexes : list, optional, by default [0].
            in case finch's score is a list, indexes of the scores that will be used in the report. 
        score_index_for_best : int, optional, by default 0
            in case finch's score is a list, the index of the score that will be used to decide which finch performs better.
        Returns
        -------
        string
            a report of the scores statistics
        """

        text=""


        #if scores is float:
        if isinstance(self.generation[0].score,(float,np.floating)):
            valid_scores=outlier_proof_statistics(np.array([b.score for b in self.generation]),return_stats="valid_scores",exclude_limit=self.valid_score_range)["valid_scores"]
        #if scores is list:
        elif isinstance(self.generation[0].score,(list,np.ndarray)):
            valid_instances=outlier_proof_statistics(np.array([b.score[test_fail_score_index] for b in self.generation]),return_stats="valid_instances",exclude_limit=self.valid_score_range)["valid_instances"] 
            valid_scores=[self.generation[i].score for i in valid_instances ]

        #print (valid_scores) #for debuggin
        failed=100*(len(self.generation)-len(valid_scores))/len(self.generation)
        text+="{:.3f}".format(failed)+"% of evaluations failed."
        if failed>25.0: text+=" Consider to increase the time limit given for evaluating score function."
        text+="\nOf the remaining:"

        [d1_value,q1_value,h1_value]=np.quantile([v[score_index_for_best] for v in valid_scores],[0.9,0.75,0.5],method="nearest") # in numpy v.1.21 and earlier, instead of "method" should be "interpolation"
        best_finch_index=np.argmax([v[score_index_for_best] for v in valid_scores]) 
        best=valid_scores[best_finch_index]
        worst_finch_index=np.argmin([v[score_index_for_best] for v in valid_scores]) 
        worst=valid_scores[worst_finch_index]
        d1_finch_index= np.where(np.isclose([v[score_index_for_best] for v in valid_scores],d1_value))[0][0]
        d1=valid_scores[d1_finch_index]
        q1_finch_index= np.where(np.isclose([v[score_index_for_best] for v in valid_scores],q1_value))[0][0]
        q1=valid_scores[q1_finch_index]
        h1_finch_index= np.where(np.isclose([v[score_index_for_best] for v in valid_scores],h1_value))[0][0]
        h1=valid_scores[h1_finch_index]

        tb,tw="",""
        t1std,t1stq,t1sth="","",""
        #for b in np.array(best)[report_scores_indexes]: tb+="{:14.4f}".format(b)
        for b in np.array(best): tb+="{:14.4f}".format(b)
        for w in np.array(worst)[report_scores_indexes]: tw+="{:14.4f}".format(w)
        for d in np.array(d1)[report_scores_indexes]: t1std+="{:14.4f}".format(d)
        for q in np.array(q1)[report_scores_indexes]: t1stq+="{:14.4f}".format(q)
        for h in np.array(h1)[report_scores_indexes]: t1sth+="{:14.4f}".format(h)

        mean,median,stddev=np.mean(valid_scores, axis=0),np.median(valid_scores, axis=0),np.std(valid_scores, axis=0)
        tm,tmd,tstd,tskw="","","",""
        for m in np.array(mean)[report_scores_indexes]: tm+="{:14.4f}".format(m)
        for m in np.array(median)[report_scores_indexes]: tmd+="{:14.4f}".format(m)            
        for m in np.array(stddev)[report_scores_indexes]: tstd+="{:14.4f}".format(m) 
        try: import scipy     
        except ImportError as err: pass
        if "scipy" in sys.modules:
            skw=scipy.stats.skew(valid_scores, axis=0)
            for m in np.array(skw)[report_scores_indexes]: tskw+="{:14.3f}".format(m)

        text+="\n- the best score is:              "+tb
        text+="\n- the worst score is:             "+tw
        text+="\n- the 1st decile score is:        "+t1std
        text+="\n- the 1st quartile score is:      "+t1stq
        text+="\n- the mean of the scores is:      "+tm
        text+="\n- the median of the scores is:    "+tmd
        text+="\n- the std.dev of the scores is:   "+tstd
        if tskw!="":
            text+="\n- the skewness of the scores is:  "+tskw


        best_age,worst_age=self.generation[best_finch_index].age,self.generation[worst_finch_index].age
        d1_avg_age=np.mean( [f.age for f in self.generation if f.score[score_index_for_best]>=d1_value ] )
        q1_avg_age=np.mean( [f.age for f in self.generation if f.score[score_index_for_best]>=q1_value ] )
        h1_avg_age=np.mean( [f.age for f in self.generation if f.score[score_index_for_best]>=h1_value ] )
        text+="\n average ages of finches:\n  best: "+str(best_age)+"   worst: "+str(worst_age)+"   1st decil: "+str(d1_avg_age)+"   1st quartil: "+str(q1_avg_age)+"   1st half: "+str(h1_avg_age)


        import time
        #ctime=time.time()
        text+="\n- the norm. diversity entropy is: "+str(self.diversity_entropy(fraction=1.0,gene_indexes="all"))
        #text+="\ntime required for norm.diversity entropy calculation is: "+str(time.time()-ctime)
        #ctime=time.time()
        #entropies=[self.finch_diversity_entropy_contribution(evaluated_finch=f,fraction=1.0,gene_indexes="all") for f in self.generation]
        #text+="\n- the list of contributions to the finch diversity entropy is: \n"+str(entropies)
        #text+="\n- the mean contribution to the finch diversity entropy is: "+str(np.mean(entropies))
        #text+="\n- the std.dev contribution to the finch diversity entropy is: "+str(np.std(entropies))
        #text+="\n- the max contribution to the finch diversity entropy is: "+str(np.max(entropies))
        #text+="\n- the min contribution to the finch diversity entropy is: "+str(np.min(entropies))
        #text+="time required for contributions of finch diversity entropy calculation is: "+str(time.time()-ctime)

        return text+"\n"













