#!/usr/bin/env python
# coding: utf-8
"""
GA for Material GmAte.py
2023/06/23 yokoyama

add ML
2023/07/11 yokoyama

"""
import time
impor = time.time()

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import random
from random import shuffle
import codecs
import sys
import glob
import datetime
import pandas as pd
import math
import inp_ga
from Specific import inp_POSCAR
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

importtime = time.time() - impor

pwd = os.getcwd()

if len(sys.argv) < 2:
    print(f"USAGE: {sys.argv[0]} -mode\n ----mode----\n  -ga: run the genetic algorithm\n  -bestgene: get the best gene and create new directory\n  -gene2pos: get the POSCAR from gene\n")
    sys.exit()
    
"""
mode select

"""
print_generate_status = True
rdfmode = "vasp" #"vasp" or "ase"
# calc_loop = 2


"""
Setting of Various Parameters

"""
#For ML
mlga = inp_ga.mlga
if mlga == True:
    mlopt1 = inp_ga.mlopt1
    mlopt2 = inp_ga.mlopt2
    mlopt3 = inp_ga.mlopt3
    mlext1 = inp_ga.mlext1
    mlext2 = inp_ga.mlext2
    mlext3 = inp_ga.mlext3
    outMLpred = inp_ga.outMLpred
    outMLreg = inp_ga.outMLreg
    dirML = inp_ga.dirML
    MLoriginal = inp_ga.MLoriginal
    from SpecificML import predML
    from SpecificML import make_model
#For GA
POPULATION = inp_ga.POPULATION              # population par generation
NUM_OF_STRINGS = inp_ga.NUM_OF_STRINGS      # num of genes
GENERATION = inp_ga.MAX_GENERATION          # number of generation
SAVE = inp_ga.SAVE                          # Number of the most stable genes that are unconditionally transmitted to the next generation
SURVIVAL_RATE = inp_ga.SURVIVAL_RATE        # Percentage of individuals allowed to survive each generation
CR_2PT_RATE = inp_ga.CR_2PT_RATE            # Percentage of 2-point crossing
CR_UNI_RATE = inp_ga.CR_UNI_RATE            # Percentage of uniform crossing
CR_UNI_PB = inp_ga.CR_UNI_PB                # Probability of occurrence of uniform crossover
MUTATION_PB = inp_ga.MUTATION_PB            # Probability of occurrence of mutation
STOP_CRITERIA = inp_ga.STOP_CRITERIA        # Stop condition when best does not change continuously
try:
    RESTART = inp_ga.RESTART                # if you want to restart =True
except:
    RESTART = False
ELEMENT_FIX = inp_ga.ELEMENT_FIX
select_mode = inp_ga.select_mode
tmp_gene = inp_ga.temp_gene
ncore = inp_ga.ncore
eval_file = inp_ga.eval_file
duplicate_crit = 200
ions = inp_POSCAR.ions
ELEM = inp_POSCAR.ELEM
runtype = inp_POSCAR.runtype
ref_poscar = "POSCAR_org"
output = inp_POSCAR.output
savefiles = inp_POSCAR.savefiles
try:
    thread = inp_POSCAR.thread
except:
    thread = False

#read inp.params >> genoms
with open(f"{pwd}/inp.params", "r") as f:
    genoms = f.read().splitlines()

"""
Warrning

"""
def check_the_num_of_strings():
    if len(genoms) != NUM_OF_STRINGS:
        sys.exit("Warning! Incorrect number of gene strings")
    else:
        pass

def check_argument():
    if len(sys.argv) == 1 :
        print(f"USAGE: {sys.argv[0]} -mode\n ----mode----\n  -ga: run the genetic algorithm\n  -bestgene: get the best gene and create new directory\n")
        sys.exit()
    else:
        pass

def tabledescript(ions):
    
    ionsl = sorted(ions)
    ionsl=ionsl[0]
    rdflist = []
    adflist = []
    listsave1 = []
    listsave2 = []
    for a in ionsl:
        for b in ionsl:
            for c in ionsl:
                label = f"adf_{a}-{b}-{c}"
                if label not in listsave2:
                    adflist.append(label)
                    listsave2.append(label)
                    if a != c:
                        listsave2.append(f"adf_{c}-{b}-{a}")
    for a in ionsl:
        for b in ionsl:
            label = f"rdf_{a}-{b}"
            if label not in listsave1:
                rdflist.append(label)
                listsave1.append(label)
    for d in ionsl:
        dd = f"rdf_{d}"
        rdflist.append(dd)
    rdflist.append("rdfall")
    adflist.append("adfall")
    with open(f"{pwd}/rdfadf_check", "w") as w:
        w.write(f"{len(rdflist)}\n")
        for i in rdflist:
            w.write(i + "\n")
        w.write(f"{len(adflist)}")
        for i in adflist:
            w.write(i + "\n")

    collist = []
    for i in rdflist:
        if "all" in i:
            n = 20
        else:
            n = 19
        for k in range(1, n+1):
            ll = f"{i}_{str(k)}"
            collist.append(ll)

    for i in adflist:
        n = 60
        for k in range(1, n+1):
            ll = f"{i}_{str(k)}"
            collist.append(ll)

    df = pd.DataFrame(columns = collist)
    return rdflist, adflist, df

if mlga == True:
    rdflist, adflist, df = tabledescript(ions) 
    df_o = df.copy()
    if rdfmode == "ase":
        df = pd.DataFrame()
        df_o = df.copy()

if "matlantis" in runtype and ncore != 1:
    print("WARNING: matlantis, ncore must be 1")
    sys.exit()

check_the_num_of_strings()
if sys.argv[1] == "-ga" and "matlantis" in runtype and thread == True:
    from Specific import opt

elif sys.argv[1] == "-ga" and "m3g" in runtype and thread == True:
    from Specific import optm3g

elif sys.argv[1] == "-ga" and "test" in runtype and thread == True:
    from Specific import opttest   


"""
Making of Additional Parameters

"""
num = int(POPULATION*SURVIVAL_RATE) #Number of individuals to be kept alive
gen_info = [] #How many of each element are there ( For checking if the number of ions is maintained )
num_of_base = [] #How many bases are used on genes

#num_of_base : [num of base at string1, string2]
for x in range(0, NUM_OF_STRINGS):
    n = len(set(str(genoms[x])))
    num_of_base.append(n)
    
#gen_info : [[string1: [1st element, num of 1st element], [2nd, n],], [string2: [1st, n], [2nd, n],], ]    
for x in range(0, NUM_OF_STRINGS):
    gen_info.append([])
    for i in range(0, num_of_base[x]):
        count = str(genoms[x]).count(str(i))
        gen_info[x].append([str(i), count])     

n_2pt = int(Decimal(str(POPULATION * CR_2PT_RATE)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
n_uni = int(Decimal(str(POPULATION * CR_UNI_RATE)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
n_mu = POPULATION - n_2pt - n_uni

"""For ML 0629"""
if mlga == True:
    n_mluni = n_uni - mlopt1
    n_ml2pt = n_2pt - mlopt2
    n_mlmu = n_mu - mlopt3
    mllist = []
    if n_mluni > 0:
        mllist.append("uni")
    if n_ml2pt > 0:
        mllist.append("2pt")
    if n_mlmu > 0:
        mllist.append("mu")

    mluni_label = "preML_uni"
    ml2pt_label = "preML_2pt"
    mlmu_label = "preML_mu"

""""""


"""
Setting of The Names of Output Files

"""
value_file = "out.value_indiv"
genestock_file = "temp.genestock"
value_only_file = "out.value"
sort_file = "sort.value_indiv"
time_file = "out.elapsedtime"
    
"""
Creation of Output Files

"""
def make_value_file(generation):
    """Save considered genes to file"""
    sort_result = sorted(generation, reverse=False, key=lambda Indivisual: Indivisual.get_score())
    if generation_index == 0:
        with open(value_file, "w") as f:
            f.write("# generation  value(error)  rank_in_gen  strings kind birth saved_count make_gene indivisual_index\n")
        # with open(value_only_file, "w") as j:
        #     j.write("")
    with open(value_file, "a") as w:
        for n,i in enumerate(sort_result):
            w.write(f"{generation_index}\t{'{:.5f}'.format(i.score)}\t{n+1}\t{(' ').join(i.genom)}\t{i.kind}\t{i.birth}\t{saved_count}\t{i.index_indiv}\n")
    with open(value_only_file, "a") as k:
        k.write(f"{generation_index}")
        for i in sort_result:
            k.write(f" {i.score}")
        k.write("\n")
            
def make_genestock(gen):
    """Save considered genes to file"""
    with open(genestock_file, "a") as f:
        w_line = " ".join(gen)
        f.write(f"{w_line}\n")    
        
def make_time_file():
    if generation_index == 0:
        with open(time_file, "w") as f:
            f.write("# generation  spent_time(2pt_cross uni_cross mutation)  total_time  finish_time\n")
        if mlga == True:
            with open("ml_time", "w") as q:
                q.write("# generation  make_candidate_gene  make_rdfadf  make_model  prediction\n")
    with open(time_file, "a") as k:
        k.write(f"generation{generation_index}  {spent_time} (2pt_cross: {cr2pt_time}, uni_cross: {cruni_time}, mutation: {mut_time})  {total_time}  (outputted at {finish_time})\n")
    if mlga == True:
        with open("ml_time", "a") as p:
            p.write(f"{generation_index} {mlgen_time}  {mkmlrdf_time} {mkmodel_time} {pred_time}\n")
    
def make_sort_file(read_file):
    value_indiv = []
    with open(read_file, "r") as f:
        head = f.readline()
        lines = f.readlines()
    for i in lines:
        indiv = i.split()
        if indiv[NUM_OF_STRINGS+3] == "saved_gen":
            pass
        else:
            value_indiv.append(indiv)
    value_indiv_sort = sorted(value_indiv, reverse=False, key=lambda x: (float(x[1]), -1*int(x[0])))
    with open("sort.value_indiv", "w") as w:
        w.write("#rank generation indivisual_index values gene\n")
        for n, i in enumerate(value_indiv_sort):
            wline = f"{str(n+1)}  {i[0]}  {i[-1]}  {i[1]}  {' '.join(i[x] for x in range(3, 3+NUM_OF_STRINGS))}"
            # wline = '\t'.join(i)
            w.write(f"{wline}\n")
    return value_indiv_sort          
            
def save_outputfile():
    os.system(f"rm {genestock_file}")
    outfile = glob.glob(f"{pwd}/out.*")
    outfile = [i.split("/")[-1] for i in outfile]
    outputfile = []
    for i in outfile:
        if "save" not in i:
            outputfile.append(i)
    if outputfile == []:
        pass
    else:
        savenum = 1
        while True:
            if glob.glob(f"{pwd}/out.*save{str(savenum)}") != []:
                savenum += 1
            else:
                for i in outputfile:
                    if "save" in i:
                        pass
                    else:
                        os.system(f"mv {pwd}/{i} {pwd}/{i}_save{str(savenum)}")
                break


"""
Functions for Checking Genes

"""
def check_duplicate(gen):
    """Check for duplicate genes: True >> Duplicate"""
    return (gen in all_gen)

def check_species(gen, x):
    """Check if the number of ion species has not changed: True >> Ok"""
    test_gen_info = []
    for i in range(0, num_of_base[x]):
        count = str(gen).count(str(i))
        test_gen_info.append([str(i), count])
    return (test_gen_info == gen_info[x])


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

"""
Indivisual

"""
class Indivisual:
    def __init__(self, genom):
        self.genom = genom[0]
        self.kind = genom[1]
        self.birth = genom[2]
        self.index_indiv = genom[3]
        # if thread == True:
        if "preML" in self.kind:
            #self.cpwd = f"{pwd}/{dirML}/{self.kind}/{str(generation_index)}_{str(self.index_indiv).zfill(3)}"
            #os.system(f"rm {pwd}/{dirML}/{self.kind}/{str(generation_index-1)}_{str(self.index_indiv).zfill(3)} -r")                
            try:
                os.system(f"rm {self.cpwd} -r")
            except:
                pass
            os.mkdir(self.cpwd)
            os.system(f"cp {pwd}/Specific/{ref_poscar} {self.cpwd}")
            self.set_genom()
            self.gene_to_POSCAR()
        else:
            if mlga == True:
                self.rdf_dir = f"{pwd}/{dirML}/dir_{generation_index}_{str(self.index_indiv).zfill(3)}" 
            #self.cpwd = f"{pwd}/{dirML}/{self.kind}/{str(generation_index)}_{str(self.index_indiv).zfill(3)}"
            if os.path.isdir(f"{pwd}/sample{str(self.index_indiv).zfill(3)}"):
                self.cpwd = f"{pwd}/sample{str(self.index_indiv).zfill(3)}"
            #if os.path.exists(self.cpwd):
            #    create_directory_if_not_exists(self.rdf_dir)
            #os.mkdir(self.rdf_dir)
            self.set_genom()
            self.gene_to_POSCAR()
            if thread == True:
                self.gene_to_POSCAR()
        self.score = 0  
        
    def set_genom(self):
        if os.path.exists(self.cpwd):
            with open(f"{self.cpwd}/{tmp_gene}", "w") as w:
                for i in range(len(self.genom)):
                    w.write(self.genom[i] + "\n")
            with open(f"{self.cpwd}/Save_info_{generation_index}", "w") as q:
                q.write(f"generation {generation_index}\n")
                q.write(f"indivisual {self.index_indiv}\n")
                q.write(f"kind {self.kind}\n")
                for i in range(len(self.genom)):
                    q.write(f"gene{str(i+1)} {self.genom[i]}\n")
        if "preML" in self.kind:
            with open(f"{self.cpwd}/birth", "w") as f:
                f.write(f"{self.birth}\n")
        else:
            os.system(f"mv {self.cpwd}/finish {self.cpwd}/wait")
        if os.path.exists(self.cpwd):    
            with open(f"{self.cpwd}/flag", "w") as k:
                k.write(str(generation_index)+"\n")

    def gene_to_POSCAR(self):
        if os.path.exists(self.cpwd):
            os.chdir(self.cpwd)
            with open(ref_poscar, "r") as f:
                all_lines = f.readlines()
            all_lines = [i.split() for i in all_lines]
            labels = all_lines[:8]
            lines = all_lines[8:]
            if "Direct" not in labels[-1] and "Cart" not in labels[-1]:
                print("Warning : POSCAR_org must be in VASP5 format.　A label with the name of the atom is required on line 6.")
                sys.exit()
            strings = self.genom
            new_labels = labels.copy()
            label_info = []
            for i in range(len(labels[5])):
                if "ELEM" in labels[5][i]: 
                    flag = labels[5][i].lstrip("ELEM")
                    if flag == "":
                        flag = 1
                    else:
                        flag = int(flag)
                    for xx, x in enumerate(ELEM[flag-1]):
                        count = str(str(strings[flag-1]).count(str(xx)))
                        if x != "Vac":
                            label_info.append([x, count])
                else:
                    label_info.append([labels[5][i], labels[6][i]])
            label_info_sort = sorted(label_info, key=lambda x: str(x[0]))
            new_labels[5] = [i[0] for i in label_info_sort]
            new_labels[6] = [i[1] for i in label_info_sort]

            count_list = [0 for _ in range(len(strings))]
            for line in lines:
                if "ELEM" in line[3]:
                    flag = line[3].lstrip("ELEM")
                    if flag == "":
                        flag = 1
                    else:
                        flag = int(flag)
                    gen = int(strings[flag-1][count_list[flag-1]])
                    count_list[flag-1] += 1
                    line[3] = ELEM[flag-1][gen]
                else:
                    pass
            lines_sort = sorted(lines, key= lambda x: str(x[3]))

            with open(f"{self.cpwd}/POSCAR", "w") as w:
                for i in new_labels:
                    w.write(" ".join(i) + "\n")
                for i in lines_sort:
                    if "Vac" not in i:
                        w.write(" ".join(i) + "\n")
            if mlga == True:
                
                self.make_rdf()
            os.chdir(pwd)

    def calc_score(self):
        os.chdir(self.cpwd)
        #calculate the score
        os.system(f"mv {self.cpwd}/wait {self.cpwd}/running &")
        # self.gene_to_POSCAR()
        print("start calc")
        os.chdir(self.cpwd)
        if runtype == "test":
            # self.test_score() #for test
            opttest.main([])
        elif runtype=="matlantis":
            input = [["mode", "opt"], ["inpf", "POSCAR"]]
            opt.main(input)
        elif runtype=="matlantis-static":
            input=[["mode", "static"], ["inpf", "POSCAR"]]
            opt.main(input)
        elif runtype == "m3g":
            input = [["mode", "relax"], ["inpf", "POSCAR"]]
            optm3g.main(input)
        elif runtype == "m3g-static":
            input = [["mode", "static"], ["inpf", "POSCAR"]]
            optm3g.main(input)
        else:
            # os.system(f"{calccode}")    
            pass
        print("calc finish")
        self.make_savefiles()
        with open(f"{self.cpwd}/{eval_file}", "r") as f:
            score = float(f.readline().rstrip())
        # time.sleep(5)

        if mlga == True:
            with open(f"{self.rdf_dir}/out.energy", "w") as w:
                w.write(str(score) + "\n")

        os.system(f"rm {self.cpwd}/{eval_file}")
        os.system(f"mv {self.cpwd}/running {self.cpwd}/finish &")
        os.chdir(pwd)
        return score
    
    # def calc_score_loop(self):
    #     os.chdir(self.cpwd)
    #     #calculate the score
    #     os.system(f"mv {self.cpwd}/wait {self.cpwd}/running &")
    #     try:
    #         os.system("rm log.energ*")
    #     except:
    #         pass
    #     # self.gene_to_POSCAR()
    #     print("start calc")
    #     os.chdir(self.cpwd)
    #     for loop in range(calc_loop):
    #         if runtype == "test":
    #             # self.test_score() #for test
    #             opttest.main([])
    #         elif runtype=="matlantis":
    #             input = [["mode", "opt"], ["inpf", "POSCAR"]]
    #             opt.main(input)
    #         elif runtype=="matlantis-static":
    #             input=[["mode", "static"], ["inpf", "POSCAR"]]
    #             opt.main(input)
    #         elif runtype == "m3g":
    #             input = [["mode", "relax"], ["inpf", "POSCAR"]]
    #             optm3g.main(input)
    #         elif runtype == "m3g-static":
    #             input = [["mode", "static"], ["inpf", "POSCAR"]]
    #             optm3g.main(input)
    #         else:
    #             # os.system(f"{calccode}")    
    #             pass
    #         if loop + 1 == calc_loop:
    #             print("calc finish")
    #             self.make_savefiles()
    #             with open(f"{self.cpwd}/{eval_file}", "r") as f:
    #                 score = float(f.readline().rstrip())
    #             os.system(f"rm {self.cpwd}/{eval_file}")
    #             os.system(f"mv {self.cpwd}/running {self.cpwd}/finish &")
    #         else:
    #             os.system("mv CONTCAR POSCAR")
    #     os.chdir(pwd) 
    #     return score
    
    def make_rdf(self):
        if "preML" not in self.kind:
            rdfdir = self.rdf_dir
            try:
                os.system(f"rm {rdfdir} -r")
            except:
                pass
            if not os.path.exists(rdfdir):
                os.mkdir(rdfdir)
            os.system(f"touch {rdfdir}/calc")
        else:
            rdfdir = self.cpwd
        os.system(f"cp POSCAR {tmp_gene} {rdfdir}")
        if rdfmode == "vasp":
            
            os.chdir(rdfdir)
            #os.system('python ../../Specific/calc_energy.py -gene2pos')
            os.system("vaspfiles -rdfcodex POSCAR &")
            # os.system("vaspfiles -adfcodex POSCAR &")


    def set_energy_from_file(self):
        smp_num = str(self.index_indiv).zfill(3)
        while True:
            if os.path.isfile(f"{pwd}/sample{smp_num}/finish") == True:
                break
            else:
                time.sleep(0.1) #time.sleep(x) every x seconds, check whether the calculation is finished in the target directory
        with open(f"{pwd}/sample{smp_num}/{eval_file}", "r") as ffff:
            #print('readeline',f.readline())
            ene = float(ffff.readline().rstrip())
        #os.system(f"rm {pwd}/sample{smp_num}/{eval_file}")
        with open(f"{self.cpwd}/{eval_file}", "r") as f:
            score = float(f.readline().rstrip())
            # time.sleep(5)

        if mlga == True:
            with open(f"{self.rdf_dir}/out.energy", "w") as w:
                w.write(str(score) + "\n")
        return ene

    def test_score(self):
        with open(f"{self.cpwd}/temp_gene", "r") as f:
            strings = f.readline().split()
        sc = 0
        time.sleep(10)
        for x in range(0, len(strings)):
            for i in range(0, len(strings[x])):
                sc += int(strings[x][i])*(i+1)
        with open(f"{self.cpwd}/{eval_file}", "w") as w:
            w.write(f"{str(sc)}\n")

    def make_savefiles(self):
        with open(f"{self.cpwd}/flag", "r") as f:
            generation = f.readline().rstrip()
        for i in savefiles:
            if i == "log.m3g":
                os.system(f"mv {self.cpwd}/{i} {self.cpwd}/Save_{i}_{generation}")
            else:
                os.system(f"cp {self.cpwd}/{i} {self.cpwd}/Save_{i}_{generation}") 
        os.system(f"cp {self.cpwd}/{output} {self.cpwd}/Save_{output}_{generation}")

    def set_score_for_test(self):
        #calculate the score (for test)
        sc = 0
        for x in range(0, NUM_OF_STRINGS):
            l = len(self.genom[x])
            for i in range(0, l):
                sc += int(self.genom[x][i])*(i+1)
        self.score = sc
    
    def get_score(self):
        #get the score
        return self.score
    
"""
for calculation 

"""
def set_calc_place():
    calc_place = []
    for i in range(1, inp_ga.POPULATION+1):
        smp_num = str(i).zfill(3)
        os.mkdir(f"sample{smp_num}")
        os.system(f"touch {pwd}/sample{smp_num}/finish")
        os.system(f"cp {pwd}/Specific/POSCAR_org {pwd}/sample{smp_num}")
        if thread == False:
            os.system(f"cp {pwd}/Specific/inp_POSCAR.py {pwd}/sample{smp_num}")
            os.system(f"cp {pwd}/Specific/calc_energy.py {pwd}/sample{smp_num}")
            os.system(f"cp {pwd}/Specific/POTCAR {pwd}/sample{smp_num}")
            os.system(f"cp {pwd}/Specific/INCAR {pwd}/sample{smp_num}")
            os.system(f"cp {pwd}/Specific/KPOINTS {pwd}/sample{smp_num}")
        calc_place.append(f"sample{smp_num}")
    if mlga == True:
        os.mkdir(dirML)
        os.chdir(dirML)
        for i in mllist:
            os.mkdir(globals()[f"ml{i}_label"])
        os.chdir(pwd)

    return calc_place   

def check_calc_status():
    while True:
        running_list = glob.glob(f"{pwd}/sample*/running")
        wait_list = glob.glob(f"{pwd}/sample*/wait")
        wait_list = sorted(wait_list)
        finish_list = glob.glob(f"{pwd}/sample*/finish")
        if len(finish_list) == POPULATION:
            return ["finish"]
        elif wait_list == []:
            return ["nowait"]
        elif ncore > len(running_list):
            return ["available", wait_list[0].rstrip("/wait")]
        else:
            pass

def submit_calc_energy():
    #submit the calculation job
    while True:
        status = check_calc_status()
        if status[0] == "available":
            os.chdir(status[1])
            os.system("python calc_energy.py &")
            os.chdir(pwd)
        elif status[0] == "finish":
            break
        elif status[0] == "nowait":
            pass
        else:
            sys.exit("WARNING: status stop")
        time.sleep(1) #time.sleep(x): every x seconds, check the calculation statue


""" for ML 0629"""

def get_mltop(children):
    global index_indiv
    for mode in mllist:
        label = globals()[f"ml{mode}_label"]
        with open(f"{pwd}/{dirML}/{label}/{outMLpred}", "r") as f:
            for i in range(globals()[f"n_ml{mode}"]):
                a = f.readline().rstrip()
                dir = f"{pwd}/{dirML}/{label}/{a}"
                with open(f"{dir}/{tmp_gene}", "r") as q:
                    g = q.read().splitlines()
                with open(f"{dir}/birth", "r") as p:
                    k = p.readline().rstrip()
                index_indiv += 1
                genom = [g, "offsprings", f"ml_{k}", index_indiv]
                all_gen.append(g)
                make_genestock(g)
                children.append(genom)
    return children 
    
def prediction(mode):
    os.chdir(f"{pwd}/{dirML}/preML_{mode}")
    print("start predML")
    # print("pred")
    predML.main(generation_index, rdflist, adflist, df_o, rdfmode)


"""
Making Initial Population

"""
def create_first_generation(POPULATION, genoms):
    """Read GENOMS and create POPULATION genes (initial population)"""
    generations = []
    first_gens = []
    index_indiv = 0
    for i in range(1, POPULATION+1):
        while len(first_gens) != i:
            tmp_genoms = []
            for x in range(0, NUM_OF_STRINGS):
                lst = list(genoms[x])
                shuffle(lst)
                gen = "".join(lst)
                tmp_genoms.append(gen)
            if tmp_genoms not in first_gens:
                index_indiv += 1
                first_gens.append([tmp_genoms, "offsprings", "initial(rdm)", index_indiv])
                all_gen.append(tmp_genoms)
                make_genestock(tmp_genoms)
            else:
                pass

    #FIXME: 後で並列処理する
    for i in first_gens:
        generations.append(Indivisual(i))
    time.sleep(1)
    # with ProcessPoolExecutor(max_workers=ncore) as tpe11:
    #     for i in first_gens:
    #         future = tpe11.submit(Indivisual(i))
    #         generations.append(future.result)
    # tpe11.shutdown()

    if thread == True:
        futures = []
        with ProcessPoolExecutor(max_workers=ncore) as tpe:
            for i in generations:
                future = tpe.submit(Indivisual.calc_score, i)
                futures.append(future)
        tpe.shutdown()
        for n, i in enumerate(futures):
            generations[n].score = i.result()

    else:
        submit_calc_energy()
        for i in generations:
            i.score = Indivisual.set_energy_from_file(i)
            #with open(f"{self.cpwd}/{eval_file}", "r") as f:
            #    score = float(f.readline().rstrip())
                # time.sleep(5)

            #if mlga == True:
            #    with open(f"{self.rdf_dir}/out.energy", "w") as w:
            #        w.write(str(score) + "\n")
    return generations

"""
Selection of Survival Genes

1, Roulet Method
2, Tournament method
3, Ranking method
4, Elite Selection (no crossover)

"""
def select_roulet(generation, num):
    """Roulet Method"""
    selected = []
    weights = [math.exp((-1)*ind.get_score()/(8.31*273)) for ind in generation]
    norm_weights = [weights[i] / sum(weights) for i in range(len(weights))]
    selected = np.random.choice(generation, size=num, p=norm_weights)
    return selected

def select_tournament(generation, num):
    """Tournament method"""
    tournament_size = 3
    selected = []
    for i in range(num):
        tournament = np.random.choice(generation, tournament_size, replace=False)
        min_genom = min(tournament, key=Indivisual.get_score)
        selected.append(min_genom)
    return selected

def select_ranking(generation, num):
    """Ranking method"""
    sort_result = sorted(generation, reverse=False, key=lambda Indivisual: Indivisual.get_score())
    selected = sort_result[:num]
    return selected

def select_elite(generation, SAVE):
    """Inherit the top nth individual(Elite Selection (no crossover))"""
    sort_result = sorted(generation, reverse=False, key=lambda Indivisual: Indivisual.get_score())
    saved_gen = sort_result[:SAVE]
    for i in range(len(saved_gen)):
        saved_gen[i].kind = "saved_gen"
    return saved_gen

"""
Mutation

"""

def mutate(selected, children, mutatemode):
    """Mutation"""
    global all_gen, index_indiv
    global mlgene

    if mlga == True:
        if mutatemode == "ga":
            num_of_mutation = mlopt3
            kind = "offsprings"
            children_add = children
        elif "ml" in mutatemode:
            num_of_mutation = mlext3
            kind = mlmu_label
            children_add = []
    else:
        num_of_mutation = POPULATION - int(POPULATION * CR_2PT_RATE / 2)*2 - int(POPULATION * CR_UNI_RATE / 2)*2
        kind = "offsprings"
        children_add = children

    count_mutation = 0
    j = 0
    while True:
        index = int(np.random.randint(len(selected)))
        child = selected[index]
        j += 1
        if count_mutation >= num_of_mutation:
                break
        else:
            tmp_genoms = []
            for x in range(0, NUM_OF_STRINGS):
                tmp_gen = ""
                for i in range(len(child.genom[x])):
                    if np.random.rand() < MUTATION_PB:
                        while True:
                            tmp = int(np.random.randint(len(gen_info[x])))
                            if tmp != child.genom[x][i]:
                                break
                        tmp_gen += gen_info[x][tmp][0]
                    else:
                        tmp_gen += child.genom[x][i]
                if ELEMENT_FIX == True and check_species(tmp_gen, x) == False:
                    break
                tmp_genoms.append(tmp_gen)
            if len(tmp_genoms) == NUM_OF_STRINGS:
                if check_duplicate(tmp_genoms) == False:
                    if mutatemode == "ga":
                        all_gen.append(tmp_genoms)
                        make_genestock(tmp_genoms)
                        if print_generate_status == True:
                            print("mutation: " + " ".join(tmp_genoms))
                    else:
                        pass
                    if mutatemode == "ga":
                        index_indiv += 1
                        indiv_number = index_indiv
                    elif "ml" in mutatemode :
                        indiv_number = count_mutation + 1
                    if mlga == False:
                        new_gen = [tmp_genoms, kind, "mutation", indiv_number]
                        count_mutation += 1
                        children_add.append(new_gen)
                    else:
                        if tmp_genoms not in mlgene:
                            mlgene.append(tmp_genoms)
                            new_gen = [tmp_genoms, kind, "mutation", indiv_number]
                            count_mutation += 1
                            children_add.append(new_gen)
                        else:
                            pass
            else:
                pass                                  
    return children_add

"""
Crossover
select parents -> creation of gen = Failure -> gen rebirth

"""
if ELEMENT_FIX == True:
    rebirth_method = "gen_rebirth"
else:
    rebirth_method = "gen_rebirth_nofix"

def cross_2pt(child1, child2, crossmode, n):
    """2-point crossover"""
    global all_gen, index_indiv
    global mlgene
    i = 0
    dupl_count = 0
    flag_rebirth = "off"
    while True:
        if dupl_count > duplicate_crit : #or spefal_count > spefal_crit
            tmp1_genoms = globals()[rebirth_method](child1.genom)
            tmp2_genoms = globals()[rebirth_method](child2.genom)
            flag_rebirth = "on"    
            break
        i += 1
        tmp1_genoms = []
        tmp2_genoms = []
        for x in range(0, NUM_OF_STRINGS):
            size = len(child1.genom[x])
            tmp1 = list(str(child1.genom[x]))
            tmp2 = list(str(child2.genom[x]))
            cxpoint1 = np.random.randint(1, size)
            cxpoint2 = np.random.randint(1, size-1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
            tmp1[cxpoint1:cxpoint2], tmp2[cxpoint1:cxpoint2] = tmp2[cxpoint1:cxpoint2],tmp1[cxpoint1:cxpoint2] 
            tmp1 = "".join(tmp1)
            tmp2 = "".join(tmp2)
            if ELEMENT_FIX == True and (check_species(tmp1, x) == False or check_species(tmp2, x) == False):
                break
            else:
                tmp1_genoms.append(tmp1)
                tmp2_genoms.append(tmp2)
        if len(tmp1_genoms) == NUM_OF_STRINGS and len(tmp2_genoms) == NUM_OF_STRINGS:
            if check_duplicate(tmp1_genoms) == False and check_duplicate(tmp2_genoms) == False:
                if mlga == False:
                    break
                else:
                    if tmp1_genoms not in mlgene and tmp2_genoms not in mlgene:
                        break
                    else:
                        dupl_count += 1
            else:
                dupl_count += 1
        else:
            pass
    if crossmode == "ga":
        all_gen += [tmp1_genoms, tmp2_genoms]
        make_genestock(tmp1_genoms)
        make_genestock(tmp2_genoms)
        kind = "offsprings"
    elif "ml" in crossmode:
        mlgene += [tmp1_genoms, tmp2_genoms]
        kind = ml2pt_label
    else:
        pass

    if flag_rebirth == "on":
        flag_name = "rebirth (rdm)"
    else:
        flag_name = "cross_2pt"
    if print_generate_status == True:
        print(f"{flag_name}: " + " ".join(tmp1_genoms) + "  " + " ".join(tmp2_genoms))
    if crossmode == "ga":
        index_indiv += 1
        indiv_number = index_indiv
    elif "ml" in crossmode :
        indiv_number = n*2 + 1
    new_gen1 = [tmp1_genoms, kind, f"{flag_name}", indiv_number]
    if crossmode == "ga":
        index_indiv += 1
        indiv_number = index_indiv
    elif "ml" in crossmode:
        indiv_number += 1
    new_gen2 = [tmp2_genoms, kind, f"{flag_name}", indiv_number]
    return new_gen1, new_gen2

def cross_uni(child1, child2, crossmode, n):
    """uniformed crossover"""
    global all_gen, index_indiv
    global mlgene
    j = 0
    dupl_count = 0
    flag_rebirth = "off"
    while True:
        j += 1
        if dupl_count  > duplicate_crit:
            tmp1_genoms = globals()[rebirth_method](child1.genom)
            tmp2_genoms = globals()[rebirth_method](child2.genom)
            flag_rebirth = "on"
            break
        tmp1_genoms = []
        tmp2_genoms = []
        for x in range(0, NUM_OF_STRINGS):
            tmp1= []
            tmp2 = []
            for i in range(len(child1.genom[x])):
                if np.random.rand() > CR_UNI_PB:
                    tmp1.append(child1.genom[x][i])
                    tmp2.append(child2.genom[x][i])
                else:
                    tmp1.append(child2.genom[x][i])         
                    tmp2.append(child1.genom[x][i])
            tmp1_str = "".join(tmp1)
            tmp2_str = "".join(tmp2)
            if ELEMENT_FIX == True and (check_species(tmp1_str, x) == False or check_species(tmp2_str, x) == False):
                break
            else:
                tmp1_genoms.append(tmp1_str)
                tmp2_genoms.append(tmp2_str)
        if len(tmp1_genoms) == NUM_OF_STRINGS and len(tmp2_genoms) == NUM_OF_STRINGS:
            if check_duplicate(tmp1_genoms) == False and check_duplicate(tmp2_genoms) == False:
                if mlga == False:
                    break
                else:
                    if tmp1_genoms not in mlgene and tmp2_genoms not in mlgene:
                        break
                    else:
                        dupl_count += 1
            else:
                dupl_count += 1
        else:
            pass
    if flag_rebirth == "on":
        flag_name = "rebirth (rdm)"
    else:
        flag_name = "cross_uni"

    if crossmode == "ga":
        all_gen += [tmp1_genoms, tmp2_genoms]
        make_genestock(tmp1_genoms)
        make_genestock(tmp2_genoms)
        kind = "offsprings"
    elif "ml" in crossmode:
        mlgene += [tmp1_genoms, tmp2_genoms]
        kind = mluni_label
    else:
        pass

    if print_generate_status == True:
        print(f"{flag_name}: " + " ".join(tmp1_genoms) + " " + " ".join(tmp2_genoms))
    if crossmode == "ga":
        index_indiv += 1
        indiv_number = index_indiv
    elif "ml" in crossmode:
        indiv_number = n*2 + 1
    new_gen1 = [tmp1_genoms, kind, f"{flag_name}", indiv_number]
    if crossmode == "ga":
        index_indiv += 1
        indiv_number = index_indiv
    elif "ml" in crossmode:
        indiv_number += 1
        print(indiv_number)
    new_gen2 = [tmp2_genoms, kind, f"{flag_name}", indiv_number]
    return new_gen1, new_gen2

def crossover(selected, crossmode):
    """perform crossover"""
    children = []
    if mlga == True:
        if crossmode == "ga":
            num_of_cross_2pt = int(mlopt2 / 2)
            num_of_cross_uni = int(mlopt1 / 2)
        elif crossmode == "ml_uni":
            num_of_cross_2pt = 0
            num_of_cross_uni = int(mlext1/2)
        elif crossmode == "ml_2pt":
            num_of_cross_uni = 0
            num_of_cross_2pt = int(mlext2/2)
    else:
        num_of_cross_2pt = int(POPULATION * CR_2PT_RATE / 2)
        num_of_cross_uni = int(POPULATION * CR_UNI_RATE / 2)

    count_2ptcross = 0
    count_unicross = 0
    k = 0
    global cr2pt_time, cruni_time
    cross_start = time.time()
    while True:
        k += 1
        index1 = 0
        index2 = 0
        while index1 == index2:
            index1 = int(np.random.randint(len(selected)))
            index2 = int(np.random.randint(len(selected)))
        child1 = selected[index1]
        child2 = selected[index2]
        if count_2ptcross >= num_of_cross_2pt:
            cr2pt_fin = time.time()
            cr2pt_time = "{:.2f}".format(cr2pt_fin - cross_start)
            if count_unicross >= num_of_cross_uni:
                cruni_time = "{:.2f}".format(time.time() - cr2pt_fin)
                break
            else:
                new_child1, new_child2 = cross_uni(child1, child2, crossmode, count_unicross)
                children.append(new_child1)
                children.append(new_child2)
                count_unicross += 1
        else:
            new_child1, new_child2 = cross_2pt(child1, child2, crossmode, count_2ptcross)
            children.append(new_child1)
            children.append(new_child2)
            count_2ptcross += 1
    return children


"""
rebirth (random)

"""
def gen_rebirth_nofix(gen):
    while True:
        rb_gen = []
        for x in range(NUM_OF_STRINGS):
            dupl_gen = list(gen[x])
            for i in range(len(dupl_gen)):
                dupl_gen[i] = str(int(np.random.randint(0, num_of_base[x])))
            dupl_gen_string = "".join(dupl_gen)
            rb_gen.append(dupl_gen_string)
        if check_duplicate(rb_gen) == False:
            break
    return rb_gen

def gen_rebirth(gen):
    while True:
        rb_gen = []
        for x in range(NUM_OF_STRINGS):
            dupl_gen = list(gen[x])
            dupl_gen = [int(i) for i in dupl_gen]
            dupl_gen_s = random.sample(dupl_gen, len(dupl_gen))
            dupl_gen_s = [str(i) for i in dupl_gen_s]
            dupl_gen_string = "".join(dupl_gen_s)
            rb_gen.append(dupl_gen_string)
        if check_duplicate(rb_gen) == False:
            break
    return rb_gen



"""
Genetic Algorithm

"""
def ga(generation):
    global saved_gen, saved_count, generation_index, all_gen, gen_start, index_indiv
    global calc_time, spent_time, total_time, mut_time, cruni_time, cr2pt_time, finish_time
    global mlgen_time, mkmlrdf_time, mkmodel_time, pred_time
    best_score = []
    best = []
    tmp_answer = 0
    print("start the Generation loop")
    print("--------------------------")
    for i in range(start_generation, GENERATION):
        bes = time.time()
        best_ind = min(generation, key=Indivisual.get_score)
        bestime = time.time() - bes
        
        if best_ind.kind == "saved_gen":
            saved_count += 1
        else:
            saved_count = 0
        best_score.append(best_ind.score)
        best.append(best_ind)
        
        gen_fin = time.time()
        spent_time_o = gen_fin - gen_start
        spent_time = "{:.2f}".format(spent_time_o)
        total_time += spent_time_o
        finish_time = datetime.datetime.now()
        
        
        print("Generation: " + str(generation_index) + "  best genom: " + str(best_ind.genom)  + "  best score: " + str(best_ind.score))
        print("num of gen: " + str(len(generation)))
        if min(best_score) == best_ind.score:
            tmp_answer = [best_ind, generation_index]
        if start_generation != 0 and generation_index == start_generation:
            pass
        else:
            make_value_file(generation)
            make_time_file()
        
        if saved_count >= STOP_CRITERIA:
            break

        gen_start = time.time()
        
        """For ML 0628"""
        #make the model for prediction
        mkmodel_start = time.time()
        global df
        #TODO: df koushin
        if mlga == True:
            os.chdir(f"{pwd}/{dirML}")
            # print(df)
            # print(df_o)
            df_new = make_model.main(generation_index, rdflist, adflist, df, rdfmode)
            os.chdir(pwd)
        mkmodel_time = time.time() - mkmodel_start
        #df = df_new
        """"""
            
        
        index_indiv = 0
        generation_index += 1

        #chose the saved genes
        saved_gen = select_elite(generation, SAVE)
        #select the genes as parents
        selected = globals()[f"select_{select_mode}"](generation, num)
        #perform crossover and make children
        children = crossover(selected, "ga")
        #perform mutation
        mut_start = time.time()
        children = mutate(selected, children, "ga")
        mut_time = "{:.2f}".format(time.time() - mut_start)

        """for ML 0628"""
        if mlga == True:
            global mlgene
            mlgene = []
            mlgen_start = time.time()
            ml_uni = crossover(selected, "ml_uni")
            ml_2pt = crossover(selected, "ml_2pt")
            ml_mu = mutate(selected, children, "ml_mu")
            mlgen_time = time.time() - mlgen_start
   
            #make gene file in dir_uni_ and make RDF ADF
            mkmlrdf = time.time()
            #FIXME: threadかprocessかも要検討
            # with ProcessPoolExecutor(max_workers=15) as tpe1:
            #     for mlmode in mllist:
            #         for i in locals()[f"ml_{mlmode}"]:
            #             tpe1.submit(Indivisual(i))
            # tpe1.shutdown()
            for mlmode in mllist:
                for i in locals()[f"ml_{mlmode}"]:
                    Indivisual(i)
            time.sleep(1)
            mkmlrdf_time = time.time() - mkmlrdf

            #pedML :make descript >> prediction
            pred_start = time.time()
            # with ProcessPoolExecutor(max_workers=len(mllist)) as tpe2:
            #     for i in mllist:
            #         tpe2.submit(prediction, i)
            #         # tpe2.result()
            # tpe2.shutdown()
            for i in mllist:
                prediction(i)
            pred_time = time.time() - pred_start
            #FIXME: ProcessPool使うと進まなかったからやめた
            # for i in mllist:
            #     prediction(i)
            # os.chdir(pwd)

            #get the top gene from ML and append the children list
            children = get_mltop(children)

        """"""

        #calc and get score
        generation = []

        #FIXME: thread使いたい
        # with ProcessPoolExecutor(max_workers=POPULATION) as tpe9:
        #     for i in children:
        #         future = tpe9.submit(Indivisual(i))
        #         generation.append(future.result)
        for i in children:
            generation.append(Indivisual(i))
        time.sleep(1)
        
        if thread == True:
            futures = []
            with ProcessPoolExecutor(max_workers=ncore) as tpe:
                for i in generation:
                    future = tpe.submit(Indivisual.calc_score, i)
                    futures.append(future)
            tpe.shutdown()
            for n, i in enumerate(futures):
                generation[n].score = i.result()
        else:
            submit_calc_energy()
            #future=Indivisual.calc_score
            for i in generation:
                i.score = Indivisual.set_energy_from_file(i)
                #with open(f"{self.cpwd}/{eval_file}", "r") as f:
                #    score = float(f.readline().rstrip())
                    # time.sleep(5)

                #if mlga == True:
                #    with open(f"{self.rdf_dir}/out.energy", "w") as w:
                #        w.write(str(score) + "\n")
        #take over the top n
        generation += saved_gen
    print("----------------------------")
    print("Finished the Generation loop\n")
    print("The best indivisual: " + "".join(tmp_answer[0].genom) + "  score: " + str(tmp_answer[0].score))
    
    return best

"""
Read the last information from the output file
>> all_gen
>> generation

"""
class Indivisual_restart:
    def __init__(self, info):
        self.genom = info[0]
        self.kind = info[1]
        self.birth = info[2]
        self.index_indiv = info[3]
        self.score = info[-1]

    def get_score(self):
        return self.score

def get_last_data():
    generation = []
    all_gen = []
    with open(value_file, "r") as g:
        g.readline()
        lines = g.readlines()
        n = POPULATION + SAVE
        tmp = lines[-n:]
        for i in lines:
            indiv = i.split()
            gen = [indiv[j] for j in range(3, 3+NUM_OF_STRINGS)]
            all_gen.append(gen)
        for i in tmp:
            indiv = i.split()
            if indiv[3+NUM_OF_STRINGS+1] == "rebirth":
                indiv = [indiv[x] for x in range(0, 3+NUM_OF_STRINGS+1)]
                indiv += ["rebirth (rdm)", i.split()[3+NUM_OF_STRINGS+3], i.split()[3+NUM_OF_STRINGS+4]]
            tmp_genom = [[indiv[j] for j in range(3, 3+NUM_OF_STRINGS)], indiv[3+NUM_OF_STRINGS], indiv[3+NUM_OF_STRINGS+1], indiv[-1], float(indiv[1])]
            indivisual = Indivisual_restart(tmp_genom)
            generation.append(indivisual)
    generation_index = int(indiv[0])
    saved_count = int(indiv[-2]) - 1
    with open("out.elapsedtime", "r") as f:
        lastline = f.readlines()[-1]
    idx = lastline.find("(outputted")
    last_time_data = float(lastline[:idx].split()[-1])
    return generation, generation_index, saved_count, all_gen, last_time_data

def get_last_df():
    output_dffilename = 'merge_table.csv'
    dfpath = f"{pwd}/{dirML}/{output_dffilename}"
    df = pd.read_csv(dfpath,sep=',',index_col=0)
    return df

def gene_2_pos(dirp, gene_file, ref_poscar):
    temp_gene_path = f"{dirp}/{gene_file}"
    with open(f"{temp_gene_path}", "r") as f:
        strings = f.read().splitlines()
    with open(ref_poscar, "r") as f:
        all_lines = f.readlines()
    all_lines = [i.split() for i in all_lines]
    labels, lines = all_lines[:8], all_lines[8:]
    if "Direct" not in labels[-1] and "Cart" not in labels[-1]:
            print("Warning : POSCAR_org must be in VASP5 format.　A label with the name of the atom is required on line 6.")
            sys.exit()

    new_labels = labels.copy()
    label_info = []
    # print(labels[5])
    for i in range(len(labels[5])):
        if "ELEM" in labels[5][i]:
            flag = labels[5][i].lstrip("ELEM")
            if flag == "":
                flag = 1
            else:
                flag = int(flag)
            for xx, x in enumerate(ELEM[flag-1]):
                count = str(str(strings[flag-1]).count(str(xx)))
                if x != "Vac":
                    label_info.append([x, count])
        else:
            label_info.append([labels[5][i], labels[6][i]])
    label_info_sort = sorted(label_info, key=lambda x: str(x[0]))
    new_labels[5] = [i[0] for i in label_info_sort]
    new_labels[6] = [i[1] for i in label_info_sort]

    count_list = [0 for _ in range(len(strings))]
    for line in lines:
        if "ELEM" in line[3]:
            flag = line[3].lstrip("ELEM")
            if flag == "":
                flag = 1
            else:
                flag = int(flag)
            gen = int(strings[flag-1][count_list[flag-1]])
            count_list[flag-1] += 1
            line[3] = ELEM[flag-1][gen]
        else:
            pass
    lines_sort = sorted(lines, key= lambda x: str(x[3]))

    with open(f"{dirp}/POSCAR", "w") as w:
        for i in new_labels:
            w.write(" ".join(i) + "\n")
        for i in lines_sort:
            if "Vac" not in i:
                w.write(" ".join(i) + "\n")            

            

"""
Perform Genetic Algorithms

"""
if __name__ == "__main__":
    check_argument()
    mode = sys.argv[1]

    if mlga == True:
        mlgene = []
    else:
        pass

    if mode == "-ga":

        """ make flag """
        calc_time = 0
        total_time = 0
        spent_time = 0
        finish_time = 0
        ltime = 0
        cr2pt_time = 0
        cruni_time = 0
        mut_time = 0
        index_indiv = 0

        mlgen_time = 0
        mkmlrdf_time = 0
        mkmodel_time = 0
        pred_time = 0

        """ Create the first generation to be calculated"""
        if RESTART == False:
            start_generation = 0
            generation_index = 0
            saved_count = 0
            all_gen = []
            gen_start = time.time()
            save_outputfile()
            if mlga == True:
                os.system(f"rm {dirML} -r")
            os.system(f"rm {pwd}/sample* -rf")
            calc_place = set_calc_place()
            generation = create_first_generation(POPULATION, genoms)      
        else:       
            #Read the last generation information from the output file
            rest = time.time()
            if os.path.isfile(value_file)==False:
                print(f"Error: no value file ({value_file}) RESTART={RESTART}")
                sys.exit()
            generation, generation_index, saved_count, all_gen, last_time_data = get_last_data()
            if mlga == True:
                df = get_last_df()
            rest_time = time.time() - rest
            with open(f"{pwd}/restart.time", "a") as w:
                w.write(f"{generation_index}generations\nrestart:{rest_time} import:{importtime} date:{datetime.datetime.now()}")
            print("output restart.time")
            total_time = last_time_data
            for i in range(1, inp_ga.POPULATION+1):
                smp_num = str(i).zfill(3)
                try:
                    os.system(f"touch {pwd}/sample{smp_num}/finish")
                    os.system(f"rm {pwd}/sample{smp_num}/{eval_file}")
                    os.system(f"rm {pwd}/sample{smp_num}/running")
                except:
                    pass
            start_generation = generation_index
            gen_start = time.time()

        gas = time.time()    
        best = ga(generation)
        

    elif mode == "-bestgene":
        if len(sys.argv) != 5:
            print(f"USAGE: python {sys.argv[0]} -bestgene Arg1 Arg2 Arg3\n  Arg1: out.value_indiv file\n  Arg2: beginning\n  Arg3: the end\n")
            sys.exit()
        valuefile = sys.argv[2]
        bigin = int(sys.argv[3])
        end = int(sys.argv[4])
        sort = make_sort_file(valuefile)
        os.system("rm best_rank* -r")
        for i in range(bigin, end+1):
            indiv = sort[i-1]
            new_dir = f"{pwd}/best_rank{str(i)}_gen{str(indiv[0])}_indiv{str(indiv[-1])}"
            os.system(f"mkdir {new_dir}")
            os.system(f"cp {pwd}/sample{str(indiv[-1]).zfill(3)}/Save*_{str(indiv[0])} {new_dir}")
            gene_file = f"Save_temp_gene_{str(indiv[0])}"
            ref_poscar = "./Specific/POSCAR_org"
            try:
                gene_2_pos(new_dir, gene_file, ref_poscar)
            except:
                pass
        
    elif mode == "-gene2pos":
        if len(sys.argv) != 4:
            print(f"USAGE: python {sys.argv[0]} -gene2pos Arg1 Arg2\n  Arg1: directory name\n  Arg2: gene file\n")
            sys.exit()
        dirp = sys.argv[2]
        gene_file = sys.argv[3]
        ref_poscar = "./Specific/POSCAR_org"
        gene_2_pos(dirp, gene_file, ref_poscar)
        
       
    else:
        print(f"{sys.argv[1]} is not defined")
        print(f"USAGE: {sys.argv[0]} -mode\n ----mode----\n  -ga: run the genetic algorithm\n  -bestgene: get the best gene and create new directory\n  -gene2pos: get the POSCAR from gene\n")

