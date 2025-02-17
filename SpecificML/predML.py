#!/usr/bin/env python3
# coding: utf-8
"""
for GAML
2023/07/11 yokoyama

"""
#TODO: sortファイルに出すいindexの表示：1_001とかにする
#TODO: calcenergyをnewdir_にコピー（GmAteの中で）
#TODO: calcenergyの中身RDFとか作れるようにする？GmAteとどっちがいい？

import os
import pandas as pd
import numpy as np
import pickle
import time
import glob
from natsort import natsorted
from sklearn.ensemble import RandomForestRegressor
from ase.geometry.analysis import Analysis
from ase.io import read

def makedescript_vasp(generation, eval_file, rdfn, rdflist, adflist, df):
    pwd = os.getcwd()
    df_t = df.copy()
    print(pwd)
    if eval_file == "none":
        dirflag = f"{generation}_*"
    else:
        dirflag = f"dir_{generation}_*"
    dirlist = glob.glob(dirflag)
    dirlist = natsorted(dirlist)

    count = 0
    for i in dirlist:
        count += 1
        for ii in adflist:
            filename = f"{pwd}/{i}/out.{ii}"
            # print(filename)
            try:
                try:
                    with open(filename, "r") as f:
                        for k in range(1, 60+1):
                            adfv = f.readline().split()[1]
                            df_t.loc[i, f"{ii}_{str(k)}"] = adfv
                except:
                    ii_list = ii.replace("adf_", "").split("-")
                    iii = f"adf_{ii_list[2]}-{ii_list[1]}-{ii_list[0]}"
                    filename = f"{pwd}/{i}/out.{iii}"
                    with open(filename, "r") as f:
                        for _ in range(1, 60+1):
                            adfv = f.readline().split()[1]
                            df_t.loc[i, f"{ii}_{str(k)}"] = adfv
            except:
                print(f"file {filename} is not exist")

        for ii in rdflist:
            filename = f"{pwd}/{i}/out.{ii}"
            n = 19
            if "all" in ii:
                n = 20
                if rdfn > 3:
                    rdfnn = -1
            else:
                rdfnn = rdfn - 1
            try:
                with open(filename, "r") as f:
                    for k in range(1, n+1):
                        rdfv = f.readline().split()[rdfnn]
                        df_t.loc[i, f"{ii}_{str(k)}"] = rdfv
            except:
                print(f"file {filename} is not exist")

                
        if eval_file != "none":
            with open(f"{pwd}/{i}/{eval_file}", "r") as f:
                ans = float(f.readline().rstrip())
            df_t.loc[i, "energy"] = ans
        else:
            pass

    return df_t

def makedescript_ase(generation, eval_file, rdflist, adflist, df):
    rmax = 4
    nbins = 20
    df_t = df.copy()
    df_this = df.copy()
    pwd = os.getcwd()
    print(pwd)
    if eval_file == "none":
        dirflag = f"{generation}_*"
    else:
        dirflag = f"dir_{generation}_*"
    dirlist = glob.glob(dirflag)
    dirlist = natsorted(dirlist)

    count = 0
    for i in dirlist:
        count += 1
        inpf = f"{pwd}/{i}/POSCAR"
        atoms = read(inpf, format="vasp")
        analysis = Analysis(atoms)
        
        radf_ll = []
        label = []
        # for ii in adflist:
        #     if "all" not in ii:
        #         nelem = tuple(ii.replace("adf_", "").split("-"))
        #     else:
        #         nelem = None
        #     adf = adflist
        #     for k in range(1, len(rdf[0])+1):
        #         df_t.loc[i, f"{ii}_{str(k)}"] = adf[0][k-1]
        
        for ii in rdflist:
            if "all" not in ii:
                nelem = tuple(ii.replace("rdf_", "").split("-"))
            else:
                nelem = None
            rdf = analysis.get_rdf(rmax, nbins, elements=nelem)
            for k in range(1, len(rdf[0])+1):
                label.append(f"{ii}_{str(k)}")
                # df_this.loc[i, f"{ii}_{str(k)}"] = rdf[0][k-1]  
            rdf_l = rdf[0].tolist()
            for r in rdf_l:
                radf_ll.append(r)
        df_this = pd.DataFrame([radf_ll], 
                columns=label,
                index=[i] )
        df_t = pd.concat([df_t, df_this])
                
        if eval_file != "none":
            with open(f"{pwd}/{i}/{eval_file}", "r") as f:
                ans = float(f.readline().rstrip())
            df_t.loc[i, "energy"] = ans
        else:
            pass
        # df_t = pd.concat([df_t, df_this])
        df_this = df.copy()

    return df_t


def main(generation, rdflist, adflist, df_o, rdfmode):
    # print(df_o)
    pwd =os.getcwd()
    os.system("rm loadRF predy -r")
    start = time.time()

    if rdfmode == "vasp":
        df_t = makedescript_vasp(generation, "none", 4, rdflist, adflist, df_o)
    elif rdfmode == "ase":
        print("make descript")
        df_t = makedescript_ase(generation, "none", rdflist, adflist, df_o)
    else:
        pass

    tabletime = time.time() - start

    with open("table_time", "w") as w:
        w.write(f"{generation} {tabletime}\n")

    # input_filename = 'merge.table_schema'
    output_filename = 'merge_table.csv'

    # df = pd.read_csv(input_filename,sep=' ',index_col=0)
    # print(df.head())
    df_t.to_csv(output_filename, sep=',')
    print(f'outputfile:{output_filename}')


    filename = 'merge_table.csv' 

    # try:
    #     df = pd.read_csv(pwd +'/'+ filename,encoding="shift-jis",index_col=0)
    #     print('Read .csv file')    
    # except:
    #     df = pd.read_excel(pwd +'/'+ filename,encoding="shift-jis",index_col=0)
    #     print('Read .xlsx file')
    # print(df.head())

    # df
    df_with = df_t.dropna(how='all',axis=1)
    df_with_ans = df_with.dropna(how='any',axis=0) 
    n,d = df_with_ans.shape
    print('\nThe number of sample with answer =',n,'\nThe number of columns = ',d)

    with open('../RandomForest_model.pickle',mode='rb') as fp:
        loaded_model = pickle.load(fp)
    os.mkdir("loadRF")
    pred_y = loaded_model.predict(df_with_ans)
    os.mkdir("predy")

    df_with_ans['pred'] = pred_y
    df_sort = df_with_ans.sort_values('pred')
    l_index = list(df_sort.index)

    sort_label = []
    for i in l_index:
        lab = i.rstrip("/")
        labe = lab.split("/")[-1]
        # lab = i.split('_')[2]
        # labe = lab.split('/')[0]
        sort_label.append(labe)


    sort_label_text = "\n".join(sort_label)
    with open('sort_label.out','w') as f:
        f.write(sort_label_text)
    print("finish")

