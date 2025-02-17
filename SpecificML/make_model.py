#!/usr/bin/env python3
# coding: utf-8
"""
for GAML
2023/07/11 yokoyama

"""

import os
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import glob
from natsort import natsorted
import time
from ase.geometry.analysis import Analysis
from ase.io import read


def rmse(y_true,y_pred):
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    return rmse

def make_hist(df,column_name1,column_name2=None):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(221)
    plt.hist(df[column_name1],bins=30)
    plt.title('Histogram plot {}'.format(column_name1))
    plt.ylabel('Count')
    plt.xlabel(column_name1)
    
    if column_name2:
        ax = fig.add_subplot(222)
        plt.hist(df[column_name2],bins=30)
        plt.title('Histogram plot {}'.format(column_name2))
        plt.ylabel('Count')
        plt.xlabel(column_name2)
        plt.show()

def test_prediction(model,X_train,X_test,y_train,y_test):
    if model.__class__.__name__ == 'Booster':
        pred_test = model.predict(X_test)       
    else:
        model.fit(X_train,y_train)
        pred_test = model.predict(X_test)
   # make_scatter_single(y_test, pred_test,'test',evaluation_name='{} Test'.format(model.__class__.__name__))
    test_rmse = rmse(y_test,pred_test)

    with open('y_test.out','w')as y:
        print(y_test,file=y)
    with open('pred_test.out','w')as p:
        print(pred_test,file=p)
        
    return test_rmse,pred_test

def diagnosis_plot(x, y,generation, color=None, s=3, dpi=600):
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    #target
    x = x
    y = y
    #rcparams プログラム全体のグラフに対して設定を変更できる。
    plt.rcParams['font.family'] = 'arial'
    rmse = np.sqrt(mean_squared_error(x, y))
    r2 = r2_score(x, y)
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    if x_min <= y_min:
        axis_lim_min = x_min
    elif x_min > y_min:
        axis_lim_min = y_min
    if x_max <= y_max:
        axis_lim_max = y_max
    elif x_max > y_max:
        axis_lim_max = x_max
    width = abs(axis_lim_max - axis_lim_min)
    p = np.linspace(axis_lim_min - width, axis_lim_max + width, 2)
    q = p
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y, s=s, color = color)
    ax.plot(p, q, color='black', alpha=0.5)
    ax.set_xlim(axis_lim_min - width*0.1, axis_lim_max + width*0.1)
    ax.set_ylim(axis_lim_min - width*0.1, axis_lim_max + width*0.1)
    boxdic = {
    "facecolor" : "whitesmoke",
    "edgecolor" : "gray",
    "boxstyle" : "square",
    "linewidth" : 1}
    s =f'RMSE : {rmse:.3}' + '\n' + f'R2 score : {r2:.3}'
    ax.text(axis_lim_min - width*0.05, axis_lim_max - width*0.05, bbox=boxdic, s=s)
    ax.set_xlabel('True')
    ax.set_ylabel('Prediction')
    ax.set_title('Diagnosis plot')
    fig.savefig(f'diag_{generation}.png',dpt=600)
    ax.grid()

def makedescript_vasp(generation, eval_file, rdfn, rdflist, adflist, df):
    df_t = df.copy()
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
            # with open(filename, "r") as f:
            #     for k in range(1, n+1):
            #         rdfv = f.readline().split()[rdfnn]
            #         df_t.loc[i, f"{ii}_{str(k)}"] = rdfv

                
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
                df_t.loc[i, f"{ii}_{str(k)}"] = rdf[0][k-1]  
                
        if eval_file != "none":
            with open(f"{pwd}/{i}/{eval_file}", "r") as f:
                ans = float(f.readline().rstrip())
            df_t.loc[i, "energy"] = ans
        else:
            pass

    return df_t

import warnings
warnings.filterwarnings('ignore')

def main(generation, rdflist, adflist, df_o, rdfmode):
    pwd=os.getcwd()

    start = time.time()
    if rdfmode == "vasp":
        df_t = makedescript_vasp(generation, "out.energy", 4, rdflist, adflist, df_o)
    elif rdfmode == "ase":
        df_t = makedescript_ase(generation, "out.energy", rdflist, adflist, df_o)
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
    #target_column = 'Unnamed: 2878' #maybe table.energy.~~~~

    # df_columns = list(df.columns)
    # for i in df_columns:
    #     if ':' in i:
    #         target_column=i
    target_column = "energy"

    df_with = df_t.dropna(how='all',axis=1) 
    df_with_ans = df_with.dropna(how='any',axis=0) 
    n,d = df_with_ans.shape
    print('\nThe number of sample with answer =',n,'\nThe number of columns = ',d)

    
    make_hist(df_with_ans,target_column)
    plt.savefig("histogra.png")
    
    X = df_with_ans.drop(target_column,axis=1)
    y = df_with_ans[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

    n_split=10
    kf = KFold(n_splits=n_split,shuffle=True,random_state=777)

    best_rmse_for_each_model_d = {}
    best_param_for_each_model_d = {}
    model_d = {}

    # from sklearn.ensemble import RandomForestRegressor

    parameters= {'max_depth':[5,10],
                'min_samples_leaf':[1,2,4],
                'n_estimators':[10,30],
                'Is_scalers':[False]
                }

    best_val_rmse = 999
    for n_estimators in parameters['n_estimators']:
        for max_depth in parameters['max_depth']:
            for min_samples_leaf in parameters['min_samples_leaf']:
                for Is_scaler in parameters['Is_scalers']:        
                    pred_validation = np.array([])
                    ans_validation = np.array([])
                    pred_train = np.array([])
                    ans_train = np.array([])         
                    for train_index,test_index in kf.split(X_train):
                        X_tr = X_train.iloc[train_index]
                        y_tr = y_train.iloc[train_index]
                        X_va = X_train.iloc[test_index]
                        y_va = y_train.iloc[test_index]
                        if Is_scaler:
                            scaler = StandardScaler()
                            scaler.fit(X_tr)
                            X_tr = scaler.transform(X_tr)
                            X_va = scaler.transform(X_va)
                        clf = RandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                                        n_estimators=n_estimators, n_jobs=-1, random_state=42)
                        clf.fit(X_tr,y_tr)
                        pred_val_y = clf.predict(X_va)
                        pred_tra_y = clf.predict(X_tr) 
                        
                        pred_validation = np.append(pred_validation,pred_val_y)
                        ans_validation = np.append(ans_validation,y_va)
                        pred_train = np.append(pred_train,pred_tra_y)
                        ans_train = np.append(ans_train,y_tr)
        
                    val_rmse = rmse(pred_validation,ans_validation)
                    tra_rmse = rmse(pred_train,ans_train)
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_tra_rmse = tra_rmse
                        best_predict={
                            'ans_train':ans_train,
                            'pred_train':pred_train,
                            'pred_validation':pred_validation,
                            'ans_validation':ans_validation,
                        }
                        best_params={
                            'max_depth':max_depth,
                            'min_samples_leaf':min_samples_leaf,
                            'n_estimators':n_estimators,
                            'Is_scaler':Is_scaler                                
                        }
                    print('max_depth: {}, min_samples_leaf:{}, n_estimators:{}, Is_Scaler: {}, train loss: {}, valid loss: {}'
                        .format(max_depth,min_samples_leaf,n_estimators,Is_scaler,round(tra_rmse,3),round(val_rmse,3)))

    print('Best params:')
    print('max_depth: {}, min_samples_leaf:{}, n_estimators:{}, Is_Scaler: {}, train loss: {}, valid loss: {}'
        .format(best_params['max_depth'],best_params['min_samples_leaf'],best_params['n_estimators'],best_params['Is_scaler']
                ,round(best_tra_rmse,3),round(best_val_rmse,3)))
    with open('save_rmse.out','a') as f:
        print('max_depth: {}, min_samples_leaf:{}, n_estimators:{}, Is_Scaler: {}, train loss: {}, valid loss: {}'
        .format(best_params['max_depth'],best_params['min_samples_leaf'],best_params['n_estimators'],best_params['Is_scaler']
                ,round(best_tra_rmse,3),round(best_val_rmse,3)),file=f)

    
    #make_scatter_single(best_predict['ans_validation'],best_predict['pred_validation'],'validation',evaluation_name='Validation')

    with open('ans_validation.out','w') as a:
        print(best_predict['ans_validation'],file=a)
    with open('pred_validation.out','w')as p:
        print(best_predict['pred_validation'],file=p) 


    model =  RandomForestRegressor(max_depth=best_params['max_depth'], min_samples_leaf=best_params['min_samples_leaf'], 
                                                    n_estimators=best_params['n_estimators'], n_jobs=-1, random_state=42)


    model_name = 'RandomForest'
    model_d[model_name] = model
    best_rmse_for_each_model_d[model_name] = best_val_rmse
    best_param_for_each_model_d[model_name] = best_params

    # def test_prediction(model,X_train=X_train,y_train=y_train,y_test=y_test):
    #     if model.__class__.__name__ == 'Booster':
    #         pred_test = model.predict(X_test)       
    #     else:
    #         model.fit(X_train,y_train)
    #         pred_test = model.predict(X_test)
    #    # make_scatter_single(y_test, pred_test,'test',evaluation_name='{} Test'.format(model.__class__.__name__))
    #     test_rmse = rmse(y_test,pred_test)

    #     with open('y_test.out','w')as y:
    #         print(y_test,file=y)
    #     with open('pred_test.out','w')as p:
    #         print(pred_test,file=p)

    #     return test_rmse,pred_test
    
    test_rmse_d = {}
    for k, model in zip(model_d.keys(),model_d.values()):
        test_rmse,pred_test = test_prediction(model, X_train, X_test, y_train, y_test)
        #print(test_rmse)
        test_rmse_d[model.__class__.__name__] = test_rmse

    max_kv = min(test_rmse_d,key=test_rmse_d.get)

    y_test_lst = y_test.values.tolist()
    # from sklearn.metrics import r2_score
    q2_score = r2_score(y_test_lst, pred_test) 
    rmse_score = rmse(y_test,pred_test)
    diagnosis_plot(y_test,pred_test,generation)

    with open('test_rmse.out','a') as a:
        print(float(rmse_score),file=a)

    with open('test_score.out','a') as a:
        print(float(q2_score),file=a)

    with open('save_test_score.out','a') as a:
        print(float(q2_score),file=a)

    # import pickle
    learning_all_data_model_d = {}
    for model_name, model in zip(model_d.keys(),model_d.values()):
        
        if best_param_for_each_model_d[model_name]['Is_scaler']:
            scaler = StandardScaler()
            scaler.fit(X)
            X_sc = scaler.transform(X)
            reg = model.fit(X_sc,y)
        
        else:
            if model.__class__.__name__ == 'Booster':
                lgb_train = lgb.Dataset(X_train, y_train)
                params = {
                        'objective': 'regression',
                        'min_data_in_leaf': lgb_best_params['min_data_in_leaf'],
                        'learning_rate': 0.001,
                        'num_leaves': int(lgb_best_params['num_leaves'])
                        }
                reg =  lgb.train(params, lgb_train,
                            verbose_eval=False,
                            num_boost_round=int(average_best_iteraion),
                        )
            else:
                reg = model.fit(X,y)
        learning_all_data_model_d[model_name] = reg  

    for model_name, model in zip(learning_all_data_model_d.keys(),learning_all_data_model_d.values()):
        output_name = model_name+'_model.pickle'
        with open(output_name,mode='wb') as fp:
            pickle.dump(model,fp)
        print('output  filename:',output_name)

    return df_t

