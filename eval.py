from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import pickle 
import time
import onnxmltools
import os
import random
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import onnxruntime as rt
import onnx
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance
from time import time
import argparse
# from util import feature_partition, likelihood, kl, cat_process, sample_pt
from data_process import read_iris, read_mkt, read_rank, read_wine
import torch
from train import model_train
from metric import align_dict, sort_dict, compare
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='mkt', help='')
parser.add_argument('--res_dir', type=str, default='/data/code_yang/paper/mkt/res/para', help='')
parser.add_argument('--save_name', type=str, default='shap_score.eps', help='')
# parser.add_argument('--shap_dir', type=str, default='/data/code_yang/paper/rank/res/shap_val.npy', help='')
# parser.add_argument('--base_dir', type=str, default='/data/code_yang/paper/mkt/res', help='')
parser.add_argument('--kl_dir', type=str, default='/data/code_yang/paper/mkt/res/para/compare_kl_01.npy', help='')
parser.add_argument('--kl_ind_dir', type=str, default='/data/code_yang/paper/mkt/res/para/kl_ind_01.npy', help='')

#parser.add_argument('--ft_num', type=int, default=4, help='')  
parser.add_argument('--save_result', type=bool, default=True, help='')
parser.add_argument('--x_dir', type=str, default='/data/code_yang/paper/mkt/rank/2021-01-27 03:00:25.556306', help='')
parser.add_argument('--title', type=str, default='feature importance for rank dataset', help='')
parser.add_argument('--metric', type=str, default='normalised accuracy', help='')
parser.add_argument('--shap', type=bool, default=False, help='')
parser.add_argument('--train', type=bool, default=True, help='')
parser.add_argument('--data_dir_iris', type=str, default='/data/code_yang/paper/iris/res', help='')
parser.add_argument('--data_dir_rank', type=str, default='/data/code_yang/paper/rank/model_1/position2', help='')  
parser.add_argument('--data_dir_mkt', type=str, default='/data/code_yang/paper/mkt/bank-full.csv', help='')
parser.add_argument('--data_dir_wine', type=str, default="/data/code_yang/paper/wine/wine.data", help='')
parser.add_argument('--model_dir_mkt', type=str, default='/data/code_yang/paper/mkt/model_2.pkl', help='')
parser.add_argument('--model_dir_iris', type=str, default='/data/code_yang/paper/iris/iris.pkl', help='')  
parser.add_argument('--model_dir_wine', type=str, default='/data/code_yang/paper/wine/wine.pkl', help='')  
parser.add_argument('--model_dir_rank', type=str, default='/data/code_yang/paper/rank/model_1/delete_feature.pkl', help='')



args = parser.parse_args()

def mk_dict(res, ftt):
    score_log={}
    for i, ft in enumerate(ftt):
        # print(ft[0]+ft[1])
        score_log[ft]=res[i]
    return score_log

def sub_model_metric(args, ttsize, rm, md_choice, ev, com_num):
    y=df['label'].values
    ft_name_arr=np.array(ft_name)
    metric=[]
    ft_columns=[]
    ft_names=[]
    index_list = list(itertools.combinations(range(len(ft_name)), com_num))
    index_map = dict(zip(index_list, range(len(index_list))))
    for ii in index_list:
        ft=ft_name_arr[tuple([ii])]
        X=df[ft].values.reshape((-1, com_num))
        ff=model_train(X, y, ttsize, rm, md_choice, ev)
        ft_columns.append(index_map[ii])
        ft_names.append(ft)
        metric.append(ff)
    return metric, ft_columns, index_map, ft_names

def normalise(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr)+1e-8)

def mk_df(met_df,name, arr):
    met_df.loc[met_df.shape[0]]=np.zeros((met_df.shape[1],))
    met_df.iloc[met_df.shape[0]-1,1:]=arr
    met_df.iloc[met_df.shape[0]-1,0]=name
    return met_df
    

global num, sel_num, alpha, ft_name, range_log, cat_ft, num_ft, onnx_path, columns, model, NAME, sta

if args.name=='mkt':
    df, ft_name, cat_ft, num_ft, ft_num=read_mkt(args)
if args.name=='wine':
    df, ft_name, cat_ft, num_ft, ft_num=read_wine(args)
if args.name=='iris':
    df, ft_name, cat_ft, num_ft, ft_num=read_iris()
if args.name=='rank':
    df, ft_name, cat_ft, num_ft, ft_num=read_rank(args)

#onnx_path = '/data/code_yang/paper/rank/model_1/delete_feature.pkl'
#f=open(onnx_path, 'rb')
#model=pickle.load(f)
# now_time = datetime.datetime.now()
#path = '/data/code_yang/paper/iris/res'


# if args.shap==True:
#     score=np.load(args.shap_dir)
#     print(score)
# else:
#     x0=np.load(args.x_dir+'/x0.npy')
#     x1=np.load(args.x_dir+'/x1.npy')
# #     x2=np.load(args.base_dir+'/x_2.npy')
#     [res_kl, res_js, res_he]=compare(x0, x1, ft_num)
#     #res2=compare(x0, x2, ft_num)
#     #res3=compare(x2, x1, ft_num)
#     score_kl=np.mean(res_kl, axis=1) #+res2+res3
#     val_kl=np.std(res_kl, axis=1)
#
#     score_js=np.mean(res_js, axis=1) #+res2+res3
#     val_js=np.std(res_js, axis=1)
#
#     score_he=np.mean(res_he, axis=1) #+res2+res3
#     val_he=np.std(res_he, axis=1)
    
if args.train==True:
    met, ft_col, index_map, ft_names = sub_model_metric(args, ttsize=0.3, rm=42, md_choice='lr', ev='acc', com_num=2)
    metric_log=mk_dict(met, ft_col)
    metric_df = pd.DataFrame(metric_log, index=[0])
    kl_arr = np.load(args.kl_dir)
    kl_ind_arr = np.load(args.kl_ind_dir)
    kl_col=[index_map[tuple(kl_ind_arr[i])] for i in range(len(kl_ind_arr))]
    kl_ss = np.zeros((metric_df.shape[1],))
    kl_ss[kl_col] = kl_arr.reshape((-1,))
    metric_df.loc[metric_df.shape[0]] = normalise(kl_ss)
    metric_df.loc[metric_df.shape[0]] = ft_names
    metric_df = metric_df.sort_values(by=0, axis=1, ascending=False)
    metric_df.to_csv(args.res_dir + '/' + 'sub_metric_score.csv')
    # metric_log=sort_dict(metric_log)
    # score_log=mk_dict(score, ft_name)
    # score_=align_dict(metric_log, score_log)
    # if args.save_result==True:
    #     tmp=pd.DataFrame(metric_log, index=['metric'])
    #     tmp.loc['score']=score_
    #     tmp.to_csv(args.res_dir+'/'+'sub_metric_score.csv')
else:
    met_df=pd.read_csv(args.res_dir+'/'+'metric_score.csv', nrows=3, index_col=0)
    met=met_df.iloc[0,1:].values
    metric_log=mk_dict(met, list(met_df.columns[1:]))
    
    score_log_kl=mk_dict(score_kl, ft_name)
    val_log_kl=mk_dict(val_kl, ft_name)
    
    score_log_js=mk_dict(score_js, ft_name)
    val_log_js=mk_dict(val_js, ft_name)
    
    score_log_he=mk_dict(score_he, ft_name)
    val_log_he=mk_dict(val_he, ft_name)
    
    
    score_kl_=align_dict(metric_log, score_log_kl)
    val_kl_=align_dict(metric_log, val_log_kl)
    
    score_js_=align_dict(metric_log, score_log_js)
    val_js_=align_dict(metric_log, val_log_js)
    
    score_he_=align_dict(metric_log, score_log_he)
    val_he_=align_dict(metric_log, val_log_he)
    
    print('---------------')
    print(met_df.head())
    print(score_kl_.shape)
    
    namelist=['score_kl', 'val_kl', 'score_js', 'val_js', 'score_he', 'val_he']
    arr_list=[score_kl_, val_kl_, score_js_, val_js_, score_he_, val_he_]
    
    for i in range(len(namelist)):
        met_df=mk_df(met_df, namelist[i], arr_list[i])
    
#     met_df.loc[met_df.shape[0]]=np.zeros((met_df.shape[1],))
#     met_df.iloc[met_df.shape[0]-1,1:]=score_kl_
#     met_df.iloc[met_df.shape[0]-1,0]='score_kl'
   
        
#     met_df.loc['val_kl']=val_kl_
#     met_df.loc['score_js']=score_js_
#     met_df.loc['val_js']=val_js_
#     met_df.loc['score_he']=score_he_
#     met_df.loc['val_he']=val_he_
    met_df.to_csv(args.res_dir+'/'+'metric_score_1.csv', index=False)
    #np.save(args.res_dir+'/om_score.npy', score)
    #metric_log=sort_dict(metric_log)


# metric_log=mk_dict(args, met)
# metric_log=sort_dict(metric_log)
# print(metric_log)
# print('------ metric -------')

# met_nm=normalise(np.array(list(metric_log.values())))
# sc_nm=normalise(np.array(score_))
# plt.plot(range(ft_num), met_nm)
# plt.plot(range(ft_num), sc_nm)
# plt.title(args.title)
# plt.xlabel('feature index')
# plt.ylabel('measure')
# plt.legend([args.metric, 'normalised score'])
# plt.savefig(args.res_dir+'/pic/'+args.save_name)
    
    
    
