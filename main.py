# from onnxmltools.convert.common.data_types import FloatTensorType
# from onnxmltools.convert import convert_xgboost
import pickle 
import time
import os
import random
# from sklearn.utils import shuffle
import numpy as np

import datetime
import argparse
from util import feature_partition, compare, cat_process, sample_pt_v2, feature_partition_ctr, DeepFM, bound_fd
from data_process import read_iris, read_rank, read_mkt, read_wine, read_gaussian, read_integer, read_ctr, read_ctr_em
import torch
from metric import happening_prob, prob_compare, prob_compare_1, symkl, hellinger, js
# from sko.GA import GA
# from sko.SA import SA
# from sko.PSO import PSO

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='ctr_em', help='')
parser.add_argument('--base_dir', type=str, default='/data/code_yang/paper/ctr_em/res', help='')
parser.add_argument('--num', type=int, default=200, help='number of sample')
parser.add_argument('--alpha', type=float, default=1e-2, help='')
parser.add_argument('--popp', type=int, default=100, help='')
parser.add_argument('--itrr', type=int, default=40, help='')
parser.add_argument('--M', type=int, default=3, help='')
parser.add_argument('--lb', type=float, default=1e-1, help='') 
parser.add_argument('--up', type=float, default=20, help='') 
parser.add_argument('--label_str', type=str, default=['0','1'], help='')
parser.add_argument('--data_dir_gaussian', type=str, default='/data/code_yang/paper/synth_gausssian/data.csv', help='')
parser.add_argument('--data_dir_integer', type=str, default='/data/code_yang/paper/synth_int/data.csv', help='')
parser.add_argument('--data_dir_rank', type=str, default='/data/code_yang/paper/rank/model_1/position2', help='')  
parser.add_argument('--data_dir_mkt', type=str, default='/data/code_yang/paper/mkt/bank-full.csv', help='')  
parser.add_argument('--data_dir_wine', type=str, default="/data/code_yang/paper/wine/wine.data", help='')
parser.add_argument('--data_dir_ctr', type=str, default="/data/code_yang/paper/ctr/tj_criteo.csv", help='')
parser.add_argument('--data_dir_ctr_em', type=str, default="/data/code_yang/paper/ctr/tj_criteo_cat.csv", help='')

parser.add_argument('--model_dir_mkt', type=str, default='/data/code_yang/paper/mkt/model_2.pkl', help='')  
parser.add_argument('--model_dir_iris', type=str, default='/data/code_yang/paper/iris/iris.pkl', help='')  
parser.add_argument('--model_dir_wine', type=str, default='/data/code_yang/paper/wine/wine.pkl', help='')  
parser.add_argument('--model_dir_rank', type=str, default='/data/code_yang/paper/rank/model_1/delete_feature.pkl', help='')
parser.add_argument('--model_dir_gaussian', type=str, default='/data/code_yang/paper/synth_gausssian/lr_gs.pickle', help='')
parser.add_argument('--model_dir_integer', type=str, default='/data/code_yang/paper/synth_int/lr_int.pickle', help='')
parser.add_argument('--model_dir_ctr', type=str, default='/data/code_yang/paper/ctr/criteo780770.pth', help='')
parser.add_argument('--model_dir_ctr_em', type=str, default='/data/code_yang/paper/ctr/Cat_7857.pth', help='')


global num, sel_num, alpha, ft_name, range_log, cat_ft, num_ft, onnx_path, columns, model, NAME, sta


args = parser.parse_args()
Name=args.name
num=args.num
alpha = args.alpha
popp=args.popp
itrr=args.itrr
M=args.M

def obj_mut1(x):  ### mutate ###
    if Name=='ctr' or Name=='ctr_em':
        range_log = {}
        for i in range(len(cat_ft)):
            X = df[cat_ft[i]].iloc[[0,1,4,5]]
            a = X.min()
            b = X.max()
            range_log[cat_ft[i]] = [a, b]
    xx, cat_ind = sample_pt_v2(df[df['label']==1], x, num, cat_ft, num_ft, ft_name, range_log)
    if Name=='rank':
        p = model.predict(xx.astype(np.float32))
    elif Name=='mkt':
        X = torch.from_numpy(xx.astype(np.long))
        p=model(X)[0][0][1].detach().numpy()
    elif Name=='ctr':
        p=model(torch.LongTensor(xx[:,13:]), torch.FloatTensor(xx[:,:13])).detach().numpy()
    elif Name=='ctr_em':
        p=model(torch.LongTensor(xx)).detach().numpy()
    else:
        p = model.predict_proba(xx.astype(np.float32))[1]

    sta_=sta['1']
    div_=0
    for i, ft in enumerate(ft_name):
        mu1 = sta_[ft][0]
        sig1 = sta_[ft][1]
        mu2 = x[2*i]
        sig2 = x[2*i+1]
        div_+=hellinger(mu1, sig1, mu2, sig2)
    # print('ppppppppp')
    # print(-np.mean(np.log(1e-8+p)))
    # print((alpha/ft_num)*div_)
    return -np.mean(np.log(1e-8+p))+(alpha/ft_num)*div_

def obj_mut0(x):  ### mutate ###
    if Name=='ctr' or Name=='ctr_em':
        range_log = {}
        for i in range(len(cat_ft)):
            X = df[cat_ft[i]].iloc[[0,1,4,5]]
            a = X.min()
            b = X.max()
            range_log[cat_ft[i]] = [a, b]
    xx, cat_ind = sample_pt_v2(df[df['label'] == 0], x, num, cat_ft, num_ft, ft_name, range_log)
    if Name=='rank':
        p = 1-model.predict(xx.astype(np.float32))
    elif Name=='mkt':
        X=torch.from_numpy(xx.astype(np.long))
        p=model(X)[0][0][0].detach().numpy()
    elif Name=='ctr':
        p=model(torch.LongTensor(xx[:,13:]), torch.FloatTensor(xx[:,:13])).detach().numpy()
    elif Name=='ctr_em':
        p=model(torch.LongTensor(xx)).detach().numpy()
    else:      
        p = model.predict_proba(xx.astype(np.float32))[0]       
    sta_=sta['0']
    div_=0
    for i, ft in enumerate(ft_name):
        mu1 = sta_[ft][0]
        sig1 = sta_[ft][1]
        mu2 = x[2*i]
        sig2 = x[2*i+1]
        div_+=hellinger(mu1, sig1, mu2, sig2)
    return -np.mean(np.log(1e-8+p))+(alpha/ft_num)*div_

def obj_mut2(x): # only for iris, wine
    # ttt = shuffle(df[df['label']==2])
    # xx, cat_ind=sample_pt(ttt, x, num, sel_num, cat_ft, num_ft, ft_name, range_log, sta['2'])
    xx, cat_ind = sample_pt_v2(df[df['label'] == 2], x, num, cat_ft, num_ft, ft_name, range_log)
    # for i, name in enumerate(ft_name):
    #     tmp = ttt[name][:sel_num]
    #     xx[num-sel_num:, i]=tmp
    
    ## redone the standardization
#     for i in range(len(cat_ft)):
#         kk=int(cat_ind[i])
#         ft_= ft_name[kk]
#         ft_min=range_log[ft_][0]
#         ft_max=range_log[ft_][1]
#         #print(xx[:num-sel_num,kk])
#         xx[:num-sel_num,kk]=ft_min+xx[:num-sel_num,kk]*(ft_max-ft_min)

#     onnx_model = onnx.load(onnx_path)
#     sess = rt.InferenceSession(onnx_path)
#     input_name = sess.get_inputs()[0].name
#     label_name = sess.get_outputs()[0].name
#     y=sess.run([label_name], {input_name: xx.astype(np.float32)})[0]
#     p=1/(1+np.exp(-y))

    p = model.predict_proba(xx.astype(np.float32))[2]       
    sta_=sta['2']
    div_=0
    for i, ft in enumerate(ft_name):
        mu1 = sta_[ft][0]
        sig1 = sta_[ft][1]
        mu2 = x[2*i]
        sig2 = x[2*i+1]
        div_+=hellinger(mu1, mu2, sig1, sig2)

    return -np.mean(np.log(1e-8+p))+(alpha/ft_num)*div_

if args.name=='mkt':
    model=torch.load('/data/code_yang/paper/mkt/model_2.pkl')
    df, ft_name, cat_ft, num_ft, ft_num=read_mkt(args)
if args.name=='wine':
    with open(args.model_dir_wine, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num=read_wine(args)
if args.name=='iris':
    with open(args.model_dir_iris, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num=read_iris()
if args.name=='rank':
    with open(args.model_dir_rank, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num=read_rank(args)

if args.name=='gaussian':
    with open(args.model_dir_gaussian, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num=read_gaussian(args)

if args.name=='integer':
    with open(args.model_dir_integer, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num=read_integer(args)

if args.name=='ctr':
    cate_fea_nuniqs=[1036,526,177397,75762,234,14,10165,482,3,23499,4501,152106,3029,
          26,7825,120291,10,3512,1704,4,138137,14,15,29183,63,22354]
    model = DeepFM(cate_fea_nuniqs)
    model = torch.load(args.model_dir_ctr, map_location='cpu')
    df, ft_name, cat_ft, num_ft, ft_num = read_ctr(args)

if args.name == 'ctr_em':
    cate_fea_nuniqs= [1036,526,177397,75762,234,14,10165,482,3,23499,4501,152106,3029,26,7825,120291,10,3512,1704,4,138137,14,15,
     29183,63,22354,23,100,63,23,102,102,73,43,102,3,13,13,33]
    model = DeepFM(cate_fea_nuniqs)
    model = torch.load(args.model_dir_ctr_em, map_location='cpu')
    df, ft_name, cat_ft, num_ft, ft_num = read_ctr_em(args)


# lb_=args.lb*np.ones((ft_num*2,))
# ub_=args.up*np.ones((ft_num*2,))
def run_and_rcd(ind, obj, low, up, x, err, optimizer):
    if optimizer=='pso':
        pso = PSO(func=obj, dim=ft_num*2, pop=popp, max_iter=itrr, lb=low, ub=up)
        pso.record_mode = True
        pso.run()
        x[:, ind] = pso.gbest_x
        err[:, ind] = pso.gbest_y_hist
        print('best_y is', pso.gbest_y)
    if optimizer=='ga':
        ga = GA( func=obj, n_dim=ft_num*2, size_pop=popp, max_iter=itrr, lb=low, ub=up, precision=1e-7)
        best_x, best_y = ga.run()
        x[:, ind] = best_x
        # err[:, ind] = ga.gbest_y_hist
        print('best_y is', best_y)
    elif optimizer=='sa':
        sa = SA(func=obj, x0=np.ones((ft_num*2,)), T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
        best_x, best_y = sa.run()
        x[:, ind] = best_x
        err[:, ind] = sa.gbest_y_hist
        print('best_y is', best_y)

# def run_and_rcd(ind, obj, low, up, x, err):
#     pso = PSO(func=obj, dim=ft_num*2, pop=popp, max_iter=itrr, lb=low, ub=up)
#     pso.record_mode = True
#     pso.run()
#     x[:,ind]=pso.gbest_x
#     # print('=========')
#     # print(pso.gbest_x)
#     err[:,ind]=pso.gbest_y_hist
#     # print('best_y is', pso.gbest_y)

if args.name=='ctr' or args.name=='ctr_em':
    sta = feature_partition_ctr(df, ft_name)
else:
    sta = feature_partition(df, ft_name)

range_log = cat_process(df, cat_ft)

# optimal search boundary
# up_arr, low_arr = bound_fd(sta, args.label_str, ft_name)

up_arr=args.up*np.ones((ft_num*2,len(args.label_str)))
low_arr=args.lb*np.ones((ft_num*2,len(args.label_str)))

indd=np.ones((ft_num,M))
x_0_arr=np.ones((2*ft_num,M))
x_1_arr=np.ones((2*ft_num,M))
x_2_arr=np.ones((2*ft_num,M))
######################################
err_0_arr=np.ones((itrr,M))
err_1_arr=np.ones((itrr,M))
err_2_arr=np.ones((itrr,M))
######################################
now_time = datetime.datetime.now()
path=args.base_dir+'/'+str(now_time)

if ~os.path.isdir(path):
    os.mkdir(path)

score_log=np.zeros((ft_num,))

for i in range(M):
    run_and_rcd(i, obj_mut0, low_arr[:,0], up_arr[:,0], x_0_arr, err_0_arr, 'pso')
    run_and_rcd(i, obj_mut1, low_arr[:,1], up_arr[:,1], x_1_arr, err_1_arr,'pso')
    if (args.name=='wine') or (args.name=='iris'):
        run_and_rcd(i, obj_mut2, low_arr[:,2], up_arr[:,2], x_2_arr, err_2_arr)

    print('finish finding')
    ft_name=np.array(ft_name)
    cp10=compare(x_0_arr[:,i], x_1_arr[:,i], ft_num)
    score_arr = np.array(cp10)
    score_log+=score_arr
    ind10=np.argsort(-cp10)
    score_arr_sort = score_arr[ind10]
    # print('---- feature difference ranking cat 1 and 0 ----')
    print(ft_name[ind10])
    print('save the '+ str(i)+'-th result')
    np.save(path+'/x0_'+str(i)+'.npy', x_0_arr)
    np.save(path+'/x1_'+str(i)+'.npy', x_1_arr)
    np.save(path+'/err0.npy', err_0_arr)
    np.save(path+'/err1.npy', err_1_arr)
    np.save(path + '/score_'+str(i)+'.npy', score_arr_sort)

    
    if (args.name=='wine') or (args.name=='iris'):
        cp12=compare(x_1_arr[:,i], x_2_arr[:,i], ft_num)
        ind12=np.argsort(-cp12)
        cp02=compare(x_0_arr[:,i], x_2_arr[:,i], ft_num)
        ind02=np.argsort(-cp02)
        print('---- feature difference ranking cat 1 and 2 ----')
        print(ft_name[ind12])
        print('---- feature difference ranking cat 0 and 2 ----')
        print(ft_name[ind02])
        np.save(path+'/x2.npy', x_2_arr)
        np.save(path+'/err2.npy', err_2_arr)

ind=np.argsort(-1*score_log)
print(ft_name[ind])
np.save(path + '/score.npy', score_log)




# class objective():
#     def __init__(self, label):
#         self.label=label
#     def fun(self, x):
#         ttt = shuffle(df[df['label']==self.label])
#         xx, cat_ind=sample_pt(ttt, x, num, sel_num, cat_ft, num_ft, ft_name, range_log)
#         for i, name in enumerate(ft_name):
#             tmp = ttt[name][:sel_num]
#             xx[num-sel_num:, i]=tmp
#         if Name=='rank':
#             p = model.predict(xx.astype(np.float32))
#         if Name=='mkt':
#             X = torch.from_numpy(xx.astype(np.long))
#             p=model(X)[0][0][1].detach().numpy()
#         else:
#             p = model.predict_proba(xx.astype(np.float32))[1]

#         sta_=sta[str(self.label)]
#         kl_=0
#         for i, ft in enumerate(ft_name):
#             mu1 = sta_[ft][0]
#             sig1 = sta_[ft][1]
#             mu2 = x[2*i]
#             sig2 = x[2*i+1]
#             kl_+=kl(mu1, mu2, sig1, sig2)
#         return -np.mean(np.log(1e-8+p))+alpha*kl_

    
