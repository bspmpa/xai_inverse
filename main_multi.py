import pickle
import os
import numpy as np
from sko.PSO import PSO
from sko.GA import GA
from sko.SA import SA
import datetime
import argparse
from util import feature_partition, cat_process, bound_fd, add_corr, compare_posterior_dist, compare_multi, DeepFM, feature_partition_ctr
from data_process import read_iris, read_rank, read_mkt, read_wine, read_ctr_em, read_ctr
import torch
from metric import happening_prob, prob_compare, prob_compare_1, symkl, hellinger, js
from train import model_train
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='ctr_em', help='')
parser.add_argument('--base_dir', type=str, default='/data/code_yang/paper/ctr_em/res_multi', help='')
parser.add_argument('--model_metric', type=str, default='acc', help='')
parser.add_argument('--num', type=int, default=300, help='')
parser.add_argument('--optimizer', type=str, default='pso', help='')
parser.add_argument('--alpha', type=float, default=1e-6, help='')
parser.add_argument('--popp', type=int, default=150, help='')
parser.add_argument('--itrr', type=int, default=40, help='')
parser.add_argument('--M', type=int, default = 3, help='')
parser.add_argument('--lb', type=float, default=1e-1, help='')
parser.add_argument('--up', type=float, default=100, help='')
parser.add_argument('--label_str', type=str, default=['0', '1'], help='')

parser.add_argument('--data_dir_rank',  type=str,   default='/data/code_yang/paper/rank/model_1/position2', help='')
parser.add_argument('--data_dir_mkt',   type=str,   default='/data/code_yang/paper/mkt/bank-full.csv', help='')
parser.add_argument('--data_dir_wine',  type=str,   default="/data/code_yang/paper/wine/wine.data", help='')
parser.add_argument('--data_dir_ctr', type=str, default="/data/code_yang/paper/ctr/tj_criteo.csv", help='')
parser.add_argument('--data_dir_ctr_em', type=str, default="/data/code_yang/paper/ctr/tj_criteo_cat.csv", help='')

parser.add_argument('--model_dir_mkt',  type=str,   default='/data/code_yang/paper/mkt/model_2.pkl', help='')
parser.add_argument('--model_dir_iris', type=str,   default='/data/code_yang/paper/iris/iris.pkl', help='')
parser.add_argument('--model_dir_wine', type=str,   default='/data/code_yang/paper/wine/wine.pkl', help='')
parser.add_argument('--model_dir_rank', type=str,   default='/data/code_yang/paper/rank/model_1/delete_feature.pkl', help=' ')
parser.add_argument('--model_dir_ctr', type=str, default='/data/code_yang/paper/ctr/criteo780770.pth', help='')
parser.add_argument('--model_dir_ctr_em', type=str, default='/data/code_yang/paper/ctr/Cat_7857.pth', help='')

args = parser.parse_args()

global num, alpha, ft_name, range_log, cat_ft, num_ft, onnx_path, columns, model, NAME, sta

Name = args.name
num = args.num
alpha = args.alpha
popp = args.popp
itrr = args.itrr
M = args.M


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

#  eval
def obj_mut1(x):  ### mutate ###
    if Name=='ctr' or Name=='ctr_em':
        range_log = {}
        for i in range(len(cat_ft)):
            X = df[cat_ft[i]].iloc[[0,1,4,5]]
            a = X.min()
            b = X.max()
            range_log[cat_ft[i]] = [a, b]
    xx= np.random.normal(loc=0, scale=1, size=[num,ft_num])
    xx_new=add_corr(x, xx, ft_num)
    if len(cat_ft):
        cat_ind = np.array([np.where(np.array(ft_name) == cat_ft[i])[0] for i in range(len(cat_ft))])
    else:
        cat_ind = []
    if len(cat_ft):
        for i in range(len(cat_ft)):
            kk = int(cat_ind[i])
            ft_ = ft_name[kk]
            low = range_log[ft_][0]
            up = range_log[ft_][1]
            tmp = np.random.normal(loc=x[2 * kk], scale=x[2 * kk + 1], size=num)
            ftt = np.round(tmp)
            ftt[ftt > up] = up-1
            ftt[ftt < low] = low+1
            xx_new[:, kk] = ftt
    # xx, cat_ind = sample_pt_v2(df[df['label'] == 1], x, num, cat_ft, num_ft, ft_name, range_log)
    if Name == 'rank':
        p = model.predict(xx_new.astype(np.float32))
    elif Name == 'mkt':
        dim_list = np.array([13, 4, 5, 3, 3, 3, 4, 13, 5])-1
        for i in range(len(dim_list)):
            xx_new[:, i] = np.round(np.abs(xx_new[:, i]))
            xx_new[xx_new[:, i] > dim_list[i], i] = dim_list[i]
        X = torch.from_numpy(xx_new.astype(np.long))
        p = model(X)[0][0][1].detach().numpy()
    elif Name == 'ctr':
        p = model(torch.LongTensor(xx_new[:, 13:]), torch.FloatTensor(xx_new[:, :13])).detach().numpy()
    elif Name == 'ctr_em':
        p = model(torch.LongTensor(xx_new)).detach().numpy()
    else:
        p = model.predict_proba(xx_new.astype(np.float32))[1]
    sta_ = sta['1']
    div_ = 0
    for i, ft in enumerate(ft_name):
        mu1 = sta_[ft][0]
        sig1 = sta_[ft][1]
        cor_mat = np.array(x[0:-ft_num]).reshape((ft_num, ft_num))
        cor_mat_ = np.matmul(cor_mat.T, cor_mat)
        mu2 = x[-ft_num * ft_num + i]
        sig2 = np.sqrt(cor_mat_[i,i])
        div_ += symkl(mu1, sig1, mu2, sig2)
        # div_ += kl(mu1, sig1, mu2, sig2)
    return -np.mean(np.log(1e-8 + p)) + (alpha / ft_num) * div_

def obj_mut0(x):  ### mutate ###
    if Name=='ctr' or Name=='ctr_em':
        range_log = {}
        for i in range(len(cat_ft)):
            X = df[cat_ft[i]].iloc[[0,1,4,5]]
            a = X.min()
            b = X.max()
            range_log[cat_ft[i]] = [a, b]
    xx= np.random.normal(loc=0, scale=1, size=[num,ft_num])
    xx_new=add_corr(x, xx, ft_num)
    if len(cat_ft):
        cat_ind = np.array([np.where(np.array(ft_name) == cat_ft[i])[0] for i in range(len(cat_ft))])
    else:
        cat_ind = []
    if len(cat_ft):
        for i in range(len(cat_ft)):
            kk = int(cat_ind[i])
            ft_ = ft_name[kk]
            low = range_log[ft_][0]
            up = range_log[ft_][1]
            tmp = np.random.normal(loc=x[2 * kk], scale=x[2 * kk + 1], size=num)
            ftt = np.round(tmp)
            ftt[ftt > up] = up
            ftt[ftt < low] = low
            xx_new[:, kk] = ftt

    if Name == 'rank':
        p = 1 - model.predict(xx_new.astype(np.float32))
    elif Name == 'mkt':
        dim_list = np.array([13, 4, 5, 3, 3, 3, 4, 13, 5])-1
        for i in range(len(dim_list)):
            xx_new[:,i] = np.round(np.abs(xx_new[:,i]))
            xx_new[xx_new[:, i] > dim_list[i], i] = dim_list[i]
        X = torch.from_numpy(xx_new.astype(np.long))
        p = model(X)[0][0][0].detach().numpy()
    elif Name == 'ctr':
        p = model(torch.LongTensor(xx_new[:, 13:]), torch.FloatTensor(xx_new[:, :13])).detach().numpy()
    elif Name == 'ctr_em':
        p = model(torch.LongTensor(xx_new)).detach().numpy()
    else:
        p = model.predict_proba(xx_new.astype(np.float32))[0]
    sta_ = sta['0']
    div_ = 0
    for i, ft in enumerate(ft_name):
        mu1 = sta_[ft][0]
        sig1 = sta_[ft][1]

        cor_mat = np.array(x[0:-ft_num]).reshape((ft_num, ft_num))
        cor_mat_ = np.matmul(cor_mat.T, cor_mat)
        mu2 = x[-ft_num * ft_num + i]
        sig2 = np.sqrt(cor_mat_[i, i])
        div_ += symkl(mu1, sig1, mu2, sig2)
    # print('&&&&&&')
    # print(div_)
    # print(-np.mean(np.log(1e-8 + p)))
    return -np.mean(np.log(1e-8 + p)) + (alpha / ft_num) * div_


def obj_mut2(x):  # only for iris, wine
    # xx, cat_ind = sample_pt(ttt, x, num, sel_num, cat_ft, num_ft, ft_name, range_log, sta['2'])
    xx = np.random.normal(loc=0, scale=1, size=[num, len(ft_name)])
    xx_new = add_corr(x, xx, ft_num)
    p = model.predict_proba(xx_new.astype(np.float32))[2]
    sta_ = sta['2']
    div_ = 0
    for i, ft in enumerate(ft_name):
        mu1 = sta_[ft][0]
        sig1 = sta_[ft][1]
        cor_mat = np.array(x[0:-ft_num]).reshape((ft_num, ft_num))
        cor_mat_ = np.matmul(cor_mat.T, cor_mat)
        mu2 = x[-ft_num * ft_num + i]
        sig2 = np.sqrt(cor_mat_[i, i])
        div_ += symkl(mu1, sig1, mu2, sig2)
        # mu2 = x[-ft_num * ft_num + i]
        # sig2 = x[i * ft_num]
        # div_ += symkl(mu1, sig1, mu2, sig2)
        # div_ += kl(mu1, mu2, sig1, sig2)
    return -np.mean(np.log(1e-8 + p)) + (alpha / ft_num) * div_


# onnx_ = '/data/code_yang/paper/rank/model_1/delete_feature.pkl'
# f=open(onnx_path, 'rb')
# model=pickle.load(f)
if args.name == 'mkt':
    model = torch.load('/data/code_yang/paper/mkt/model_2.pkl')
    df, ft_name, cat_ft, num_ft, ft_num = read_mkt(args)
    # df, ft_name, cat_ft, num_ft, ft_num= process_data(args)
if args.name == 'wine':
    with open(args.model_dir_wine, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num = read_wine(args)
if args.name == 'iris':
    with open(args.model_dir_iris, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num = read_iris()
if args.name == 'rank':
    with open(args.model_dir_rank, 'rb') as f:
        model = pickle.load(f)
        df, ft_name, cat_ft, num_ft, ft_num = read_rank(args)
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

if args.name=='ctr' or args.name=='ctr_em':
    sta = feature_partition_ctr(df, ft_name)
else:
    sta = feature_partition(df, ft_name)

range_log = cat_process(df, cat_ft)

# optimal search boundary
up_arr, low_arr = bound_fd(sta, args.label_str, ft_name)

para_num=ft_num*(ft_num+1)

# up_arr = np.tile(df[ft_name].max(),ft_num+1)
# low_arr = np.tile(df[ft_name].min(),ft_num+1)

# scaler = MinMaxScaler()
# df[ft_name] = scaler.fit_transform(df[ft_name].values)

up_arr =  args.up * np.ones((para_num,))
low_arr = args.lb * np.ones((para_num,))

##########################################
############ Initialization ##############
##########################################
# print('=========')
# print(df[ft_name].min())
# print(df[ft_name].max())
# print('=========')

indd = np.ones((ft_num, M))
x_0_arr = np.ones((para_num, M))
x_1_arr = np.ones((para_num, M))
x_2_arr = np.ones((para_num, M))
######################################
err_0_arr = np.ones((itrr, M))
err_1_arr = np.ones((itrr, M))
err_2_arr = np.ones((itrr, M))

######################################
now_time = datetime.datetime.now()
path = args.base_dir + '/' + str(now_time)
if ~os.path.isdir(path):
    os.mkdir(path)


def run_and_rcd(ind, obj, low, up, x, err, para_num, optimizer):
    if optimizer=='pso':
        pso = PSO(func=obj, dim=para_num, pop=popp, max_iter=itrr, lb=low, ub=up)
        pso.record_mode = True
        pso.run()
        x[:, ind] = pso.gbest_x
        # err[:, ind] = pso.gbest_y_hist
        print('best_y is', pso.gbest_y)
    if optimizer=='ga':
        ga = GA( func=obj, n_dim=para_num, size_pop=popp, max_iter=itrr, lb=low, ub=up, precision=1e-7)
        best_x, best_y = ga.run()
        x[:, ind] = best_x
        # err[:, ind] = ga.gbest_y_hist
        print('best_y is', best_y)
    elif optimizer=='sa':
        sa = SA(func=obj, x0=np.ones((para_num,)), T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
        best_x, best_y = sa.run()
        x[:, ind] = best_x
        # err[:, ind] = ga.gbest_y_hist
        print('best_y is', best_y)

for i in range(M):
    run_and_rcd(i, obj_mut0, low_arr, up_arr, x_0_arr, err_0_arr, para_num, args.optimizer)
    run_and_rcd(i, obj_mut1, low_arr, up_arr, x_1_arr, err_1_arr, para_num, args.optimizer)
    if (args.name == 'wine') or (args.name == 'iris'):
        run_and_rcd(i, obj_mut2, low_arr, up_arr, x_2_arr, err_2_arr, para_num, args.optimizer)

np.save(path + '/x0.npy', x_0_arr)
np.save(path + '/x1.npy', x_1_arr)
np.save(path + '/err0.npy', err_0_arr)
np.save(path + '/err1.npy', err_1_arr)
if (args.name == 'wine') or (args.name == 'iris'):
    np.save(path + '/x2.npy', x_2_arr)
    np.save(path + '/err2.npy', err_2_arr)
print('finish finding')


########################

def tmp(x0, x1, ft_num):
    num_comb = int(ft_num * (ft_num - 1) / 2)
    kl=np.zeros((num_comb, x0.shape[1]))
    for i in range(x0.shape[1]):
        x_0_=x0[:,i]
        x_1_=x1[:,i]
        kl_, kl_ind = compare_multi(x_0_, x_1_, ft_num)
        xxx=kl_.reshape((num_comb,))
        kl[:,i]=normalise(xxx)
    return kl

def normalise(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def ft_index(start_ind, end_ind, step):
    a=[np.arange(start_ind, end_ind, step).astype(np.int32)]
    return list(combinations(a, 2))

start_ind=0
step= 1
end_ind=ft_num*(ft_num-1)/2
ft_index_dict = ft_index(start_ind, end_ind, step)
kl1=tmp(x_0_arr, x_1_arr, ft_num, para_num)
np.save(path + '/kl_score.npy', kl1)
kl1_arr = np.mean(kl1, axis=1)
ind=np.argsort(-kl1_arr)
print('++++++++++++++++++++')
print(ft_index_dict[ind])

# stt = np.std(kl1,axis=1)
# print('=======')
# print(para_num)
# print(stt.shape)
# print(feature_name)
# print(stt[feature_name])


#######################
# kl_arr_01, kl_ind_arr_01= compare_posterior_dist(x_0_arr, x_1_arr, '01', path, ft_num)
# if (args.name == 'wine') or (args.name == 'iris'):
#     kl_arr_12, kl_ind_arr_12 = compare_posterior_dist(x_1_arr, x_2_arr, '12', path, ft_num)
#     kl_arr_02, kl_ind_arr_02 = compare_posterior_dist(x_0_arr, x_2_arr, '02', path, ft_num)
#     kl_ind_arr=np.array(list(itertools.combinations(range(0,ft_num), 2)))
#     kl_arr=np.array(kl_arr_01)+np.array(kl_arr_02)+np.array(kl_arr_12)
# else:
#     kl_arr, kl_ind_arr = kl_arr_01, kl_ind_arr_01
#
# met, ft_col, index_map, ft_names = sub_model_metric(args, ttsize=0.3, rm=42, md_choice='lr', ev=args.model_metric, com_num=2)
# metric_log=mk_dict(met, ft_col)
# metric_df = pd.DataFrame(metric_log, index=[0])
# # kl_arr = np.load(args.kl_dir)
# # kl_ind_arr = np.load(args.kl_ind_dir)
# kl_col=[index_map[tuple(kl_ind_arr[i])] for i in range(len(kl_ind_arr))]
# kl_ss = np.zeros((metric_df.shape[1],))
# kl_ss[kl_col] = kl_arr.reshape((-1,))
# metric_df.loc[metric_df.shape[0]] = normalise(kl_ss)
# metric_df.loc[metric_df.shape[0]] = ft_names
# metric_df = metric_df.sort_values(by=0, axis=1, ascending=False)
# metric_df.to_csv(path + '/' + 'sub_metric_score.csv')
# print('finish saving')





    # ft_name = np.array(ft_name)
    # cp10 = compare_multi(x_0_arr[:, i], x_1_arr[:, i], ft_num)
    # ind10 = np.argsort(-cp10)
    # print('---- feature difference ranking cat 1 and 0 ----')
    # print(ft_name[ind10])
    # print('save the ' + str(i) + '-th result')


# if (args.name == 'wine') or (args.name == 'iris'):
#     cp12 = compare(x_1_arr[:, i], x_2_arr[:, i], ft_num)
#     ind12 = np.argsort(-cp12)
#     cp02 = compare(x_0_arr[:, i], x_2_arr[:, i], ft_num)
#     ind02 = np.argsort(-cp02)
#     print('---- feature difference ranking cat 1 and 2 ----')
#     print(ft_name[ind12])
#     print('---- feature difference ranking cat 0 and 2 ----')
#     print(ft_name[ind02])
#     np.save(path + '/x2.npy', x_2_arr)
#     np.save(path + '/err2.npy', err_2_arr)






