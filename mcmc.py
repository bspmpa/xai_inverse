import numpy as np
import scipy.stats as st
import torch
import datetime
import argparse
from util import feature_partition, compare, cat_process, sample_pt_v2, feature_partition_ctr, DeepFM, bound_fd
from data_process import read_iris, read_rank, read_mkt, read_wine, read_gaussian, read_integer, read_ctr, read_ctr_em
import pickle
import random
import scipy.stats


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='wine', help='')
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



def target_1(Name, model, xx):
    if Name == 'rank':
        p = model.predict(xx.astype(np.float32))
    elif Name == 'mkt':
        X = torch.from_numpy(xx.astype(np.long))
        p = model(X)[0][0][1].detach().numpy()
    # elif Name == 'ctr':
    #     p = model(torch.LongTensor(xx[:, 13:]), torch.FloatTensor(xx[:, :13])).detach().numpy()
    # elif Name == 'ctr_em':
    #     p = model(torch.LongTensor(xx)).detach().numpy()
    else:
        p = model.predict_proba(xx.astype(np.float32))[:,1]

    return p.mean()

def target_0(Name, model, xx):
    if Name=='rank':
        p = 1-model.predict(xx.astype(np.float32))
    elif Name=='mkt':
        X=torch.from_numpy(xx.astype(np.long))
        p=model(X)[0][0][0].detach().numpy()
    # elif Name=='ctr':
    #     p=model(torch.LongTensor(xx[:,13:]), torch.FloatTensor(xx[:,:13])).detach().numpy()
    # elif Name=='ctr_em':
    #     p=model(torch.LongTensor(xx)).detach().numpy()
    else:
        p = model.predict_proba(xx.astype(np.float32))[:,0]

    return p.mean()


#
# def pgauss(x, y):
#     return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)


def metropolis_hastings(ft_num, pos=1, scale_ = 1, iter=1000, decay = 0.9):

    # samples = np.zeros((iter, ft_num))
    samples = []

    if pos==1:
        p_0 = 0.5
        x = np.zeros((1, ft_num))
        for i in range(iter):
            x_star= x + np.random.normal(loc=0, scale=scale_, size = (1, ft_num))
            # print(x_star)
            p = target_1(Name, model, x_star)
            if (random.uniform(0.8, 1) < p / p_0) or (p>0.6):
                x = x_star
                p_0 = p
                scale_ = scale_*decay
                samples.append(x)
    else:
        p_0 = 0.5
        x = np.zeros((1, ft_num))
        for i in range(iter):
            x_star= x + np.random.normal(loc=0, scale=scale_, size = (1, ft_num))
            p = target_0(Name, model, x_star)
            if (random.uniform(0.9, 1) > p / p_0) or (p<0.45):

                x = x_star
                p_0 = p
                scale_ = scale_ * decay
                samples.append(x)


    return np.array(samples).reshape((-1,ft_num))

if __name__ == '__main__':
    args = parser.parse_args()
    Name = args.name
    num = args.num
    alpha = args.alpha
    popp = args.popp
    itrr = args.itrr
    M = args.M

    if args.name == 'mkt':
        model = torch.load('/data/code_yang/paper/mkt/model_2.pkl')
        df, ft_name, cat_ft, num_ft, ft_num = read_mkt(args)
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

    if args.name == 'gaussian':
        with open(args.model_dir_gaussian, 'rb') as f:
            model = pickle.load(f)
            df, ft_name, cat_ft, num_ft, ft_num = read_gaussian(args)

    if args.name == 'integer':
        with open(args.model_dir_integer, 'rb') as f:
            model = pickle.load(f)
            df, ft_name, cat_ft, num_ft, ft_num = read_integer(args)

    sample_1 = metropolis_hastings(ft_num, pos=1, iter=10000)
    sample_0 = metropolis_hastings(ft_num, pos=0, iter=10000)

    ll = min(sample_1.shape[0], sample_1.shape[1])

    print(sample_1.shape)
    print(sample_0.shape)


    ft_score=[]
    for i in range(sample_0.shape[1]):
        # print(sample_1[:ll,i])
        # print(']]]]')
        # print(sample_0[:ll,i])
        # print('=====')

        ft_score.append(scipy.stats.entropy(sample_1[:ll,i], sample_0[:ll,i]))
    print(ft_score)
    print(ft_name)



