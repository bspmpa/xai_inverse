import numpy as np
from metric import symkl, js, hellinger
from numpy.linalg import inv,eig, det
import torch.nn as nn
import torch

def bound_onelabel(sta, cl, ft_name):
    up=[]
    low=[]
    for ft in ft_name:
        a0=np.sort(np.array((1.5*sta[cl][ft][0],0.5*sta[cl][ft][0])))
        a1 = np.sort(np.array((1.5 * sta[cl][ft][1], 0.5 * sta[cl][ft][1])))
        up.append(a0[1])
        up.append(a1[1])
        low.append(a0[0])
        low.append(a1[0])
    return up, low

def bound_fd(sta, label, ft_name):
    up_arr=np.zeros((2*len(ft_name),len(label)))
    low_arr=np.zeros((2*len(ft_name),len(label)))
    for i, label_ in enumerate(label):
        up_, low_=bound_onelabel(sta, label_, ft_name)
        # print('$$$$$$$$')
        #
        # print(sum(np.array(up_)<np.array(low_)))
        up_arr[:,i]=up_
        low_arr[:,i]=low_
    return up_arr, low_arr

def feature_partition_ctr(df, ft_name):
    ll=[0,1]
    ddd={}
    for i in range(len(ll)):
        df_ = df[df['label']==ll[i]]
        sta={}
        for name in ft_name:
            sta[name] = [df_[name].iloc[2], df_[name].iloc[3]]
        ddd[str(i)]=sta     
    return ddd

def feature_partition(df, ft_name):
    ll = list(set(df['label']))
    ddd={}
    for i in range(len(ll)):
        ttt = df[df['label']==ll[i]]
        sta={}
        for name in ft_name:
            mu=np.mean(ttt[name])
            sigma=np.std(ttt[name])
            sta[name]=[mu,sigma]
        ddd[str(i)]=sta
    return ddd

def likelihood(X, num, ft_num):
    xx=np.zeros((num,ft_num)) 
    for i in range(ft_num):
        xx[:,i]=np.random.normal(loc=X[2*i],scale=X[2*i+1],size=num)
    y = model.predict_proba(xx)
    return y

# def kl(mu1, mu2, sig1, sig2): # 1/2
#     if sig1!=0 and sig2!=0:
#         a=-np.log(sig2/sig1+1e-8)-1+sig1/(1e-8+sig2)+(mu2-mu1)**2/(1e-8+sig2)
#         b=-np.log(sig1/sig2+1e-8)-1+sig2/(1e-8+sig1)+(mu1-mu2)**2/(1e-8+sig1)
#         return 0.5*(a+b)
#     else:
#         return 0feature_partition
    
def compare(x_1, x_2, ft_num):
    div=[]
    for i in range(ft_num):
        mu1=x_1[i*2]
        sig1=x_1[i*2+1]
        mu2=x_2[i*2]
        sig2=x_2[i*2+1]
        div.append(symkl(mu1, sig1, mu2, sig2))
    return np.array(div)
    
def cat_process(df, cat_ft):
    if len(cat_ft):
        range_log={}
        for i in range(len(cat_ft)):
            X=df[cat_ft[i]]
            a=X.min()
            b=X.max()
            range_log[cat_ft[i]]=[a,b]
        return range_log   

def sample_pt_v2(df, x, num, cat_ft, num_ft, ft_name, range_log):
    xx = np.zeros((num, len(ft_name)))
    # categorical index
    if len(cat_ft):
        cat_ind = np.array([np.where(np.array(ft_name) == cat_ft[i])[0] for i in range(len(cat_ft))])
    else:
        cat_ind = []
    # numerical index
    num_ind = [np.where(np.array(ft_name) == num_ft[i])[0] for i in range(len(num_ft))]
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
            # print(low)
            xx[:, kk] = ftt

    if len(num_ft):
        for i in range(len(num_ft)):
            pp = int(num_ind[i])
            ft_ = ft_name[pp]
            low = df[ft_].min()
            up = df[ft_].max()
            ftt = np.random.normal(loc=x[2 * pp], scale=x[2 * pp + 1], size=num)
            ftt[ftt > up] = up
            ftt[ftt < low] = low
            xx[:, pp] = ftt
    return xx, cat_ind


def add_corr(cor, ind_x, ft_num):
    cor_mat = np.array(cor[0:ft_num * ft_num]).reshape((ft_num, ft_num))
    return (np.matmul(cor_mat, ind_x.T) + np.array(cor[-ft_num:]).reshape((-1, 1))).T

def kl_multi(mu1, mu2, sig1, sig2):
    def met(mu1, mu2, sig1, sig2):
        return np.log(1e-4 + det(sig1)) - np.log(1e-4 + det(sig2)) - 2 \
               + np.trace(np.matmul(inv(sig2), sig1)) + np.matmul(np.matmul((mu1 - mu2).T, inv(sig2)), (mu1 - mu2))
    return 0.25 * (met(mu1, mu2, sig1, sig2) + met(mu2, mu1, sig2, sig1))

def compare_multi(x_1, x_2, ft_num):
    kl_div = []
    kl_ind = []
    for i in range(ft_num):
        for j in np.arange(i + 1, ft_num):
            mu1 = np.ones((2, 1))
            mu2 = np.ones((2, 1))
            sig1 = np.ones((2, 2))
            sig2 = np.ones((2, 2))
            mu1[0] = x_1[-ft_num * ft_num + i]
            mu1[1] = x_1[-ft_num * ft_num + j]
            mu2[0] = x_2[-ft_num * ft_num + i]
            mu2[1] = x_2[-ft_num * ft_num + j]
            sig1[0, 0] = x_1[i * ft_num]
            sig1[1, 1] = x_1[j * ft_num]
            sig1[0, 1] = x_1[j * ft_num + i]
            sig1[1, 0] = x_1[j * ft_num + i]
            sig2[0, 0] = x_2[i * ft_num]
            sig2[1, 1] = x_2[j * ft_num]
            sig2[0, 1] = x_2[j * ft_num + i]
            sig2[1, 0] = x_2[j * ft_num + i]
            sig11 = np.matmul(sig1.T, sig1)
            sig22 = np.matmul(sig2.T, sig2)
            if det(sig11) > 0 and det(sig22) > 0:
                # print('00000000')
                #                 print(kl_multi(mu1, mu2, sig11, sig22))
                # print(js(mu1, sig11, mu2, sig22))
                # kl_div.append(js(mu1, sig11, mu2, sig22))
                # print('====')
                # print(sig11)
                # print(sig22)
                # print('=====')
                kl_div.append(kl_multi(mu1, mu2, sig11, sig22))
                kl_ind.append([i, j])
            else:
                kl_div.append(0)
                kl_ind.append([i, j])


    return np.array(kl_div), kl_ind

def compare_posterior_dist(x0, x1, n1, save_dir, ft_num):
    x_0=np.mean(x0, axis=1)
    x_1=np.mean(x1, axis=1)
    kl, kl_ind=compare_multi(x_0, x_1, ft_num)
    np.save(save_dir+'/compare_kl_'+n1+'.npy', kl)
    np.save(save_dir+'/kl_ind_'+n1+'.npy', kl_ind)
    return kl, kl_ind

# class DeepFM(nn.Module): # 连续
#     def __init__(self, cate_fea_nuniqs,
#                  nume_fea_size=1, emb_size=16, hid_dims=[64], num_classes=1, dropout=[0.5]):
#         """
#         cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
#         nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况
#         """
#         super().__init__()
#         self.cate_fea_size = len(cate_fea_nuniqs)
#         self.nume_fea_size = nume_fea_size
#
#         """FM部分"""
#         # 一阶
#         if self.nume_fea_size != 0:
#             self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示
#         self.fm_1st_order_sparse_emb = nn.ModuleList([
#             nn.Embedding(voc_size, 1) for voc_size in cate_fea_nuniqs])  # 类别特征的一阶表示
#
#         # 二阶
#         self.fm_2nd_order_sparse_emb = nn.ModuleList([
#             nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs])  # 类别特征的二阶表示
#
#         """DNN部分"""
#         self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
#         self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
#         self.relu = nn.ReLU()
#         # for DNN
#         for i in range(1, len(self.all_dims)):
#             setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
#             setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
#             setattr(self, 'activation_' + str(i), nn.ReLU())
#             setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))
#         # for output
#         self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, X_sparse, X_dense=None):
#         """
#         X_sparse: 类别型特征输入  [bs, cate_fea_size]
#         X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
#         """
#
#         """FM 一阶部分"""
#         fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1)
#                              for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
#         fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
#         fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]
#
#         if X_dense is not None:
#             fm_1st_dense_res = self.fm_1st_order_dense(X_dense)
#             fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
#         else:
#             fm_1st_part = fm_1st_sparse_res  # [bs, 1]
#
#         """FM 二阶部分"""
#         fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
#         fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)
#
#         # 先求和再平方
#         sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
#         square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
#         # 先平方再求和
#         square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
#         sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
#         # 相减除以2
#         sub = square_sum_embed - sum_square_embed
#         sub = sub * 0.5  # [bs, emb_size]
#
#         fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # [bs, 1]
#
#         """DNN部分"""
#         dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, n * emb_size]
#
#         if X_dense is not None:
#             dense_out = self.relu(self.dense_linear(X_dense))  # [bs, n * emb_size]
#             dnn_out = dnn_out + dense_out  # [bs, n * emb_size]
#
#         for i in range(1, len(self.all_dims)):
#             dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
#             dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
#             dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
#             dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
#
#         dnn_out = self.dnn_linear(dnn_out)  # [bs, 1]
#         out = fm_1st_part + fm_2nd_part + dnn_out  # [bs, 1]
#         out = self.sigmoid(out)
#         return out

class DeepFM(nn.Module):
    def __init__(self, cate_fea_nuniqs,
                 nume_fea_size=0, emb_size=16, hid_dims=[128, 64], num_classes=1, dropout=[0.5, 0.5]):
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size

        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_nuniqs])  # 类别特征的一阶表示

        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs])  # 类别特征的二阶表示

        """DNN部分"""
        self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
        if self.nume_fea_size != 0:
            self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))
        # for output
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """

        """FM 一阶部分"""
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res  # [bs, 1]

        """FM 二阶部分"""
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, emb_size]

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # [bs, 1]

        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, n * emb_size]

        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))  # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out  # [bs, n * emb_size]

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.dnn_linear(dnn_out)  # [bs, 1]
        out = fm_1st_part + fm_2nd_part + dnn_out  # [bs, 1]
        out = self.sigmoid(out)
        return out


