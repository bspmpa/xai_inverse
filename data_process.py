import pickle 
import time
import os
import random
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def padding(v):
    if len(v.split(','))>=5:
        return v.split(',')[:5]
    pad=0
    pad=5-len(v.split(','))
    pad_list=[1]*pad
    return v.split(',')+pad_list


def index(v):
    if len(v.split(','))>=5:
        if v.split(',')[0]=='1':
            return 0
        return 5
    else:
        return len(v.split(','))
    
def preprocess1(dataframe):
    df1=dataframe#.iloc[:1000]
    df1['detail_position'].fillna('1'+',1'*4,inplace=True)
    df1['addf_position'].fillna('1'+',1'*4,inplace=True)
    detail=df1['detail_position']
    addf=df1['addf_position']

    x=np.array(list(map(padding,detail)))
    df1=pd.merge(df1,pd.DataFrame(x,columns=['detail_position0',
                                             'detail_position1',
                                             'detail_position2',
                                             'detail_position3',
                                             'detail_position4',
                                            ]),left_index=True,right_index=True)

    x=np.array(list(map(padding,addf)))
    df1=pd.merge(df1,pd.DataFrame(x,columns=['addf_position0',
                                             'addf_position1',
                                             'addf_position2',
                                             'addf_position3',
                                             'addf_position4'
                                            ]),left_index=True,right_index=True)
    x=np.array(list(map(index,detail)))
    df1=pd.merge(df1,pd.DataFrame(x,columns=['detail_index']),left_index=True,right_index=True)
    x=np.array(list(map(index,addf)))
    df1=pd.merge(df1,pd.DataFrame(x,columns=['addf_index']),left_index=True,right_index=True)

    df3 = df1['label'].to_numpy().astype(np.float32)
    df2=df1[ft_name]
    df2=df2.to_numpy().astype(np.float32)

    return df2,df3

def preprocess_rank(dataframe, ft_name):
    df1=dataframe #.iloc[:1000]               
    df2=df1[ft_name].to_numpy().astype(np.float32)
    df3 = df1['label'].to_numpy().astype(np.float32)
    return df2,df3

def function(n):
    if n==0:
        return 1
    else:
        return 0
    
def read_mkt(args):
    data = pd.read_csv(args.data_dir_mkt, sep=';', na_values=r'\N')
    y0 = np.array(list(map(function, data['y'])))
    data = pd.merge(data, pd.DataFrame(y0, columns=['y0']), left_index=True, right_index=True)
    y={'yes': 1, 'no': 0}
    job = {'unknown': 0, 'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5,
           'retired': 6, 'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10, 'unemployed': 11}
    marital = {'married': 1, 'single': 2, 'divorced': 3}
    education = {'tertiary': 1, 'secondary': 2, 'unknown': 0, 'primary': 3}
    default = {'yes': 1, 'no': 0}
    housing = {'yes': 1, 'no': 0}
    loan = {'yes': 1, 'no': 0}
    contact = {'unknown': 0, 'cellular': 1, 'telephone': 2}
    month = {'may': 1, 'jun': 2, 'jul': 3, 'aug': 4, 'oct': 5, 'nov': 6, 'dec': 7, 'jan': 8, 'feb': 9,
             'mar': 10, 'apr': 11, 'sep': 12}
    poutcome = {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3}
    L = {'job': job, 'marital': marital, 'education': education, 'default': default, 'housing': housing, 'loan': loan,
         'contact': contact, 'month': month, 'poutcome': poutcome, 'y':y}
    for i in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome','y']:
        data[i] = data[i].replace(L[i])

    #y = data['y'].values
    X = data[['job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'month', 'poutcome','age', 'balance', 'day', 'duration', 'campaign', 'pdays',
       'previous']].values
    ft_name=['job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'month', 'poutcome','age', 'balance', 'day', 'duration', 'campaign', 'pdays',
       'previous']
    cat_ft=['job', 'marital', 'education', 'default', 'housing','loan', 'contact', 'month', 'poutcome']
    num_ft=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    df=pd.DataFrame(X, columns=ft_name)
    df['label']=data['y'].values

    return df, ft_name, cat_ft, num_ft, len(ft_name)


def read_iris():
    iris = load_iris()
    ft_name=iris['feature_names']
    df_=pd.DataFrame(iris['data'],columns=ft_name)
    df_['label']=iris['target']
    return df_, ft_name, [], ft_name, len(ft_name)

def read_wine(args):
    data = np.loadtxt (args.data_dir_wine, delimiter=',') 
    ft_name= np.array(['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 
          'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
          'OD280/OD315 of diluted wines', 'Proline'])
    y=data[:,0]-1
    df_=pd.DataFrame(data[:,1:],columns=ft_name)
    df_['label']=y
    return df_, ft_name, [], ft_name, len(ft_name)

def read_rank(args):
    columns=['label','l_rank','geek_position','geek_city','geek_workyears','geek_degree','geek_major','el','eh',\
         'geek_paddf_rate_7d','geek_success_times_7d',\
         'boss_position','boss_city','job_workyears','job_degree','boss_title_type','jl','jh','boss_min_chat_tdiff',\
         'job_min_active_tdiff','job_paddf_rate_7d','job_success_times_7d','boss_paddf_success_times_2d',\
         'boss_paddf_success_rate_2d',\
         'boss_paddf_pchat_rate_2d','boss_paddf_rate_2d','job_pas_addf_num_24h','job_paddf_success_times_2d',\
         'job_paddf_success_times_7d','job_paddf_rate_14d','job_success_times_2d','job_psuccess_times_7d',\
         'job_paddfchat_times_7d','g2b_w2v_int_orig_gof','g2b_w2v_int_pref_gof','g2b_title_type_gof','job_type','brow',\
         'detail_position','addf_position']

# ft_name=['l_rank','geek_position','geek_city','geek_workyears','geek_degree','geek_major','el','eh','geek_paddf_rate_7d',\
# 'geek_success_times_7d','boss_position','boss_city','job_workyears','job_degree','boss_title_type','jl','jh',\
# 'boss_min_chat_tdiff','job_min_active_tdiff','job_paddf_rate_7d','job_success_times_7d','boss_paddf_success_times_2d',\
# 'boss_paddf_success_rate_2d','boss_paddf_pchat_rate_2d','boss_paddf_rate_2d','job_pas_addf_num_24h',\
# 'job_paddf_success_times_2d','job_paddf_success_times_7d','job_paddf_rate_14d','job_success_times_2d',\
# 'job_psuccess_times_7d','job_paddfchat_times_7d', 'g2b_w2v_int_orig_gof','g2b_w2v_int_pref_gof','brow','detail_position0',\
# 'detail_position1','detail_position2','detail_position3','detail_position4','addf_position0','addf_position1',\
# 'addf_position2','addf_position3','addf_position4','detail_index','addf_index']

# ft_name=['l_rank','geek_position','geek_city','geek_workyears','geek_degree','geek_major',\
#          'el','eh','boss_position','boss_city','job_workyears','job_degree','boss_title_type',\
#          'jl','jh','boss_min_chat_tdiff','boss_paddf_success_times_2d','boss_paddf_success_rate_2d',\
#          'job_paddf_success_times_2d','job_paddf_success_times_7d', \
#          'g2b_w2v_int_orig_gof','g2b_w2v_int_pref_gof']

    ft_name=['l_rank','el','eh','job_workyears','jl','jh', 'g2b_w2v_int_orig_gof','g2b_w2v_int_pref_gof',                                      'boss_min_chat_tdiff','job_min_active_tdiff', 
             'job_paddf_rate_7d','job_success_times_7d',\
             'boss_paddf_success_times_2d','boss_paddf_success_rate_2d','boss_paddf_pchat_rate_2d',\
             'boss_paddf_rate_2d','job_pas_addf_num_24h','job_paddf_success_times_2d','job_paddf_success_times_7d',\
             'job_paddf_rate_14d','job_success_times_2d','job_psuccess_times_7d','job_paddfchat_times_7d']

    cat_ft=['l_rank','el','eh','job_workyears','jl','jh']

#cat_ft = ['geek_position','geek_city','geek_workyears','geek_degree','geek_major','geek_major', 
 #         'boss_position','boss_city','job_workyears','job_degree','boss_title_type']
    num_ft=['g2b_w2v_int_orig_gof','g2b_w2v_int_pref_gof','boss_min_chat_tdiff','job_min_active_tdiff',\
            'job_paddf_rate_7d','job_success_times_7d','boss_paddf_success_times_2d','boss_paddf_success_rate_2d',\
            'boss_paddf_pchat_rate_2d','boss_paddf_rate_2d','job_pas_addf_num_24h','job_paddf_success_times_2d',\
            'job_paddf_success_times_7d','job_paddf_rate_14d','job_success_times_2d','job_psuccess_times_7d',\
            'job_paddfchat_times_7d']
    df_test=pd.read_csv(args.data_dir_rank, sep='\001',na_values=r'\N',header=None)
    df_test.columns=columns
    df_test=df_test[df_test['l_rank']!=-1]
    df_test=df_test[~df_test.isnull()]
    df2,df3=preprocess_rank(df_test, ft_name) 
    df=pd.DataFrame(df2, columns=ft_name)
    kk=df3.astype(np.float32)
    df['label']=kk
    return df, ft_name, cat_ft, num_ft, len(ft_name)


def read_gaussian(args):
    data = pd.read_csv(args.data_dir_gaussian)
    ft_name= np.array(['f1', 'f2'])
    df_=data.rename(columns={"labels": "label"})
    return df_, ft_name, [], ft_name, len(ft_name)

def read_gaussian(args):
    data = pd.read_csv(args.data_dir_gaussian)
    ft_name= np.array(['f1', 'f2'])
    df_=data.rename(columns={"labels": "label"})
    return df_, ft_name, [], ft_name, len(ft_name)

def read_integer(args):
    data = pd.read_csv(args.data_dir_integer)
    ft_name= np.array(['f1', 'f2'])
    df_=data.rename(columns={"labels": "label"})
    return df_, ft_name, [], ft_name, len(ft_name)

def read_ctr(args):
    data = pd.read_csv(args.data_dir_ctr)
    ft_name = list(data.columns[2:])
    num_features = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
    cat_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
                    'C16',
                    'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    return data, ft_name, cat_features, num_features, len(ft_name)

def read_ctr_em(args):
    data = pd.read_csv(args.data_dir_ctr_em)
    ft_name = list(data.columns[2:])
    return data, ft_name, ft_name, [], len(ft_name)







