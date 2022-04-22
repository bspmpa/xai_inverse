import numpy as np
from scipy import stats
from math import e, pi
from numpy.linalg import inv,eig, det

def happening_prob(x, sta):
    ll=[(x[i]-sta[2*i])/(1e-8+sta[2*i+1]) for i in range(len(x))]
    y = stats.norm.cdf(ll)
    return y

def kl(mu1, sig1, mu2, sig2):
    return np.log(sig2/sig1+1e-8)+(sig1**2+(mu2-mu1)**2)/(1e-8+sig2**2)

def symkl(mu1, sig1, mu2, sig2): # 1/2
    if sig1!=0 and sig2!=0:
        return 0.5*(kl(mu1, sig1, mu2, sig2)+kl(mu2, sig2, mu1, sig1))
    else:
        return 0
    
def df_entropy(sig):  
    return 0.5*np.log2(1e-8+2*pi*e*det(sig)**2)

def js(mu1, sig1, mu2, sig2):
    # sig=np.sqrt((sig1**2+sig2**2)/2)
    sig = (sig1 + sig2) / 2
    return df_entropy(sig)-0.5*(df_entropy(sig1)+df_entropy(sig2))

def hellinger(mu1, sig1, mu2, sig2):
    if (sig1!=0) or (sig2!=0):
        return 1-np.sqrt((2*sig1*sig2)/(sig1**2+sig2**2+1e-8))*np.exp(-0.25*(mu1-mu2)**2/(1e-8+sig1**2+sig2**2))
    else:
        print('both sigs are zero')
        return 0
    
def compare(X_1, X_2, ft_num):
    res_kl=np.zeros((ft_num, X_1.shape[1]))
    res_js=np.zeros((ft_num, X_1.shape[1]))
    res_he=np.zeros((ft_num, X_1.shape[1]))
    for j in range(X_1.shape[1]): # current for M times
        x_1=X_1[:,j]
        x_2=X_2[:,j]
        kl_div=[]
        js_div=[]
        he_div=[]
        for i in range(ft_num):
            mu1=x_1[i*2]
            sig1=x_1[i*2+1]
            mu2=x_2[i*2]
            sig2=x_2[i*2+1]
            kl_div.append(symkl(mu1, sig1, mu2, sig2))
            js_div.append(js(mu1, sig1, mu2, sig2))
            he_div.append(hellinger(mu1, sig1, mu2, sig2))
       
        res_kl[:,j]=np.array(kl_div)
        res_js[:,j]=np.array(js_div)
        res_he[:,j]=np.array(he_div)
    return res_kl, res_js, res_he

def prob_compare(df, tl, x, kk):
    ttt = df[df['label']==tl]
    xx=np.zeros((kk,len(ft_name)))
    for i, name in enumerate(ft_name):
        xx[:kk,i]=ttt[name][:kk]
    aaa=[]
    for i in range(len(xx)):
        kkk=happening_prob(xx[i,:], x)
        ppp=0
        for i in range(len(kkk)):
            ppp=np.log(1e-8+kkk[i])+ppp
        aaa.append(ppp)
    return aaa

def prob_compare_1(x_fk, x_true, kk):
    xx=np.zeros((kk,len(ft_name)))
    for i, name in enumerate(ft_name):
        xx[:kk,i]=np.random.randn(kk)*x_fk[2*i+1]+x_fk[2*i]
        
    aaa=[]
    for i in range(len(xx)):
        kkk=happening_prob(xx[i,:], x_true)
        ppp=0
        for i in range(len(kkk)):
            ppp=np.log(1e-8+kkk[i])+ppp
        aaa.append(ppp)
    return aaa

def ppp(ind):
    dd=np.zeros((ind.shape[0],))
    for j in range(ind.shape[1]):
        ind_=ind[:,j]
        for i, e in enumerate(ind_):
            dd[int(e)]=dd[int(e)]+i
    return dd  

def kkk(ind, topp):
    #dd=np.zeros((ind.shape[1],))
    dd=[]
    for j in range(ind.shape[1]):
        ind_=ind[:,j]
        for i, e in enumerate(ind_): 
            if i<topp:
                dd.append(e)
    rr={}
    res=list(set(dd))
    for i in range(len(res)):
        rr[str(res[i])]=sum(dd==res[i])/len(dd)
    return rr, res   

def align_dict(dict1, dict2):
    k1=np.array(list(dict1.keys()))
    k2=np.array(list(dict2.keys()))
    v1=np.array(list(dict1.values()))
    v2=np.array(list(dict2.values()))
    pp=np.zeros((len(k1),))
    for i in range(len(k1)):
        kk=k1[i]
        ii=np.where(k2==kk)[0]
        if len(ii):
            pp[i]=v2[ii]      
    return pp   

def sort_dict(dict2):
    k2=np.array(list(dict2.keys()))
    v2=np.array(list(dict2.values()))
    indd=np.argsort(-1*v2)
    log_={}
    for i in indd:
        log_[k2[i]]=v2[i]
    return log_

