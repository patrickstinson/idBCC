import argparse

import numpy as np
from scipy.special import softmax, logsumexp

from utils import *
from ibp import *
from samplers import *


def idbcc(argparser):
    
    args = argparser.parse_args()
    dataset = args.dataset
    seed = args.seed
    begin_sampling_t = args.begin_sampling_t
    begin_sampling_deps = args.begin_sampling_deps
    niters = args.niters
    burnin = args.burnin
    subsample = args.subsample
    alpha_ibp_pos = args.alpha_ibp_pos
    alpha_ibp_neg = args.alpha_ibp_neg
    ntaylor = args.ntaylor
    var = args.var
    Knew_max = args.Knew_max
    
    data = np.loadtxt('datasets/'+dataset+'/label.csv',skiprows=1,delimiter=',').astype(int)
    foo = np.loadtxt('datasets/'+dataset+'/truth.csv',skiprows=1,delimiter=',').astype(int)

    M = data[:,1].max()+1
    N = data[:,0].max()+1
    L = data[:,-1].max()+1
    
    truth = np.zeros(N)
    truth[foo[:,0]]=foo[:,-1]

    rated_idx = [np.array(data[data[:,1]==m][:,0]) for m in np.arange(M)]
    labels = [np.array(data[data[:,1]==m][:,-1]) for m in np.arange(M)]

    rater_idx = [np.array(data[data[:,0]==n][:,1]) for n in np.arange(N)]
    item_labels = [np.array(data[data[:,0]==n][:,-1]) for n in np.arange(N)]
    
    np.random.seed(seed)


    U = np.repeat((2.5*np.eye(L))[None,:,:],M,axis=0)
    Upos = np.zeros((M,L,L,0))
    Uneg = np.zeros((M,L,L,0))

    Vpos=np.zeros((N,0))
    Vneg=np.zeros((N,0))

    Kpos = 0
    Kneg = 0

    t = np.array([argmax_with_ties(np.bincount(data[data[:,0]==i][:,-1],minlength=L)) for i in np.arange(N)])

    UU = np.concatenate([np.expand_dims(U,-1),Upos,Uneg],-1)
    VV = np.concatenate([np.ones((N,1)),Vpos,Vneg],-1)

    ts = np.zeros((niters,N,L))

    for it in np.arange(niters):

        U = sample_Uk(rated_idx,labels,UU,VV,t,k=0,sign=None,lamba=1/var)
        UU = np.concatenate([np.expand_dims(U,-1),Upos,Uneg],-1)
        for k in np.arange(1,Kpos+1):
            Upos[:,:,:,k-1] = sample_Uk(rated_idx,labels,UU,VV,t,k=k,sign='pos',lamba=1/var)
            UU = np.concatenate([np.expand_dims(U,-1),Upos,Uneg],-1)

        for k in np.arange(Kpos+1,Kpos+Kneg+1):
            Uneg[:,:,:,k-Kpos-1] = sample_Uk(rated_idx,labels,UU,VV,t,k=k,sign='neg',lamba=1/var)
            UU = np.concatenate([np.expand_dims(U,-1),Upos,Uneg],-1)

        if it>begin_sampling_deps:
            VV,UU = sample_Z(rater_idx,item_labels,rated_idx,labels,UU.copy(),VV.copy(),t,
                            alpha_ibp_pos,alpha_ibp_neg,var,Knew_max,ntaylor)   

            U = UU[:,:,:,0]
            pos_idx = UU[0,0,0,1:]>0
            neg_idx = UU[0,0,0,1:]==0
            Upos = UU[:,:,:,1:][:,:,:,pos_idx]
            Vpos = VV[:,1:][:,pos_idx]
            Uneg = UU[:,:,:,1:][:,:,:,neg_idx]
            Vneg = VV[:,1:][:,neg_idx]

            Kpos = pos_idx.sum()
            Kneg = neg_idx.sum()
        
            U = sample_Uk(rated_idx,labels,UU,VV,t,k=0,sign=None)
            UU = np.concatenate([np.expand_dims(U,-1),Upos,Uneg],-1)
            for k in np.arange(1,Kpos+1):
                Upos[:,:,:,k-1] = sample_Uk(rated_idx,labels,UU,VV,t,k=k,sign='pos',lamba=1/var)
                UU = np.concatenate([np.expand_dims(U,-1),Upos,Uneg],-1)

            for k in np.arange(Kpos+1,Kpos+Kneg+1):
                Uneg[:,:,:,k-Kpos-1] = sample_Uk(rated_idx,labels,UU,VV,t,k=k,sign='neg',lamba=1/var)
                UU = np.concatenate([np.expand_dims(U,-1),Upos,Uneg],-1)

        var = 1/np.random.gamma(1e-3+.5*(UU>0).sum(),1/(1e-3+.5*((UU[UU>0])**2).sum()))
        
        t_logprobs = get_t_logprobs(rated_idx,labels,U,Upos,Uneg,Vpos,Vneg)
        if it>begin_sampling_t:
            t = np.array([np.where(np.random.multinomial(1,softmax(t_logprobs[i],-1)))[0][0] for i in np.arange(N)])
            ts[it] = onehot(t,L)
            if it>burnin:
                running_est = ts[burnin:it][::subsample].mean(0).argmax(-1)
        else:
            tt = np.array([np.where(np.random.multinomial(1,softmax(t_logprobs[i],-1)))[0][0] for i in np.arange(N)])
        
        
        if it>burnin and (it+1)%100==0:
            print('iteration {}/{}, estimate accuracy {}'.format(it+1,niters,(running_est==truth).mean()))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset',choices=['face','dog','ms','web','bird'])
    p.add_argument('-Knew_max', dest='Knew_max', action='store',type=int, default=1)
    p.add_argument('-ntaylor', dest='ntaylor', action='store',type=int, default=5)
    p.add_argument('-var', dest='var', action='store',type=int, default=5)
    p.add_argument('-seed', dest='seed', action='store',type=int, default=1)
    p.add_argument('-begin_sampling_t', dest='begin_sampling_t', action='store',type=int, default=10)
    p.add_argument('-begin_sampling_deps', dest='begin_sampling_deps', action='store',type=int, default=20)
    p.add_argument('-niters', dest='niters', action='store',type=int, default=1000)
    p.add_argument('-burnin', dest='burnin', action='store',type=int, default=100)
    p.add_argument('-subsample', dest='subsample', action='store',type=int, default=10)
    p.add_argument('-alpha_ibp_pos', dest='alpha_ibp_pos', action='store',type=float, default=1e-3)
    p.add_argument('-alpha_ibp_neg', dest='alpha_ibp_neg', action='store',type=float, default=1e-3)

    idbcc(p)
    