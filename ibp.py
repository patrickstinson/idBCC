import numpy as np
from scipy.special import softmax, logsumexp
from utils import *
from samplers import *


def sample_Z(rater_idx,item_labels,rated_idx,labels,U,Z,t,alpha_ibp_pos,alpha_ibp_neg,var,Knew_max=1,ndeg=5):
    eps=1e-8
    
    centered_trunc_moments = np.zeros((ndeg,Knew_max+1))
    for nn in np.arange(ndeg):
        for k in np.arange(1,Knew_max+1):
            centered_trunc_moments[nn,k] = centered_trunc_normal_sum_expectation(var,nn,k)/factorial(nn)
    
    trunc_mean = trunc_normal_moment(var,1)
    
    coeffs = get_deriv_coeffs(ndeg)
    
    
    N,K = Z.shape
    M,L = U.shape[:2]
    K -= 1
    
    for n in np.arange(N):
        m_idx = rater_idx[n]
        zero = np.zeros(K).astype(bool)
        for k in np.random.permutation(np.arange(1,K+1)):
            if (Z[:,k].sum()-Z[n,k])>0:
                
                c_0 = (U[m_idx][:,t[n]]*Z[n][None,None,:]).sum(-1)-U[m_idx][:,t[n],:,k]*Z[n,k]
                c_1 = c_0 + U[m_idx][:,t[n],:,k]

                pi_0 = softmax(c_0+eps,-1)
                pi_1 = softmax(c_1+eps,-1)
                
                log_pz1 = np.log(pi_1[:,item_labels[n]]).sum()
                log_pz0 = np.log(pi_0[:,item_labels[n]]).sum()
                
                if U[0,0,0,k]>0:
                    alpha_ibp = alpha_ibp_pos
                    KK = (U[0,0,0,1:]>0).sum()
                else:
                    alpha_ibp = alpha_ibp_neg
                    KK = (U[0,0,0,1:]==0).sum()
                
                prior1 = (Z[:,k].sum()-Z[n,k]+alpha_ibp/KK)/(N+alpha_ibp/KK)
                pz1 = sigm(log_pz1-log_pz0+np.log(prior1)-np.log(1-prior1))

                Z[n,k] = np.where(np.random.multinomial(1,np.array([1-pz1,pz1])))[0][0]
        
            else:
                zero[k-1] = 1

        Z[n,1:][zero]=0
        mu = (U[m_idx,t[n]]*Z[n][None,:]).sum(-1)
        l=t[n]
        
        signs = np.random.permutation(2)
        for s in np.arange(2):
            expectations = np.zeros((m_idx.size,L,Knew_max+1))
            pK_new = np.zeros(Knew_max+1)
            logpx = np.zeros(Knew_max+1)
            
            if signs[s]==1:
                sign='pos'
                for ll in np.arange(L):
                    expectations[:,:,0] = softmax(mu,1)
                    for k in np.arange(1,Knew_max+1): 
                        if l==ll:
                            muu = mu[:,l]-np.log(np.exp(mu).sum(-1)-np.exp(mu[:,l]))+k*trunc_normal_moment(var,1)
                            derivatives_vec=(coeffs[None,:,:]*sigm(muu)[:,None,None]**np.arange(1,ndeg+2)[None,None,:]).sum(-1)
                            expectations[:,ll,k] = sigm(muu)+(centered_trunc_moments[1:,k][None,:]*derivatives_vec).sum(-1)
                        else:
                            m = np.exp(mu).sum(-1)-np.exp(mu[:,l])
                            muu = np.log(m)-mu[:,l]-k*trunc_normal_moment(var,1)
                            derivatives_vec=(coeffs[None,:,:]*sigm(muu)[:,None,None]**np.arange(1,ndeg+2)[None,None,:]).sum(-1)
                            expectation = sigm(muu)+((-1)**np.arange(1,ndeg)*centered_trunc_moments[1:,k][None,:]*derivatives_vec).sum(-1)
                            expectations[:,ll,k] = np.exp(mu[:,ll])/m*expectation
                            
            else:
                sign = 'neg'
                for ll in np.arange(L):
                    expectations[:,:,0] = softmax(mu,1)
                    
                    for k in np.arange(1,Knew_max+1):
                        if l==ll:
                            muu = mu[:,l]-np.log(np.exp(mu).sum(-1)-np.exp(mu[:,l]))-k*trunc_normal_moment(var,1)
                            derivatives_vec=(coeffs[None,:,:]*sigm(muu)[:,None,None]**np.arange(1,ndeg+2)[None,None,:]).sum(-1)
                            expectations[:,ll,k] = sigm(muu)+((-1)**np.arange(1,ndeg)*centered_trunc_moments[1:,k][None,:]*derivatives_vec).sum(-1)
                        else:
                            m = np.exp(mu).sum(-1)-np.exp(mu[:,l])
                            muu = np.log(m)-mu[:,l]+k*trunc_normal_moment(var,1)
                            derivatives_vec=(coeffs[None,:,:]*sigm(muu)[:,None,None]**np.arange(1,ndeg+2)[None,None,:]).sum(-1)
                            expectation = sigm(muu)+(centered_trunc_moments[1:,k][None,:]*derivatives_vec).sum(-1)
                            expectations[:,ll,k] = np.exp(mu[:,ll])/m*expectation
                            
            
            for k in np.arange(Knew_max+1):
                logpx[k] = np.log((onehot(item_labels[n],L)*expectations[:,:,k]).sum(-1)+eps).sum()
                pK_new[k] = logpx[k] + np.log(pois(k,alpha_ibp_neg/N))
                
            Knew = np.where(np.random.multinomial(1,softmax(pK_new)))[0][0]

            if Knew>0:
                Z = np.concatenate([Z,np.tile(onehot(n,N),(Knew,1)).T],1)
                U = np.concatenate([U,np.zeros((M,L,L,Knew))],-1)

                for k in np.arange(K+1,K+Knew+1):
                    U[:,:,:,k] = sample_Uk(rated_idx,labels,U,Z,t,k,sign,lamba=1/var)
                K += Knew
                
                mu = (U[m_idx,t[n]]*Z[n][None,:]).sum(-1)
    
    keep = Z.sum(0)>0
    return Z[:,keep],U[:,:,:,keep]