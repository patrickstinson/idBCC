import numpy as np

from scipy.special import factorial, factorial2, comb, softmax

def onehot(a,k):
    
    if np.isscalar(a):
        b = np.zeros(k)
        b[a]=1
    else:
        b = np.zeros((a.size, k))
        b[np.arange(a.size), a] = 1
    return b

def sigm(x):
    if np.isscalar(x):
        if x>=0:
            return 1/(1+np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    else:
        nxx = x.size
        sigmx = np.zeros(nxx)
        sigmx[x>=0] = 1/(1+np.exp(-x[x>=0]))
        sigmx[x<0] = np.exp(x[x<0])/(1+np.exp(x[x<0]))
        return sigmx
    
def pois(k,lam):
    return lam**k*np.exp(-lam)/np.math.factorial(k)

def argmax_with_ties(counts):
    return np.random.permutation(np.where(counts==counts.max())[0])[0]


def get_deriv_coeffs(n):

    D = np.zeros((n+1,n+1))
    for i in np.arange(0,n):    
        D[i,i]=i+1
        D[i+1,i] = -(i+1)
    D[-1,-1] = n+1
    
    coeffs = np.zeros((n,n+1))
    coeffs[0,0] = 1
    for i in np.arange(1,n):
        coeffs[i] = np.dot(D,coeffs[i-1])
        
    return coeffs[1:]

def choose(n,k):
    return factorial(n)[:,None]/(factorial(n[:,None]-k[None,:])*factorial(k[None,:]))

def get_derivatives(x,n):
    foo = (-1)**np.arange(n+1)[None,:]*np.arange(1,n+2)[None,:]**n*\
    choose(np.arange(n+1),np.arange(n+1))*sigm(x)**np.arange(1,n+2)[:,None]
    
    foo[np.isinf(foo)]=0
    
    return (foo*np.tril(np.ones((n+1,n+1)))).sum()

def normal_moment(v,p):
    return np.sqrt(v)**p*factorial2(p-1)*(p%2==0)

def trunc_normal_moment(var,n):
    if n==0:
        return 1
    elif n%2==0:
        return normal_moment(var,n)
    else:
        m = 1/np.sqrt(2*np.pi)*2*np.sqrt(var)
        for k in np.arange(3,n+1,2):
            m *= (k-1)*var
        return m

def centered_trunc_normal_moment(var,n):
    foo = np.array([trunc_normal_moment(var,k) for k in (n-np.arange(n+1))])
    return (choose(n*np.ones(1),np.arange(n+1))*(foo)*\
        (-trunc_normal_moment(var,1))**np.arange(n+1)).sum()

def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def centered_trunc_normal_sum_expectation(var,n,k):
    
    def multinomial_coeff(n,ks):
        return factorial(n)/np.array([factorial(ee) for ee in ks]).prod()
    
    partitions_ = []
    for ele in partitions(n):
        if len(ele)<=k:
            partitions_.append(ele)
    
    foo = 0
    for ele in list(partitions_):
        foo += comb(k,len(ele))*multinomial_coeff(n,ele)*len(ele)*\
            np.array([centered_trunc_normal_moment(var,ee) for ee in ele]).prod()
    return foo


def get_t_logprobs(rated_idx,labels,U,Upos,Uneg,Vpos,Vneg):
    N = Vpos.shape[0]
    M,L = U.shape[:2]
    logprobs = np.zeros((N,M,L))
    for m in np.arange(M):
        for i in np.arange(rated_idx[m].size):
            n_idx = rated_idx[m][i]
            for l in np.arange(L):
                pi = softmax(U[m,l] + np.dot(Upos[m,l],Vpos[n_idx])+np.dot(Uneg[m,l],Vneg[n_idx]))
                logprobs[n_idx,m,l] += np.log(pi[labels[m][i]]+1e-8)
    return logprobs.sum(1)